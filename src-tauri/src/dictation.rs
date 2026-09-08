// ============================================================
// QazTriber — диктовка в любом месте (push-to-talk).
// Хоткей (global-hotkey) → захват микрофона (cpal) → WAV →
// POST /api/dictate (sidecar) → вставка текста в активное поле
// (enigo: печать / вставка из буфера; arboard: буфер обмена).
// Весь runtime живёт в Rust — WebView не участвует, поэтому
// фоновый троттлинг/скрытие окна на запись не влияют.
// ============================================================

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut};

// В dev-сборке backend запускают на :8000 (vite-прокси), в релизе sidecar живёт на :8765.
const BACKEND_URL: &str = if cfg!(debug_assertions) {
    "http://127.0.0.1:8000"
} else {
    "http://127.0.0.1:8765"
};
const MIN_RECORD_SECS: f64 = 0.3;
const MAX_RECORD_SECS: u64 = 120;
const DICTATE_TIMEOUT_SECS: u64 = 400;
const WARMUP_TIMEOUT_SECS: u64 = 190;

const MODELS: [&str; 2] = ["220m", "600m"];
const LANGUAGES: [&str; 3] = ["kazakh", "russian", "mixed"];
const TRIGGERS: [&str; 2] = ["hold", "toggle"];
const INSERT_MODES: [&str; 3] = ["type", "paste", "clipboard"];

#[derive(Clone, Serialize)]
struct DictationEvent {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Clone)]
pub struct Config {
    pub enabled: bool,
    pub hotkey: String,
    pub trigger: String,
    pub insert_mode: String,
    pub model: String,
    pub language: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: false,
            hotkey: "alt+shift+KeyD".into(),
            trigger: "hold".into(),
            insert_mode: "type".into(),
            model: "220m".into(),
            language: "mixed".into(),
        }
    }
}

/// Активный захват. `cpal::Stream` здесь не хранится: на Windows он `!Send`
/// (WASAPI/COM-указатели), поэтому живёт в выделенном потоке захвата, а
/// останавливается через канал `stop` (drop отправителя тоже останавливает).
struct Recording {
    buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    started: Instant,
    generation: usize,
    stop: mpsc::Sender<()>,
}

/// Готовый захват, возвращённый потоком записи.
struct CaptureHandle {
    buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    stop: mpsc::Sender<()>,
}

pub struct TrayHandles {
    pub dictation_item: tauri::menu::MenuItem<tauri::Wry>,
}

pub struct DictationManager {
    app: AppHandle,
    config: Mutex<Config>,
    // GlobalHotKeyManager на Windows !Send — им владеет плагин
    // global-shortcut (регистрация идёт в main thread), здесь только
    // текущая активная комбинация для матчинга событий и apply().
    registered_hotkey: Mutex<Option<Shortcut>>,
    recording: Mutex<Option<Recording>>,
    processing: AtomicBool,
    generation: AtomicUsize,
    mic_error_reported: Arc<AtomicBool>,
}

fn sanitize(value: &str, allowed: &[&str], fallback: &str) -> String {
    if allowed.contains(&value) {
        value.into()
    } else {
        fallback.into()
    }
}

impl DictationManager {
    pub fn new(app: AppHandle) -> Self {
        Self {
            app,
            config: Mutex::new(Config::default()),
            registered_hotkey: Mutex::new(None),
            recording: Mutex::new(None),
            processing: AtomicBool::new(false),
            generation: AtomicUsize::new(0),
            mic_error_reported: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Событие хоткея от плагина global-shortcut (press/release).
    pub fn handle_shortcut(&self, shortcut: &Shortcut, pressed: bool) {
        {
            let active = self.registered_hotkey.lock().unwrap();
            if active.as_ref() != Some(shortcut) {
                return;
            }
        }
        if pressed {
            self.on_press();
        } else {
            self.on_release();
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.lock().unwrap().enabled
    }

    pub fn status(&self) -> serde_json::Value {
        let config = self.config.lock().unwrap();
        serde_json::json!({
            "enabled": config.enabled,
            "hotkey": config.hotkey,
            "trigger": config.trigger,
            "insertMode": config.insert_mode,
            "model": config.model,
            "language": config.language,
            "recording": self.recording.lock().unwrap().is_some(),
            "processing": self.processing.load(Ordering::SeqCst),
        })
    }

    /// Применяет новую конфигурацию: (пере)регистрирует хоткей, обновляет трей,
    /// запускает warmup модели. Ошибки регистрации хоткея возвращаются вызывающему.
    pub fn apply(&self, config: Config) -> Result<(), String> {
        if !config.enabled && self.recording.lock().unwrap().take().is_some() {
            // Поток захвата завершится по drop отправителя stop-канала.
            self.emit("recording-stopped", None, None);
        }

        let hotkey_str = config.hotkey.clone();
        let enabled = config.enabled;

        // Регистрация хоткея через плагин global-shortcut: register/unregister
        // выполняются в main thread (на Windows GlobalHotKeyManager !Send).
        // Новый регистрируем ДО снятия старого: при неудаче прежний хоткей
        // продолжает работать; при неизменной комбинации нет цикла
        // unregister/register — системный захват клавиши не мигает.
        let mut registered = self.registered_hotkey.lock().unwrap();
        if enabled {
            let hotkey: Shortcut = hotkey_str
                .parse()
                .map_err(|e| format!("[hotkey_register_failed] Некорректная комбинация «{hotkey_str}»: {e}"))?;
            if registered.as_ref() != Some(&hotkey) {
                self.app
                    .global_shortcut()
                    .register(hotkey)
                    .map_err(|e| format!("[hotkey_register_failed] Комбинация «{hotkey_str}» занята другой программой: {e}"))?;
                if let Some(old) = registered.take() {
                    let _ = self.app.global_shortcut().unregister(old);
                }
                *registered = Some(hotkey);
            }
        } else if let Some(old) = registered.take() {
            let _ = self.app.global_shortcut().unregister(old);
        }
        drop(registered);

        *self.config.lock().unwrap() = config.clone();

        if enabled {
            self.spawn_warmup(&config.model);
        }
        self.update_tray();
        self.emit("enabled-changed", None, None);
        Ok(())
    }

    /// Переключение из трея.
    pub fn toggle_enabled(&self) {
        let mut config = self.config.lock().unwrap().clone();
        config.enabled = !config.enabled;
        if let Err(error) = self.apply(config) {
            // Вернуть прежнее состояние и показать ошибку в тосте.
            self.update_tray();
            self.emit("error", Some("hotkey_register_failed"), Some(&error));
        }
    }

    fn update_tray(&self) {
        let config = self.config.lock().unwrap();
        let recording = self.recording.lock().unwrap().is_some();
        let item_text = if recording {
            "● Идёт запись…".to_string()
        } else if config.enabled {
            format!("Диктовка: включена ({})", format_hotkey(&config.hotkey))
        } else {
            "Диктовка: выключена".to_string()
        };
        let tooltip = if recording {
            "QazTriber — идёт запись".to_string()
        } else if config.enabled {
            "QazTriber — диктовка включена".to_string()
        } else {
            "QazTriber".to_string()
        };
        let app = self.app.clone();
        std::thread::spawn(move || {
            if let Some(handles) = app.try_state::<TrayHandles>() {
                let item = handles.dictation_item.clone();
                let _ = app.run_on_main_thread(move || {
                    let _ = item.set_text(&item_text);
                });
            }
            if let Some(tray) = app.tray_by_id("qaztriber-tray") {
                let _ = tray.set_tooltip(Some(&tooltip));
                #[cfg(target_os = "macos")]
                {
                    let _ = tray.set_title(if recording { Some("●") } else { None });
                }
            }
        });
    }

    fn emit(&self, kind: &str, error_code: Option<&str>, detail: Option<&str>) {
        let _ = self.app.emit(
            "dictation-event",
            DictationEvent {
                kind: kind.into(),
                error_code: error_code.map(String::from),
                detail: detail.map(String::from),
            },
        );
    }

    fn spawn_warmup(&self, model: &str) {
        let model = model.to_string();
        std::thread::spawn(move || {
            let client = match reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(WARMUP_TIMEOUT_SECS))
                .build()
            {
                Ok(c) => c,
                Err(_) => return,
            };
            let url = format!("{BACKEND_URL}/api/dictate/warmup?model={model}");
            // Одна повторная попытка: на старте приложения sidecar мог ещё не подняться.
            for attempt in 0..2 {
                if attempt == 1 {
                    std::thread::sleep(Duration::from_secs(5));
                }
                match client.post(&url).send() {
                    Ok(response) if response.status().is_success() => return,
                    _ => continue,
                }
            }
            // Не критично: первая диктовка просто загрузит модель на месте.
        });
    }

    fn on_press(&self) {
        if self.processing.load(Ordering::SeqCst) {
            return;
        }
        if self.recording.lock().unwrap().is_some() {
            // toggle: второй press останавливает.
            if self.config.lock().unwrap().trigger == "toggle" {
                self.stop_and_process();
            }
            return;
        }
        match self.start_capture() {
            Ok(handle) => {
                let mut recording = self.recording.lock().unwrap();
                if recording.is_some() {
                    return; // двойной press: вторую попытку бросаем
                }
                let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
                self.mic_error_reported.store(false, Ordering::SeqCst);
                *recording = Some(Recording {
                    buffer: handle.buffer,
                    sample_rate: handle.sample_rate,
                    channels: handle.channels,
                    started: Instant::now(),
                    generation,
                    stop: handle.stop,
                });
                drop(recording);
                play_sound("start");
                self.update_tray();
                self.emit("recording-started", None, None);
                // Watchdog: автозавершение при слишком длинном удержании.
                let app = self.app.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_secs(MAX_RECORD_SECS));
                    let manager = app.state::<DictationManager>();
                    let active = manager.recording.lock().unwrap().as_ref().map(|r| r.generation);
                    if active == Some(generation) {
                        manager.stop_and_process();
                    }
                });
            }
            Err(error) => {
                self.emit("error", Some("mic_error"), Some(&error));
            }
        }
    }

    fn on_release(&self) {
        if self.config.lock().unwrap().trigger == "hold" {
            self.stop_and_process();
        }
    }

    fn stop_and_process(&self) {
        let recording = self.recording.lock().unwrap().take();
        let Some(recording) = recording else { return };
        let _ = recording.stop.send(()); // поток захвата завершится и уронит Stream
        play_sound("stop");
        self.update_tray();
        self.emit("recording-stopped", None, None);

        let elapsed = recording.started.elapsed();
        if elapsed.as_secs_f64() < MIN_RECORD_SECS {
            self.emit("ignored", Some("too_short"), None);
            return;
        }

        let mut samples = recording.buffer.lock().unwrap().clone();
        let channels = recording.channels.max(1) as usize;
        if channels > 1 {
            // Даунмикс в моно.
            let mono: Vec<f32> = samples
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect();
            samples = mono;
        }

        let config = self.config.lock().unwrap().clone();
        self.processing.store(true, Ordering::SeqCst);
        self.emit("processing", None, None);

        let app = self.app.clone();
        std::thread::spawn(move || {
            let manager = app.state::<DictationManager>();
            manager.process(samples, recording.sample_rate, config);
        });
    }

    fn start_capture(&self) -> Result<CaptureHandle, String> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let mic_reported = self.mic_error_reported.clone();
        let app = self.app.clone();

        // Stream живёт в этом потоке до сигнала остановки: на Windows
        // cpal::Stream — !Send, его нельзя унести в managed state.
        std::thread::spawn(move || {
            let host = cpal::default_host();
            let Some(device) = host.default_input_device() else {
                let _ = ready_tx.send(Err("Микрофон не найден.".to_string()));
                return;
            };
            let supported = match device.default_input_config() {
                Ok(config) => config,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("Микрофон недоступен: {e}")));
                    return;
                }
            };
            let config: cpal::StreamConfig = supported.clone().into();

            let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
            let stream_buffer = buffer.clone();
            let max_samples = (MAX_RECORD_SECS as u32) * supported.sample_rate();

            let error_callback = move |_error| {
                if !mic_reported.swap(true, Ordering::SeqCst) {
                    let _ = app.emit(
                        "dictation-event",
                        DictationEvent {
                            kind: "error".into(),
                            error_code: Some("mic_error".into()),
                            detail: Some("Ошибка захвата микрофона (проверьте доступ в настройках системы).".into()),
                        },
                    );
                }
            };

            let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = stream_buffer.lock().unwrap();
                if buf.len() < max_samples as usize {
                    buf.extend_from_slice(data);
                }
            };

            let stream = match device.build_input_stream::<f32, _, _>(
                &config,
                data_callback,
                error_callback,
                None,
            ) {
                Ok(stream) => stream,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("Не удалось начать запись с микрофона: {e}")));
                    return;
                }
            };
            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(format!("Не удалось начать запись с микрофона: {e}")));
                return;
            }

            let _ = ready_tx.send(Ok((supported.sample_rate(), supported.channels(), buffer)));
            // Спим до сигнала остановки (или drop отправителя) — Stream умрёт здесь же.
            let _ = stop_rx.recv();
        });

        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok((sample_rate, channels, buffer))) => Ok(CaptureHandle {
                buffer,
                sample_rate,
                channels,
                stop: stop_tx,
            }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err("Микрофон не ответил вовремя (5 с).".to_string()),
        }
    }

    fn process(&self, samples: Vec<f32>, sample_rate: u32, config: Config) {
        let result = self.transcribe_and_insert(samples, sample_rate, &config);
        self.processing.store(false, Ordering::SeqCst);
        match result {
            Ok(()) => self.emit("inserted", None, None),
            Err((code, detail)) => self.emit("error", Some(&code), Some(&detail)),
        }
    }

    fn transcribe_and_insert(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        config: &Config,
    ) -> Result<(), (String, String)> {
        if samples.is_empty() {
            return Err(("mic_error".into(), "Микрофон не записал звук (проверьте доступ).".into()));
        }

        let wav = encode_wav(&samples, sample_rate).map_err(|e| ("wav_error".into(), e))?;

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(DICTATE_TIMEOUT_SECS))
            .build()
            .map_err(|e| ("network_error".into(), e.to_string()))?;
        let url = format!(
            "{BACKEND_URL}/api/dictate?model={}&expected_language={}",
            config.model, config.language
        );
        let response = client
            .post(&url)
            .header("Content-Type", "audio/wav")
            .body(wav)
            .send()
            .map_err(|e| ("network_error".into(), format!("Нет связи с локальным сервисом: {e}")))?;

        let status = response.status();
        let body: serde_json::Value = response
            .json()
            .map_err(|e| ("network_error".into(), format!("Некорректный ответ сервиса: {e}")))?;
        if !status.is_success() {
            let detail = body["detail"].as_str().unwrap_or("Ошибка транскрипции").to_string();
            return Err(("transcription_failed".into(), detail));
        }

        let mut text = body["text"].as_str().unwrap_or("").trim().to_string();
        if text.is_empty() {
            return Err(("empty_result".into(), "Речь не распознана — попробуйте ещё раз.".into()));
        }
        // Стандартное поведение диктовки: первая буква — заглавная.
        if let Some(first) = text.get_mut(0..1) {
            let upper = first.to_uppercase();
            text.replace_range(0..1, &upper);
        }

        if let Err((code, detail)) = insert_text(&config.insert_mode, &text) {
            // Fallback: не теряем текст — копируем в буфер обмена.
            if config.insert_mode != "clipboard" {
                if copy_to_clipboard(&text).is_ok() {
                    return Err((code, format!("{detail}. Текст скопирован в буфер обмена.")));
                }
            }
            return Err((code, detail));
        }
        Ok(())
    }
}

fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).map_err(|e| e.to_string())?;
        for sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            writer
                .write_sample((clamped * i16::MAX as f32) as i16)
                .map_err(|e| e.to_string())?;
        }
        writer.finalize().map_err(|e| e.to_string())?;
    }
    Ok(cursor.into_inner())
}

/// macOS: право Accessibility нужно, чтобы отправлять клавиатурные события в чужие приложения.
#[cfg(target_os = "macos")]
pub fn has_accessibility(prompt: bool) -> bool {
    use core_foundation::base::TCFType;
    use core_foundation::boolean::CFBoolean;
    use core_foundation::dictionary::{CFDictionary, CFDictionaryRef};
    use core_foundation::string::{CFString, CFStringRef};

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        static kAXTrustedCheckOptionPrompt: CFStringRef;
        fn AXIsProcessTrustedWithOptions(options: CFDictionaryRef) -> bool;
    }
    let key = unsafe { CFString::wrap_under_create_rule(kAXTrustedCheckOptionPrompt) };
    let options = CFDictionary::from_CFType_pairs(&[(key, CFBoolean::from(prompt))]);
    unsafe { AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef()) }
}

#[cfg(not(target_os = "macos"))]
pub fn has_accessibility(_prompt: bool) -> bool {
    true
}

fn copy_to_clipboard(text: &str) -> Result<(), String> {
    arboard::Clipboard::new()
        .and_then(|mut clipboard| clipboard.set_text(text.to_string()))
        .map_err(|e| e.to_string())
}

/// Мягкий звуковой сигнал старта/стопа записи — системные звуки macOS
/// (afplay всегда установлен). Fire-and-forget: spawn не блокирует поток хоткея.
fn play_sound(kind: &str) {
    #[cfg(target_os = "macos")]
    {
        let path = if kind == "start" {
            "/System/Library/Sounds/Tink.aiff"
        } else {
            "/System/Library/Sounds/Pop.aiff"
        };
        let _ = std::process::Command::new("afplay")
            .args(["-v", "0.4", path])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
    #[cfg(not(target_os = "macos"))]
    let _ = kind;
}

fn insert_text(mode: &str, text: &str) -> Result<(), (String, String)> {
    if mode == "clipboard" {
        return copy_to_clipboard(text).map_err(|e| ("clipboard_error".into(), e));
    }

    #[cfg(target_os = "macos")]
    if !has_accessibility(false) {
        return Err((
            "accessibility_denied".into(),
            "Нужно право Accessibility: Системные настройки → Конфиденциальность → Универсальный доступ."
                .into(),
        ));
    }

    let mut enigo = enigo::Enigo::new(&enigo::Settings::default())
        .map_err(|e| ("insert_error".into(), e.to_string()))?;
    if mode == "paste" {
        copy_to_clipboard(text).map_err(|e| ("clipboard_error".into(), e))?;
        let modifier = if cfg!(target_os = "macos") {
            enigo::Key::Meta
        } else {
            enigo::Key::Control
        };
        use enigo::{Direction, Keyboard};
        enigo
            .key(modifier, Direction::Press)
            .and_then(|_| enigo.key(enigo::Key::Unicode('v'), Direction::Click))
            .and_then(|_| enigo.key(modifier, Direction::Release))
            .map_err(|e| ("insert_error".into(), e.to_string()))?;
    } else {
        use enigo::Keyboard;
        enigo
            .text(text)
            .map_err(|e| ("insert_error".into(), e.to_string()))?;
    }
    Ok(())
}

fn format_hotkey(hotkey: &str) -> String {
    // "alt+shift+KeyD" → "Alt+Shift+D" для отображения в трее.
    hotkey
        .split('+')
        .map(|part| {
            let part = part.trim();
            if let Some(key) = part.strip_prefix("Key") {
                key.to_uppercase()
            } else if let Some(key) = part.strip_prefix("Digit") {
                key.to_string()
            } else {
                match part.to_lowercase().as_str() {
                    "alt" => "Alt".into(),
                    "shift" => "Shift".into(),
                    "ctrl" | "control" => "Ctrl".into(),
                    "cmd" | "super" | "command" => "Cmd".into(),
                    other => {
                        let mut chars = other.chars();
                        match chars.next() {
                            Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                            None => String::new(),
                        }
                    }
                }
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}

// ------------------------------------------------------------------
// Tauri-команды (вызывает frontend)
// ------------------------------------------------------------------

#[tauri::command]
pub fn dictation_configure(
    app: AppHandle,
    enabled: bool,
    hotkey: String,
    trigger: String,
    insert_mode: String,
    model: String,
    language: String,
) -> Result<(), String> {
    let config = Config {
        enabled,
        hotkey: hotkey.trim().to_lowercase(),
        trigger: sanitize(&trigger, &TRIGGERS, "hold"),
        insert_mode: sanitize(&insert_mode, &INSERT_MODES, "type"),
        model: sanitize(&model, &MODELS, "220m"),
        language: sanitize(&language, &LANGUAGES, "mixed"),
    };
    app.state::<DictationManager>().apply(config)
}

#[tauri::command]
pub fn dictation_status(app: AppHandle) -> serde_json::Value {
    app.state::<DictationManager>().status()
}

#[tauri::command]
pub fn dictation_permissions() -> serde_json::Value {
    serde_json::json!({
        "accessibility": has_accessibility(false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_hotkey_strings_parse() {
        // JS KeyboardEvent.code → "модификаторы+Code" — формат должен парситься плагином.
        for combo in [
            "alt+shift+KeyD",
            "ctrl+shift+Space",
            "alt+Digit1",
            "cmd+KeyR",
            "F5",
            "alt+Comma",
        ] {
            assert!(
                combo.parse::<Shortcut>().is_ok(),
                "не распарсился хоткей: {combo}"
            );
        }
    }

    #[test]
    fn sanitize_falls_back() {
        assert_eq!(sanitize("220m", &MODELS, "220m"), "220m");
        assert_eq!(sanitize("hack", &MODELS, "220m"), "220m");
        assert_eq!(sanitize("hold", &TRIGGERS, "hold"), "hold");
        assert_eq!(sanitize("toggle", &TRIGGERS, "hold"), "toggle");
        assert_eq!(sanitize("paste", &INSERT_MODES, "type"), "paste");
    }

    #[test]
    fn format_hotkey_is_human_readable() {
        assert_eq!(format_hotkey("alt+shift+KeyD"), "Alt+Shift+D");
        assert_eq!(format_hotkey("ctrl+Digit1"), "Ctrl+1");
        assert_eq!(format_hotkey("F5"), "F5");
    }

    #[test]
    fn encode_wav_produces_valid_file() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_wav(&samples, 16000).expect("wav encode");
        let mut reader = hound::WavReader::new(Cursor::new(bytes)).expect("wav parse");
        assert_eq!(reader.spec().sample_rate, 16000);
        assert_eq!(reader.spec().channels, 1);
        let read: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(read.len(), samples.len());
        assert_eq!(read[1], (0.5 * i16::MAX as f32) as i16);
    }

    #[test]
    fn mono_downmix_math() {
        let stereo = vec![0.2f32, 0.4, -0.6, 0.2];
        let channels = 2usize;
        let mono: Vec<f32> = stereo
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        assert!(mono.iter().zip([0.3, -0.2]).all(|(a, b)| (a - b).abs() < 1e-6));
    }
}
