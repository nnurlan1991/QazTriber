use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{Emitter, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

mod dictation;

use dictation::{DictationManager, TrayHandles};

const BACKEND_HOST: &str = "127.0.0.1";
const BACKEND_PORT: u16 = 8765;
const DEV_URL: &str = "http://localhost:5173";
const BACKEND_STARTUP_TIMEOUT_SECS: u64 = 60;
const MAX_RESTART_ATTEMPTS: u32 = 5;
const MIN_UPTIME_FOR_RESET_SECS: u64 = 30;

type SharedChild = Arc<Mutex<Option<Child>>>;

#[derive(Clone, serde::Serialize)]
struct SidecarStatus {
    status: String,
    attempt: u32,
}

fn backend_is_ready() -> bool {
    TcpStream::connect((BACKEND_HOST, BACKEND_PORT)).is_ok()
}

fn wait_for_backend(timeout_secs: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    while Instant::now() < deadline {
        if backend_is_ready() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

fn spawn_sidecar_thread(
    app_handle: tauri::AppHandle,
    resource_path: PathBuf,
    shared_child: SharedChild,
) {
    std::thread::spawn(move || {
        let mut attempt: u32 = 0;
        loop {
            let mut command = Command::new(&resource_path);
            command
                .env("PYTHONUTF8", "1")
                .env("PYTHONIOENCODING", "utf-8")
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::process::CommandExt;
                // CREATE_NO_WINDOW (0x08000000) — предотвращает появление
                // консольного окна cmd.exe при спавне sidecar на Windows.
                const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                command.creation_flags(CREATE_NO_WINDOW);
            }
            let child = match command.spawn() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to spawn sidecar: {}", e);
                    attempt += 1;
                    if attempt > MAX_RESTART_ATTEMPTS {
                        let _ = app_handle.emit(
                            "sidecar-status",
                            SidecarStatus {
                                status: "failed".into(),
                                attempt,
                            },
                        );
                        return;
                    }
                    let delay = std::cmp::min(2u64.pow(attempt.min(6)), 60);
                    std::thread::sleep(Duration::from_secs(delay));
                    continue;
                }
            };

            // Share the child handle so the exit handler can kill it
            {
                let mut guard = shared_child.lock().unwrap();
                *guard = Some(child);
            }

            if !wait_for_backend(BACKEND_STARTUP_TIMEOUT_SECS) {
                // BUG C2: kill old process before respawn
                attempt += 1;
                {
                    let mut guard = shared_child.lock().unwrap();
                    if let Some(mut old) = guard.take() {
                        let _ = old.kill();
                        let _ = old.wait();
                    }
                }
                if attempt > MAX_RESTART_ATTEMPTS {
                    let _ = app_handle.emit(
                        "sidecar-status",
                        SidecarStatus {
                            status: "failed".into(),
                            attempt,
                        },
                    );
                    return;
                }
                let _ = app_handle.emit(
                    "sidecar-status",
                    SidecarStatus {
                        status: "restarting".into(),
                        attempt,
                    },
                );
                let delay = std::cmp::min(2u64.pow(attempt.min(6)), 60);
                std::thread::sleep(Duration::from_secs(delay));
                continue;
            }

            // BUG C3: track uptime — only reset attempt if sidecar ran long enough
            let started_at = Instant::now();
            let _ = app_handle.emit(
                "sidecar-status",
                SidecarStatus {
                    status: "connected".into(),
                    attempt,
                },
            );

            // BUG C2 + S1: improved monitoring loop
            loop {
                std::thread::sleep(Duration::from_millis(500));
                match shared_child.lock().unwrap().as_mut() {
                    Some(c) => match c.try_wait() {
                        Ok(Some(_)) => break,  // child exited
                        Ok(None) => continue,   // still running
                        Err(_) => {
                            // BUG S1: retry once before assuming exit
                            std::thread::sleep(Duration::from_millis(100));
                            match c.try_wait() {
                                Ok(Some(_)) => break,
                                Ok(None) => continue,
                                Err(_) => {
                                    // Still error — force kill and break
                                    let _ = c.kill();
                                    let _ = c.wait();
                                    break;
                                }
                            }
                        }
                    },
                    None => break, // child was taken by exit handler
                }
            }

            // BUG C3: only reset attempt if uptime was sufficient
            let uptime = started_at.elapsed();
            {
                let mut guard = shared_child.lock().unwrap();
                if let Some(mut old) = guard.take() {
                    let _ = old.kill();
                    let _ = old.wait();
                }
            }
            if uptime.as_secs() >= MIN_UPTIME_FOR_RESET_SECS {
                attempt = 0;
            } else {
                attempt += 1;
            }
            if attempt > MAX_RESTART_ATTEMPTS {
                let _ = app_handle.emit(
                    "sidecar-status",
                    SidecarStatus {
                        status: "failed".into(),
                        attempt,
                    },
                );
                return;
            }
            let _ = app_handle.emit(
                "sidecar-status",
                SidecarStatus {
                    status: "restarting".into(),
                    attempt,
                },
            );
            let delay = std::cmp::min(2u64.pow(attempt.min(6)), 60);
            std::thread::sleep(Duration::from_secs(delay));
        }
    });
}

async fn start_health_monitor(app_handle: tauri::AppHandle, shutdown: Arc<AtomicBool>) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return,
    };

    let mut consecutive_failures: u32 = 0;
    let mut was_unreachable = false;

    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }

        tokio::time::sleep(Duration::from_secs(5)).await;

        if shutdown.load(Ordering::SeqCst) {
            return;
        }

        let health_url = format!("http://{}:{}/api/health", BACKEND_HOST, BACKEND_PORT);
        let result = client.get(&health_url).send().await;

        match result {
            Ok(resp) if resp.status().is_success() => {
                consecutive_failures = 0;
                if was_unreachable {
                    was_unreachable = false;
                    let _ = app_handle.emit("sidecar-status", SidecarStatus {
                        status: "connected".into(),
                        attempt: 0,
                    });
                }
            }
            _ => {
                consecutive_failures += 1;
                if consecutive_failures >= 3 && !was_unreachable {
                    was_unreachable = true;
                    let _ = app_handle.emit("sidecar-status", SidecarStatus {
                        status: "unreachable".into(),
                        attempt: consecutive_failures,
                    });
                }
            }
        }
    }
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! This is QazTriber.", name)
}

/// Opens a directory in the native file manager (Finder on macOS, Explorer on Windows).
/// Returns true if the directory was opened (or already open), false if path is missing.
#[tauri::command]
fn open_folder(path: String) -> Result<bool, String> {
    use std::path::Path;
    use std::process::Command;

    let target = Path::new(&path);
    if !target.is_dir() {
        return Ok(false);
    }

    let result = if cfg!(target_os = "windows") {
        Command::new("explorer")
            .arg(&path)
            .spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open")
            .arg(&path)
            .spawn()
    } else {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
    };

    match result {
        Ok(_) => Ok(true),
        Err(e) => Err(format!("Не удалось открыть папку: {}", e)),
    }
}

/// Opens a URL in the default browser. WKWebView blocks window.open() for external
/// URLs, so we shell out to the OS default browser handler.
#[tauri::command]
fn open_url(url: String) -> Result<bool, String> {
    use std::process::Command;

    let result = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(["/C", "start", "", &url])
            .spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open")
            .arg(&url)
            .spawn()
    } else {
        Command::new("xdg-open")
            .arg(&url)
            .spawn()
    };

    match result {
        Ok(_) => Ok(true),
        Err(e) => Err(format!("Не удалось открыть URL: {}", e)),
    }
}

/// Saves text content to a file in the system Downloads directory and opens it
/// with the default application (TextEdit on macOS, Notepad on Windows).
/// Returns the saved file path on success.
#[tauri::command]
fn save_and_open_txt(content: String, filename: String) -> Result<String, String> {
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    // Resolve Downloads directory without external crates (dirs is config-zone blocked)
    let downloads_dir = if cfg!(target_os = "windows") {
        std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .map(|h| h.join("Downloads"))
            .map_err(|_| "USERPROFILE not set".to_string())?
    } else {
        std::env::var("HOME")
            .map(PathBuf::from)
            .map(|h| h.join("Downloads"))
            .map_err(|_| "HOME not set".to_string())?
    };

    fs::create_dir_all(&downloads_dir)
        .map_err(|e| format!("Не удалось создать папку: {}", e))?;

    // Sanitize filename — keep it simple, only alphanumerics + dash + .txt
    let safe_name = filename
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '.' { c } else { '_' })
        .collect::<String>();
    let file_path: PathBuf = downloads_dir.join(if safe_name.ends_with(".txt") {
        safe_name
    } else {
        format!("{}.txt", safe_name)
    });

    fs::write(&file_path, &content)
        .map_err(|e| format!("Не удалось сохранить файл: {}", e))?;

    let path_str = file_path.to_string_lossy().to_string();

    let result = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(["/C", "start", "", "", &path_str])
            .spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open")
            .arg(&path_str)
            .spawn()
    } else {
        Command::new("xdg-open")
            .arg(&path_str)
            .spawn()
    };

    match result {
        Ok(_) => Ok(path_str),
        Err(e) => Err(format!("Файл сохранён, но не удалось открыть: {}", e)),
    }
}

/// Трей-иконка: статус диктовки, открытие окна, выход.
fn setup_tray(app: tauri::AppHandle) -> Result<(), tauri::Error> {
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::TrayIconBuilder;

    let dictation_item = MenuItem::with_id(&app, "dictation-toggle", "Диктовка: выключена", true, None::<&str>)?;
    let open_item = MenuItem::with_id(&app, "open", "Открыть QazTriber", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(&app, "quit", "Выход", true, None::<&str>)?;
    let menu = Menu::with_items(&app, &[&dictation_item, &open_item, &quit_item])?;

    let mut tray = TrayIconBuilder::with_id("qaztriber-tray")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .tooltip("QazTriber")
        .on_menu_event(|app_handle, event| match event.id().as_ref() {
            "dictation-toggle" => {
                app_handle.state::<DictationManager>().toggle_enabled();
            }
            "open" => {
                if let Some(window) = app_handle.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "quit" => app_handle.exit(0),
            _ => {}
        });
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(&app)?;
    app.manage(TrayHandles {
        dictation_item,
    });
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let shared_child: SharedChild = Arc::new(Mutex::new(None));
    let shutdown = Arc::new(AtomicBool::new(false));

    // Build separately so we can pass shared_child to both
    // the setup closure (via managed state) and the run callback
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    app.state::<DictationManager>().handle_shortcut(
                        shortcut,
                        event.state == tauri_plugin_global_shortcut::ShortcutState::Pressed,
                    );
                })
                .build(),
        )
        .invoke_handler(tauri::generate_handler![
            greet,
            open_folder,
            open_url,
            save_and_open_txt,
            dictation::dictation_configure,
            dictation::dictation_status,
            dictation::dictation_permissions,
        ])
        .manage(shared_child.clone())
        .manage(shutdown.clone())
        .on_window_event(|window, event| {
            // Диктовка должна работать при закрытом окне: прячем в трей вместо выхода.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let enabled = window
                    .app_handle()
                    .try_state::<DictationManager>()
                    .map(|manager| manager.is_enabled())
                    .unwrap_or(false);
                if enabled {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .setup(move |app| {
            let url = if cfg!(debug_assertions) {
                DEV_URL.to_string()
            } else {
                let resource_path = app
                    .path()
                    .resource_dir()?
                    .join("binaries")
                    .join("qaztriber-backend")
                    .join(if cfg!(target_os = "windows") {
                        "qaztriber-backend.exe"
                    } else {
                        "qaztriber-backend"
                    });

                spawn_sidecar_thread(app.handle().clone(), resource_path, shared_child.clone());

                // Не возвращаем Err при неудаче старта sidecar — показываем окно
                // с диагностикой вместо молчаливого краха. С `panic = "abort"` в
                // release паника убивает процесс мгновенно, без окна. Frontend
                // покажет ошибку соединения; health monitor сообщит unreachable.
                if !wait_for_backend(BACKEND_STARTUP_TIMEOUT_SECS) {
                    eprintln!(
                        "Backend failed to start within {}s — window will show connection error",
                        BACKEND_STARTUP_TIMEOUT_SECS
                    );
                }

                // Start health monitor after backend is confirmed ready
                let shutdown_for_health = shutdown.clone();
                let app_handle_for_health = app.handle().clone();
                std::thread::spawn(move || {
                    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
                    rt.block_on(async move {
                        start_health_monitor(app_handle_for_health, shutdown_for_health).await;
                    });
                });

                format!("http://{}:{}", BACKEND_HOST, BACKEND_PORT)
            };

            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url.parse()?))
                .title("QazTriber")
                .inner_size(1200.0, 800.0)
                .min_inner_size(900.0, 600.0)
                .center()
                .build()?;

            setup_tray(app.handle().clone())?;
            let dictation = DictationManager::new(app.handle().clone());
            app.manage(dictation);

            Ok(())
        })
        .build(tauri::generate_context!());

    let app = match app {
        Ok(app) => app,
        Err(e) => {
            // С `panic = "abort"` .expect() убивает процесс мгновенно без окна.
            // Логируем и выходим с кодом — пользователь хотя бы увидит stderr
            // при запуске из терминала. Основной сценарий ошибок sidecar уже
            // обработан в setup (логируем, но не паникуем).
            eprintln!("Failed to build Tauri app: {}", e);
            std::process::exit(1);
        }
    };

    // BUG C1: kill sidecar on app exit via RunEvent
    app.run(move |app_handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            // Signal health monitor to stop
            if let Some(shutdown) = app_handle.try_state::<Arc<AtomicBool>>() {
                shutdown.store(true, Ordering::SeqCst);
            }
            // Kill sidecar
            if let Some(shared) = app_handle.try_state::<SharedChild>() {
                if let Ok(mut guard) = shared.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
            }
        }
    });
}
