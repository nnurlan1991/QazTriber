import os
import gc
import math
import threading
import torch
import whisper
import numpy as np
from typing import Optional, Callable
from .base import BaseASREngine


# ──────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _get_dir_size(path: str) -> int:
    """Возвращает суммарный размер всех файлов в директории."""
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _filter_repetitions(text: str, max_repeats: int = 4) -> str:
    """Обрезает зациклившийся текст (галлюцинации Whisper)."""
    if not text:
        return text
    words = text.split()
    if len(words) < max_repeats * 2:
        return text
    for n in range(1, 4):
        i = n
        repeat_count = 1
        while i + n <= len(words):
            if words[i:i + n] == words[i - n:i]:
                repeat_count += 1
                if repeat_count >= max_repeats:
                    cut = i - n * (repeat_count - 1)
                    return " ".join(words[:cut]) + " [⚠️ повтор обрезан]"
            else:
                repeat_count = 1
            i += n
    return text


# ──────────────────────────────────────────────────────────────
# Класс движка
# ──────────────────────────────────────────────────────────────

class WhisperEngine(BaseASREngine):
    def __init__(self):
        self._engine = None
        self._model = None
        self._processor = None
        self._use_transformers = False
        self.beam_size = 5
        self.device = "cpu"
        self._sr = 16000
        self._torch_dtype = torch.float32

    # ── Загрузка модели ──────────────────────────────────────

    def load_model(self,
                   model_path: str,
                   device: str,
                   beam_size: int,
                   callback: Optional[Callable] = None,
                   cache_dir: Optional[str] = None) -> None:
        self.device = device
        self.beam_size = beam_size

        def report(msg: str, progress: Optional[float] = None):
            if callback:
                try:
                    callback(msg, progress)
                except TypeError:
                    callback(msg)

        report(f"🔍 Опрос модели: {model_path}", 0.0)

        final_path = model_path
        is_hf = "/" in model_path and not os.path.isabs(model_path) and not os.path.exists(model_path)

        # ── Определяем тип модели ────────────────────────────
        if is_hf:
            try:
                from huggingface_hub import list_repo_files
                repo_files = list(list_repo_files(repo_id=model_path))
                self._use_transformers = "config.json" in repo_files

                if not self._use_transformers:
                    # Нативный Whisper .pt/.bin
                    candidates = [f for f in repo_files if f.endswith(".pt") or f.endswith(".bin")]
                    best = next((f for f in candidates if "model" in f.lower()), None) or (candidates[0] if candidates else None)
                    if not best:
                        raise FileNotFoundError(f"В репозитории '{model_path}' не найдено весов.")
                    from huggingface_hub import hf_hub_download
                    report(f"⬇️ Скачивание: {best}...", 0.15)
                    final_path = hf_hub_download(repo_id=model_path, filename=best, cache_dir=cache_dir)
            except Exception as e:
                raise RuntimeError(f"Ошибка HuggingFace: {e}")
        else:
            known_openai = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"}
            self._use_transformers = model_path not in known_openai and not model_path.endswith(".pt")

        # ── Получаем ожидаемый размер для прогресса ──────────
        total_mb = 0.0
        if is_hf and self._use_transformers and cache_dir:
            try:
                from huggingface_hub import model_info as hf_model_info
                info = hf_model_info(model_path)
                total_mb = sum(
                    f.size for f in info.siblings
                    if hasattr(f, "size") and f.size
                ) / (1024 * 1024)
            except Exception:
                total_mb = 0.0

        try:
            if self._use_transformers:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

                # ВНИМАНИЕ: На Apple Silicon (MPS) использование float16 часто приводит к галлюцинациям 
                # в виде бесконечных "!" или пустых строк для этой модели. 
                # Принудительно используем float32 для стабильности.
                if torch.backends.mps.is_available():
                    self._torch_dtype = torch.float32
                    print("[WhisperEngine] Apple Silicon detected. Forcing float32 for stability.")
                else:
                    self._torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                # ── Мониторинг скачивания в MB ──────────────────
                _monitor_stop = threading.Event()
                start_cache_size = _get_dir_size(cache_dir) if cache_dir else 0

                def _download_monitor():
                    while not _monitor_stop.is_set():
                        if total_mb > 0 and cache_dir:
                            cur_mb = max(0, _get_dir_size(cache_dir) - start_cache_size) / (1024 * 1024)
                            frac = min(0.95, cur_mb / total_mb)
                            report(
                                f"⬇️ Загрузка весов: {cur_mb:.1f} MB / {total_mb:.1f} MB",
                                frac
                            )
                        else:
                            report("📦 Загрузка весов нейросети...", None)
                        _monitor_stop.wait(0.6)

                monitor_t = threading.Thread(target=_download_monitor, daemon=True)
                monitor_t.start()

                try:
                    self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        final_path, 
                        torch_dtype=self._torch_dtype,
                        low_cpu_mem_usage=True,
                        cache_dir=cache_dir
                    )
                finally:
                    _monitor_stop.set()

                self._model.to(device)



                report("📝 Загрузка токенизатора...", 0.0)
                self._processor = AutoProcessor.from_pretrained(final_path, cache_dir=cache_dir)
                processor = self._processor

                # --- MONKEYPATCH ДЛЯ ЗАЩИТЫ ОТ OVERFLOWERROR В WHISPER ---
                _orig_decode = processor.tokenizer.decode
                def _safe_decode(token_ids, *args, **kwargs):
                    def flatten(l):
                        for el in l:
                            if hasattr(el, "__iter__") and not isinstance(el, (str, bytes)):
                                yield from flatten(el)
                            else:
                                yield el

                    # Превращаем вход в плоский список ID
                    if hasattr(token_ids, "tolist"):
                        ids_list = token_ids.tolist()
                    elif hasattr(token_ids, "__iter__"):
                        ids_list = list(token_ids)
                    else:
                        ids_list = [token_ids]
                    
                    flat_ids = list(flatten(ids_list))
                    
                    max_id = processor.tokenizer.vocab_size + len(processor.tokenizer.get_added_vocab())
                    
                    filtered_ids = []
                    for t in flat_ids:
                        try:
                            val = int(t)
                            if 0 <= val < max_id:
                                filtered_ids.append(val)
                        except (TypeError, ValueError):
                            continue
                            
                    try:
                        return _orig_decode(filtered_ids, *args, **kwargs)
                    except Exception:
                        try:
                            return _orig_decode([t for t in filtered_ids if t < processor.tokenizer.vocab_size], *args, **kwargs)
                        except Exception:
                            return ""
                processor.tokenizer.decode = _safe_decode
                # ---------------------------------------------------------

                report("🚀 Модель готова...", 0.9)
                # Вызов pipeline удален, используем model.generate напрямую для стабильности
                self._engine = None 
            else:
                report("📦 Загрузка Whisper...", 0.0)
                self._engine = whisper.load_model(
                    final_path, device=device, download_root=cache_dir
                )

            report("✅ Модель загружена!", 1.0)
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации '{model_path}': {e}")

    # ── Транскрибация ────────────────────────────────────────

    def transcribe(self,
                   audio_chunk: np.ndarray,
                   sr: int = 16000,
                   initial_prompt: Optional[str] = None,
                   progress_callback: Optional[Callable] = None,
                   stop_event: Optional[threading.Event] = None,
                   text_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Транскрибирует аудио с реальным прогрессом по чанкам.
        stop_event.is_set() → немедленное прерывание между чанками.
        text_callback выводит текст по мере его распознавания.
        """
        # Нормализация
        audio = np.array(audio_chunk, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        max_val = float(np.abs(audio).max())
        if max_val > 10.0:
            audio = audio / 32768.0
        elif max_val > 1.0:
            audio = audio / max_val
        if len(audio) == 0:
            return ""

        def prog(msg, frac=None):
            if progress_callback:
                try:
                    progress_callback(msg, frac)
                except TypeError:
                    progress_callback(msg)

        if self._use_transformers:
            return self._transcribe_hf_chunked(audio, sr, initial_prompt, prog, stop_event, text_callback)
        else:
            return self._transcribe_openai(audio, initial_prompt, prog)

    def _transcribe_hf_chunked(self,
                                audio: np.ndarray,
                                sr: int,
                                initial_prompt: Optional[str],
                                prog: Callable,
                                stop_event: Optional[threading.Event],
                                text_callback: Optional[Callable[[str], None]]) -> str:

        total_dur = len(audio) / sr
        CHUNK_SEC = 29
        chunk_samples = CHUNK_SEC * sr

        starts = list(range(0, len(audio), chunk_samples))
        n_chunks = len(starts)
        texts: list[str] = []

        # generate_kwargs — подсказка языка (казахский) для шалақазахский речи.
        # pipeline сам обрабатывает decoder_start_token_id, forced_decoder_ids,
        # attention_mask — НЕ делаем это вручную.
        generate_kwargs: dict = {
            "language": "kk",
            "task": "transcribe",
        }
        if self.beam_size > 1:
            generate_kwargs["num_beams"] = self.beam_size

        for i, s in enumerate(starts):
            if stop_event and stop_event.is_set():
                raise InterruptedError("Остановлено пользователем")

            e = min(s + chunk_samples, len(audio))
            chunk = audio[s:e]

            if len(chunk) < sr * 0.3:
                continue

            t0 = _fmt_time(s / sr)
            t1 = _fmt_time(min(e / sr, total_dur))
            prog(f"⚙️ Чанк {i + 1}/{n_chunks}  {t0}–{t1}", i / n_chunks)

            try:
                # ── Прямой вызов генерации вместо pipeline ──────────────────
                # Это дает больше контроля и стабильности с whisper-turbo-ksc2
                input_features = self._processor(
                    chunk, sampling_rate=sr, return_tensors="pt"
                ).input_features.to(self.device).to(self._torch_dtype)

                with torch.no_grad():
                    predicted_ids = self._model.generate(
                        input_features,
                        **generate_kwargs
                    )

                text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                text = text.strip()

                if text:
                    text = _filter_repetitions(text)
                    texts.append(text)
                    if text_callback:
                        text_callback(text + " ")
                else:
                    # Пустой результат — возможно, тишина
                    pass
            except Exception as exc:
                import traceback
                print(f"[Чанк {i + 1}] Ошибка: {exc}")
                traceback.print_exc()

        prog(f"✅ Транскрибация завершена ({n_chunks} чанков)", 1.0)
        return " ".join(texts)

    def _transcribe_openai(self, audio: np.ndarray,
                           initial_prompt: Optional[str],
                           prog: Callable) -> str:
        use_fp16 = self.device not in ("cpu", "mps")
        prog("⚙️ Распознавание (openai-whisper)...", 0.1)
        result = self._engine.transcribe(
            audio,
            beam_size=self.beam_size,
            fp16=use_fp16,
            verbose=False,
            initial_prompt=initial_prompt
        )
        prog("✅ Готово", 1.0)
        return result["text"].strip()

    # ── Управление кэшем ─────────────────────────────────────

    @staticmethod
    def get_downloaded_models(cache_dir: str) -> list:
        downloaded = []
        if not cache_dir or not os.path.exists(cache_dir):
            return downloaded
        for f in os.listdir(cache_dir):
            if f.endswith(".pt"):
                downloaded.append({"name": f.replace(".pt", ""), "type": "whisper"})
        for f in os.listdir(cache_dir):
            if f.startswith("models--") and os.path.isdir(os.path.join(cache_dir, f)):
                parts = f.split("--")
                if len(parts) >= 3:
                    downloaded.append({"name": f"{parts[1]}/{parts[2]}", "type": "huggingface"})
        return downloaded

    @staticmethod
    def delete_model(model_name: str, cache_dir: str) -> None:
        import shutil
        if not cache_dir or not os.path.exists(cache_dir):
            return
        whisper_file = os.path.join(cache_dir, f"{model_name}.pt")
        if os.path.exists(whisper_file):
            os.remove(whisper_file)
        repo_folder = "models--" + model_name.replace("/", "--")
        hf_path = os.path.join(cache_dir, repo_folder)
        if os.path.isdir(hf_path):
            shutil.rmtree(hf_path)

    @staticmethod
    def is_model_downloaded(model_name: str, cache_dir: str) -> bool:
        if not cache_dir or not os.path.exists(cache_dir):
            return False
        if os.path.exists(os.path.join(cache_dir, f"{model_name}.pt")):
            return True
        if "/" in model_name:
            repo_folder = "models--" + model_name.replace("/", "--")
            if os.path.isdir(os.path.join(cache_dir, repo_folder)):
                return True
        return False

    def unload_model(self) -> None:
        self._engine = None
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()
