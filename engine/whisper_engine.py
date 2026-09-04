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
        self._use_transformers = False
        self._processor = None
        self.beam_size = 5
        self.device = "cpu"
        self._sr = 16000

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
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        final_path, dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        cache_dir=cache_dir
                    )
                finally:
                    _monitor_stop.set()

                model.to(device)

                report("📝 Загрузка токенизатора...", 0.0)
                processor = AutoProcessor.from_pretrained(final_path, cache_dir=cache_dir)

                report("🚀 Сборка pipeline...", 0.5)
                self._engine = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    batch_size=1,
                    dtype=torch_dtype,
                    device=model.device,
                    ignore_warning=True,
                )
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
                   stop_event: Optional[threading.Event] = None) -> str:
        """
        Транскрибирует аудио с реальным прогрессом по чанкам.
        stop_event.is_set() → немедленное прерывание между чанками.
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

        print(f"[Transcribe] shape={audio.shape}, max={np.abs(audio).max():.4f}")

        def prog(msg, frac=None):
            if progress_callback:
                try:
                    progress_callback(msg, frac)
                except TypeError:
                    progress_callback(msg)

        if self._use_transformers:
            return self._transcribe_hf_chunked(audio, sr, initial_prompt, prog, stop_event)
        else:
            return self._transcribe_openai(audio, initial_prompt, prog)

    def _transcribe_hf_chunked(self,
                                audio: np.ndarray,
                                sr: int,
                                initial_prompt: Optional[str],
                                prog: Callable,
                                stop_event: Optional[threading.Event]) -> str:
        CHUNK_SEC = 30
        chunk_samples = CHUNK_SEC * sr
        total_dur = len(audio) / sr
        n_chunks = max(1, math.ceil(len(audio) / chunk_samples))
        texts = []

        generate_kwargs: dict = {
            "num_beams": self.beam_size,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
            "max_new_tokens": 256,
        }

        for i in range(n_chunks):
            # ── Жёсткий стоп ────────────────────────────────
            if stop_event and stop_event.is_set():
                raise InterruptedError("Остановлено пользователем")

            s = i * chunk_samples
            e = min(s + chunk_samples, len(audio))
            chunk = audio[s:e]
            t0 = _fmt_time(s / sr)
            t1 = _fmt_time(min(e / sr, total_dur))
            frac = i / n_chunks

            prog(f"⚙️ Чанк {i + 1}/{n_chunks}  {t0}–{t1}", frac)

            audio_input = {"array": chunk, "sampling_rate": sr}
            result = self._engine(audio_input, generate_kwargs=generate_kwargs)
            print(f"[Chunk {i+1}/{n_chunks}] {result}")

            if isinstance(result, dict):
                texts.append(_filter_repetitions(result.get("text", "").strip()))

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
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()
