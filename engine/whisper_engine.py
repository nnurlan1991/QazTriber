import os
import gc
import threading
import numpy as np
from typing import Optional, Callable
from faster_whisper import WhisperModel
from .base import BaseASREngine

import sys

class WhisperEngine(BaseASREngine):
    def __init__(self):
        self._model = None
        self.device = "cpu"
        self.beam_size = 5
        self.current_model_name = "whisper-turbo-ksc2-ct2-v3-clean"
        
        # --- PATH HANDLING ---
        if getattr(sys, 'frozen', False):
            # Bundled: check for _internal folder (PyInstaller 6+)
            base_dir = sys._MEIPASS
            internal_path = os.path.join(base_dir, "_internal")
            if os.path.exists(internal_path):
                base_dir = internal_path
        else:
            base_dir = os.getcwd()
            
        self.models_dir = os.path.join(base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def load_model(self, model_name: Optional[str] = None, callback: Optional[Callable] = None) -> None:
        """Loads a model by name or path. Downloads standard models to models/ if needed."""
        if model_name:
            if self.current_model_name != model_name:
                self.unload_model()
                self.current_model_name = model_name

        def report(msg: str, progress: Optional[float] = None):
            if callback:
                try: callback(msg, progress)
                except TypeError: callback(msg)

        if self._model is not None:
            return

        report(f"🚀 Загрузка {self.current_model_name}...", 0.3)
        
        # Use full path for custom models, or name for standard ones
        if self.current_model_name == "whisper-turbo-ksc2-ct2-v3-clean":
            path = os.path.join(self.models_dir, self.current_model_name)
            if not os.path.exists(path):
                report("⚠️ Модель не найдена", 0.0)
                raise RuntimeError("Основная модель не найдена в папке models/")
        else:
            path = self.current_model_name # Let faster-whisper handle download by name

        try:
            # Optimized for ARM Mac (auto selects float32/int8 based on availability)
            self._model = WhisperModel(
                path, 
                device="cpu", 
                compute_type="auto",
                download_root=self.models_dir
            )
            report("✅ Модель готова!", 1.0)
        except Exception as e:
            report("❌ Ошибка загрузки", 0.0)
            raise RuntimeError(f"Ошибка инициализации {self.current_model_name}: {e}")

    def transcribe(self,
                   audio_chunk: np.ndarray,
                   sr: int = 16000,
                   progress_callback: Optional[Callable] = None,
                   stop_event: Optional[threading.Event] = None,
                   text_callback: Optional[Callable[[str], None]] = None) -> str:
        
        def prog(msg, frac=None):
            if progress_callback:
                try: progress_callback(msg, frac)
                except TypeError: progress_callback(msg)

        prog("🕒 Обработка...", 0.1)
        
        # Normalize and ensure mono
        audio = np.array(audio_chunk, dtype=np.float32)
        if audio.ndim > 1: audio = audio[:, 0]
        
        # Standardize volume for model
        max_val = float(np.abs(audio).max())
        if max_val > 1e-6:
            audio = audio / max_val

        segments, info = self._model.transcribe(
            audio, 
            beam_size=5, 
            best_of=5,
            temperature=0,
            language=None, 
            task="transcribe",
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500, threshold=0.35),
            condition_on_previous_text=True,
            initial_prompt=None
        )
        
        texts = []
        for segment in segments:
            if stop_event and stop_event.is_set():
                break
            
            text = segment.text.strip()
            if text:
                texts.append(text)
                if text_callback:
                    text_callback(text + " ")
            
            progress = segment.end / info.duration if info.duration > 0 else 0.5
            prog(f"⚙️ Обработка: {segment.end:.1f}с / {info.duration:.1f}с", progress)

        prog("✅ Завершено", 1.0)
        return " ".join(texts)

    def unload_model(self) -> None:
        self._model = None
        gc.collect()

    @staticmethod
    def is_model_ready() -> bool:
        path = os.path.join("models", "whisper-turbo-ksc2-ct2-v3-clean")
        return os.path.isdir(path) and os.path.exists(os.path.join(path, "model.bin"))
