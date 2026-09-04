import os
import glob
from typing import Optional
import numpy as np

# Patch typeguard before importing espnet
import sys
class DummyTypeguard:
    class TypeCheckError(Exception):
        pass
    @staticmethod
    def typechecked(target=None, **kwargs):
        if target is None:
            return lambda t: t
        if isinstance(target, type):
            return target
        return target
    @staticmethod
    def check_type(*args, **kwargs):
        pass
    @staticmethod
    def check_argument_types(*args, **kwargs):
        return True
    @staticmethod
    def check_return_type(*args, **kwargs):
        return True

sys.modules['typeguard'] = DummyTypeguard()  # type: ignore

from espnet2.bin.asr_inference import Speech2Text
from .base import BaseASREngine

class ESPnetEngine(BaseASREngine):
    def __init__(self):
        self._engine: Optional[Speech2Text] = None

    def load_model(self, model_path: str, device: str, beam_size: int, callback=None) -> None:
        """
        Loads the ESPnet model from a local folder.
        Expects a .yaml config file and a .pth weights file in the folder.
        """
        if callback:
            callback("Поиск конфигурации (config.yaml) и весов (.pth)...")
        
        yaml_files = glob.glob(os.path.join(model_path, "*.yaml"))
        pth_files = glob.glob(os.path.join(model_path, "*.pth"))
        
        if not yaml_files:
            raise FileNotFoundError(f"Файл конфигурации (.yaml) не найден в папке {model_path}.")
        if not pth_files:
            raise FileNotFoundError(f"Файл весов (.pth) не найден в папке {model_path}.")
            
        config_path = yaml_files[0]
        model_weights_path = pth_files[0]
        
        if callback:
            callback(f"Загрузка модели в память ({device})...")
            
        self._engine = Speech2Text(
            asr_train_config=config_path,
            asr_model_file=model_weights_path,
            device=device,
            beam_size=beam_size
        )
        
        if callback:
            callback("Модель успешно загружена!")

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribes an audio chunk, returning the recognized text.
        """
        if self._engine is None:
            raise RuntimeError("Движок ASR не был инициализирован. Сначала вызовите load_model().")
        
        # result is a list of tuples (text, token, token_int, hypothesis)
        result = self._engine(audio_chunk)
        return result[0][0]
