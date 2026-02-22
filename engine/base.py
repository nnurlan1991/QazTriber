from abc import ABC, abstractmethod
import numpy as np

class BaseASREngine(ABC):
    @abstractmethod
    def load_model(self, model_path: str, device: str, beam_size: int, callback=None) -> None:
        """
        Loads the ASR model from the given path.
        """
        pass

    @abstractmethod
    def transcribe(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribes an audio chunk, returning the recognized text.
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Unloads the model from memory.
        """
        pass
