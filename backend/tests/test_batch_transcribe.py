from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf
import torch

from backend.app.services.audio import split_wav_arrays
from backend.app.services.gigaam import GigaAMService


def test_split_wav_arrays_short(tmp_path: Path) -> None:
    wav_path = tmp_path / "short.wav"
    data = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds
    sf.write(wav_path, data, 16000, subtype="PCM_16")

    chunks = split_wav_arrays(wav_path, max_seconds=20.0)
    assert len(chunks) == 1
    assert len(chunks[0]) == 16000 * 5


def test_split_wav_arrays_long(tmp_path: Path) -> None:
    wav_path = tmp_path / "long.wav"
    # 45 seconds -> at 20s max with 0.4s overlap, should yield 3 chunks
    data = np.zeros(16000 * 45, dtype=np.float32)
    sf.write(wav_path, data, 16000, subtype="PCM_16")

    chunks = split_wav_arrays(wav_path, max_seconds=20.0, overlap_seconds=0.4)
    assert len(chunks) == 3


def test_gigaam_service_transcribe_batch_logic(tmp_path: Path) -> None:
    service = GigaAMService(tmp_path / "models")

    # Mock the internal model
    mock_model = MagicMock()
    mock_model._device = torch.device("cpu")
    mock_model._dtype = torch.float32
    mock_model.forward.return_value = (torch.zeros(2, 10, 64), torch.tensor([10, 10]))
    mock_model._decode.return_value = [("Сәлем", None), ("Әлем", None)]

    service._model = mock_model
    service._active_model_id = "220m"

    # Create dummy 30s wav (2 chunks)
    wav_path = tmp_path / "test.wav"
    sf.write(wav_path, np.zeros(16000 * 30, dtype=np.float32), 16000, subtype="PCM_16")

    reports = []
    text = service.transcribe(
        model_id="220m",
        wav_path=wav_path,
        chunks_dir=tmp_path / "chunks",
        report=lambda stage, progress: reports.append((stage, progress)),
        cancelled=lambda: False,
    )

    assert "Сәлем Әлем" in text
    assert mock_model.forward.called
    assert mock_model._decode.called
