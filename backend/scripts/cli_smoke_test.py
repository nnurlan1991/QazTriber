#!/usr/bin/env python3
"""Smoke-тест CLI-транскрипции: синтезирует WAV, гоняет CLI, проверяет контракт.

Работает и на macOS, и на Windows (stdlib only):
    backend/.venv/bin/python backend/scripts/cli_smoke_test.py
"""
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def venv_python() -> str:
    candidates = [
        REPO_ROOT / "backend" / ".venv" / "bin" / "python",
        REPO_ROOT / "backend" / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def make_wav(path: Path, seconds: float = 2.0, freq: float = 440.0) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / 16000)))
            for i in range(int(16000 * seconds))
        )
        w.writeframes(frames)


def run_cli(*cli_args: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [venv_python(), "-m", "backend.app.cli", *cli_args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qaztriber-cli-smoke-") as tmp:
        wav = Path(tmp) / "tone.wav"
        make_wav(wav)

        # 1. Ошибка: файл не существует → exit 1, сообщение в stderr.
        missing = run_cli("transcribe", str(Path(tmp) / "nope.mp3"))
        assert missing.returncode == 1, f"exit={missing.returncode}, stderr={missing.stderr}"
        assert missing.stdout.strip() == "", "stdout должен быть пуст при ошибке"
        print("ok: отсутствующий файл → exit 1")

        # 2. Модель не скачана → exit 2 (проверяемо только если кэша нет).
        sys.path.insert(0, str(REPO_ROOT))
        from backend.app.config import settings

        from backend.app.services.gigaam import GigaAMService

        if not GigaAMService(settings.models_dir).is_cached("220m"):
            no_model = run_cli("transcribe", str(wav))
            assert no_model.returncode == 2, f"exit={no_model.returncode}, stderr={no_model.stderr}"
            print("ok: модель не скачана → exit 2 (с подсказкой про --download)")

        # 3. Успешный прогон → exit 0, валидный JSON, текст в stdout.
        result = run_cli("transcribe", str(wav), "--json")
        print(result.stderr, file=sys.stderr)
        assert result.returncode == 0, f"exit={result.returncode}, stderr={result.stderr}"
        payload = json.loads(result.stdout)
        assert "text" in payload and "duration_seconds" in payload, payload
        assert payload["duration_seconds"] > 0
        print(f"ok: транскрипция → JSON, duration={payload['duration_seconds']}s")

    print("CLI smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
