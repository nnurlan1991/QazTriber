"""Диктовка в любом месте (push-to-talk): синхронная транскрипция без job-менеджера.

Отличия от /api/transcriptions:
- синхронный ответ (Rust-клиент диктовки ждёт текст, polling не нужен);
- ничего не пишется в jobs/ — история сессий не замусоривается;
- модель остаётся в памяти после ответа (jobs-воркер, наоборот, выгружает).
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from ..services.audio import run_ffmpeg, wav_duration_seconds
from ..services.gigaam import GigaAMService, MODELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dictate", tags=["dictation"])

MAX_DICTATE_BYTES = 30 * 1024 * 1024  # ~30 МБ WAV хватает на ~3 минуты 16 kHz mono
MIN_DURATION_SECONDS = 0.25


def _noop_report(stage: str, progress: float) -> None:
    del stage, progress


@router.post("")
async def dictate(
    request: Request,
    model: str = Query(...),
    expected_language: str = Query("mixed"),
) -> dict[str, object]:
    """Тело запроса — сырой WAV (16-bit PCM, любая частота). Ответ: {"text": ...}."""
    if model not in MODELS:
        return JSONResponse(status_code=422, content={"detail": "Выберите модель 220M или 600M."})
    if expected_language not in {"kazakh", "russian", "mixed"}:
        return JSONResponse(status_code=422, content={"detail": "Допустимы языки: казахский, русский или смешанный."})

    data = await request.body()
    if not data:
        return JSONResponse(status_code=422, content={"detail": "Пустая аудиозапись."})
    if len(data) > MAX_DICTATE_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Запись слишком длинная."})

    gigaam: GigaAMService = request.app.state.gigaam
    try:
        with tempfile.TemporaryDirectory(prefix="qaztriber-dictate-") as tmp:
            tmp_dir = Path(tmp)
            raw_path = tmp_dir / "raw.wav"
            normalized = tmp_dir / "normalized.wav"
            raw_path.write_bytes(data)
            run_ffmpeg(raw_path, normalized, None, None)
            duration = wav_duration_seconds(normalized)
            if duration < MIN_DURATION_SECONDS:
                return {"text": "", "duration_seconds": duration}
            text = gigaam.transcribe(model, normalized, tmp_dir / "chunks", _noop_report, lambda: False)
    except Exception as error:
        logger.error("Dictation failed: %s", error)
        return JSONResponse(status_code=500, content={"detail": f"Ошибка диктовки: {error}"})
    return {"text": text, "duration_seconds": duration}


@router.post("/warmup")
def dictate_warmup(request: Request, model: str = Query(...)) -> dict[str, str]:
    """Загружает модель в память синхронно — вызывается при включении диктовки.

    Синхронно, чтобы Rust-клиент видел состояние «загрузка» по незавершённому HTTP-запросу.
    """
    if model not in MODELS:
        return JSONResponse(status_code=422, content={"detail": "Выберите модель 220M или 600M."})
    gigaam: GigaAMService = request.app.state.gigaam
    if gigaam.is_loaded(model):
        return {"status": "ready"}
    try:
        gigaam.load(model, _noop_report)
    except Exception as error:
        logger.error("Dictation warmup failed: %s", error)
        return JSONResponse(status_code=500, content={"detail": f"Не удалось загрузить модель: {error}"})
    return {"status": "ready"}
