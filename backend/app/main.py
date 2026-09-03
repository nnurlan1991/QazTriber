from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.transcriptions import router
from .config import settings
from .logging_config import init_logging
from .services.gigaam import GigaAMService, ModelPreloadManager
from .services.jobs import JobManager
from .services.punct_restore import PunctRestoreService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_logging(settings.app_data_dir)
    settings.ensure_directories()
    app.state.gigaam = GigaAMService(settings.models_dir)
    app.state.preload = ModelPreloadManager(app.state.gigaam)
    app.state.punct_restorer = PunctRestoreService(settings.punct_model_dir)
    if not app.state.punct_restorer.is_available:
        logger.warning(
            "Punct-restore модель не найдена в %s — транскрипты будут без "
            "восстановленной пунктуации, пока вы не скопируете туда веса.",
            settings.punct_model_dir,
        )
    app.state.jobs = JobManager(settings.jobs_dir, app.state.gigaam, app.state.punct_restorer)
    yield


app = FastAPI(title="QazTriber Local API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "QazTriber Local API"}


def frontend_dist() -> Path:
    """Находит production frontend и в обычном запуске, и внутри PyInstaller."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return root / "frontend" / "dist"


if frontend_dist().is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist(), html=True), name="frontend")
