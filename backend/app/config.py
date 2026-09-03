from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "QazTriber"
    host: str = os.getenv("QAZTRIBER_HOST", "127.0.0.1")
    port: int = int(os.getenv("QAZTRIBER_PORT", "8000"))
    max_upload_bytes: int = int(os.getenv("QAZTRIBER_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
    # Timeout in seconds if no chunk progress during transcription.
    # Valid range: 300–3600 (5 min – 1 hour). Default 600s (10 min).
    transcription_timeout: int = int(os.getenv("QAZTRIBER_TRANSCRIPTION_TIMEOUT", "600"))

    @property
    def app_data_dir(self) -> Path:
        """Папка данных, принятая для каждой настольной ОС.

        Модели весят сотни мегабайт, поэтому на Windows используем LocalAppData,
        а не каталог рядом с приложением или временную память.
        """
        configured = os.getenv("QAZTRIBER_DATA_DIR")
        if configured:
            return Path(configured).expanduser()
        if os.name == "nt":
            return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "QazTriber"
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "QazTriber"
        return Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "QazTriber"

    @property
    def models_dir(self) -> Path:
        return self.app_data_dir / "models"

    @property
    def punct_model_dir(self) -> Path:
        """Локальная папка с моделью восстановления пунктуации/регистра.

        Без механизма скачивания (dev-контур): содержимое
        punct_case_model_v2_final/ копируется сюда вручную (см. Шаг 1 ТЗ).
        Переопределяется переменной окружения QAZTRIBER_PUNCT_MODEL_DIR.
        """
        configured = os.getenv("QAZTRIBER_PUNCT_MODEL_DIR")
        if configured:
            return Path(configured).expanduser()
        return self.models_dir / "punct_v2"

    @property
    def jobs_dir(self) -> Path:
        return self.app_data_dir / "jobs"

    @property
    def logs_dir(self) -> Path:
        return self.app_data_dir / "logs"

    @property
    def downloads_status_path(self) -> Path:
        return self.app_data_dir / "downloads.json"

    @property
    def initialized_marker(self) -> Path:
        """Маркер первого запуска приложения."""
        return self.app_data_dir / ".initialized"

    def ensure_directories(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
