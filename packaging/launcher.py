"""Tauri sidecar: поднимает локальный API. Окно открывает Tauri."""
from __future__ import annotations

import logging
import sys

import uvicorn

from backend.app.main import app

HOST = "127.0.0.1"
PORT = 8765


def main() -> None:
    # PyInstaller запускает Windows-версию без консоли: sys.stdout/sys.stderr
    # могут быть None. Uvicorn вызывает isatty() для цветного вывода и падает.
    #
    # Defense-in-depth: даже если PYTHONUTF8 env var не установлен (например,
    # в dev-режиме), перенастраиваем stdout/stderr на UTF-8. errors="replace"
    # предотвращает крах на недекодируемых байтах — они становятся ? вместо
    # возбуждения UnicodeEncodeError.
    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if sys.stderr is not None:
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    # CLI-режим: `qaztriber-backend transcribe file.mp3` — без HTTP-сервера
    # (см. backend/app/cli.py и docs/CLI.md).
    if len(sys.argv) > 1 and sys.argv[1] == "transcribe":
        from backend.app.cli import main as cli_main

        raise SystemExit(cli_main(sys.argv[2:]))
    # Логирование настроено в backend.app.main lifespan через init_logging()
    # (RotatingFileHandler в sidecar.log). Uvicorn-конфиг отключён, чтобы
    # избежать дублирования обработчиков.
    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_config=None,
        access_log=False,
    )
    # Подавляем шумные access-логи uvicorn, но оставляем error-логи
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    server = uvicorn.Server(config)
    # Главный поток — процесс живёт, пока Tauri не завершит его.
    server.run()


if __name__ == "__main__":
    main()
