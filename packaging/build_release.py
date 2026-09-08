"""Собирает Tauri sidecar (PyInstaller бинарь) для текущей ОС.

Запускать после сборки frontend (`cd frontend && npm run build`).
Бинарь кладётся в src-tauri/binaries/qaztriber-backend/ — Tauri включает
его в bundle через `bundle.resources` и запускает при старте приложения.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr — Windows console defaults to cp1252 and crashes
# on Cyrillic in print() statements (UnicodeEncodeError).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"
BINARIES_DIR = ROOT / "src-tauri" / "binaries"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SIDECAR_DIR_NAME = "qaztriber-backend"


def main() -> None:
    if not FRONTEND_DIST.is_dir():
        raise SystemExit("Frontend не собран. Выполните: cd frontend && npm run build")

    separator = ";" if os.name == "nt" else ":"
    BINARIES_DIR.mkdir(parents=True, exist_ok=True)

    target_sidecar = BINARIES_DIR / SIDECAR_DIR_NAME
    if target_sidecar.exists():
        shutil.rmtree(target_sidecar)
    shutil.rmtree(DIST, ignore_errors=True)
    shutil.rmtree(BUILD, ignore_errors=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        # Скрывает консольное окно на Windows. На macOS/no-op (launcher не имеет GUI).
        # Без этого на Windows при спавне sidecar висит активное окно cmd.
        "--windowed",
        "--name",
        SIDECAR_DIR_NAME,
        "--add-data",
        f"{FRONTEND_DIST}{separator}frontend/dist",
        "--collect-all",
        "gigaam",
        "--collect-all",
        "imageio_ffmpeg",
        "--collect-all",
        "hydra",
        "--collect-all",
        "omegaconf",
        "--collect-all",
        "torchaudio",
        # transformers импортирует модели динамически (importlib) — PyInstaller
        # не видит их статически; без этого punct-restore падает в бандле:
        # "No module named 'transformers.models.metaclip_2'" (fallback без пунктуации).
        "--hidden-import",
        "transformers.models.metaclip_2",
        "--paths",
        str(ROOT),
        str(ROOT / "packaging" / "launcher.py"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    shutil.copytree(DIST / SIDECAR_DIR_NAME, target_sidecar)
    strip_sidecar(target_sidecar)
    print(f"Sidecar собран: {target_sidecar}")


def strip_sidecar(sidecar_dir: Path) -> None:
    """Удаляет дубликаты libtorch и неиспользуемые модули для уменьшения размера.

    PyInstaller кладёт libtorch библиотеки в двух местах:
    _internal/ и _internal/torch/lib/. Вторая копия — дубликат.
    Удаляем дубликаты из torch/lib/, оставляем в _internal/ (нужны для @rpath/@loader).

    Работает кроссплатформенно: .dylib (macOS) и .dll (Windows).
    """
    internal = sidecar_dir / "_internal"
    if not internal.is_dir():
        return

    # Расширения библиотек по платформе
    if sys.platform == "win32":
        lib_ext = ".dll"
        lib_prefix = ""
    else:
        lib_ext = ".dylib"
        lib_prefix = "lib"

    # 1. Дубликаты libtorch в torch/lib/ (оригиналы в _internal/, нужны для @rpath)
    torch_lib_names = [
        f"{lib_prefix}torch_cpu{lib_ext}",
        f"{lib_prefix}torch_python{lib_ext}",
        f"{lib_prefix}torch{lib_ext}",
        f"{lib_prefix}torch_global_deps{lib_ext}",
        f"{lib_prefix}c10{lib_ext}",
        f"{lib_prefix}omp{lib_ext}",
    ]
    for name in torch_lib_names:
        path = internal / "torch" / "lib" / name
        canonical = internal / name
        if path.is_file() and canonical.is_file():
            size = path.stat().st_size
            path.unlink()
            print(f"  удалён дубликат: torch/lib/{name} ({size / 1024 / 1024:.0f} МБ)")

    # 2. __pycache__ везде (безопасно — .pyc перегенерируется)
    for cache in internal.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
