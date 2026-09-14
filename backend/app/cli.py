"""CLI-режим QazTriber: одноразовая транскрипция из терминала.

`qaztriber-backend transcribe file.mp3` — тот же бинарь, что и sidecar, но без
HTTP-сервера: грузит модель, расшифровывает, печатает текст в stdout и
выгружает модель из памяти (load → transcribe → unload, как jobs-воркер).

Контракт для агентов:
- stdout — только текст (или JSON с --json);
- stderr — прогресс и предупреждения;
- exit codes: 0 — ок, 1 — ошибка, 2 — модель не скачана (перезапустить с --download).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .config import settings
from .services.audio import run_ffmpeg, wav_duration_seconds
from .services.gigaam import GigaAMService, MODELS
from .services.punct_restore import PunctRestoreService

MIN_DURATION_SECONDS = 0.25


def _report(stage: str, progress: float) -> None:
    print(f"[{stage} {int(progress * 100)}%]", file=sys.stderr, flush=True)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr, flush=True)


def _collect_inputs(paths: list[str]) -> tuple[list[tuple[Path, str]], list[Path]]:
    """Разбирает аргументы в [(Path, метка)]; `-` читается из stdin. Возвращает (входы, временные файлы)."""
    inputs: list[tuple[Path, str]] = []
    tmp_files: list[Path] = []
    for raw in paths:
        if raw == "-":
            data = sys.stdin.buffer.read()
            if not data:
                raise ValueError("Пустой ввод (stdin).")
            fd, name = tempfile.mkstemp(prefix="qaztriber-stdin-", suffix=".audio")
            tmp = Path(name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            inputs.append((tmp, "-"))
            tmp_files.append(tmp)
            continue
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")
        inputs.append((path, str(path)))
    return inputs, tmp_files


def _transcribe_files(args: argparse.Namespace) -> int:
    inputs, tmp_files = _collect_inputs(args.files)
    try:
        gigaam = GigaAMService(settings.models_dir)
        if not gigaam.is_cached(args.model):
            if not args.download:
                print(
                    f"Модель {args.model} не скачана. Запустите ещё раз с --download "
                    f"или скачайте её в приложении QazTriber (раздел «Модели»).",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            gigaam.ensure_download(args.model, _report)

        punct: PunctRestoreService | None = None
        if not args.no_punct:
            punct = PunctRestoreService(settings.punct_model_dir)
            if not punct.is_available:
                _warn(
                    f"punct-модель не найдена в {settings.punct_model_dir} — "
                    "текст будет без восстановленной пунктуации."
                )
                punct = None

        results: list[dict[str, object]] = []
        try:
            gigaam.load(args.model, _report)
            with tempfile.TemporaryDirectory(prefix="qaztriber-cli-") as tmp:
                work = Path(tmp)
                for source, label in inputs:
                    normalized = work / "input.wav"
                    run_ffmpeg(source, normalized, None, None)
                    duration = wav_duration_seconds(normalized)
                    text = ""
                    if duration >= MIN_DURATION_SECONDS:
                        text = gigaam.transcribe(
                            args.model, normalized, work / "chunks", _report, lambda: False
                        )
                        if punct is not None:
                            try:
                                text = punct.restore(text)
                            except Exception as error:  # как jobs-воркер: падение punct не валим результат
                                _warn(f"punct-restore failed, отдаю сырой текст: {error}")
                    results.append(
                        {
                            "file": label,
                            "text": text,
                            "duration_seconds": round(duration, 2),
                        }
                    )
        finally:
            # Выгрузка сразу после транскрипции — как jobs-воркер.
            try:
                gigaam.unload()
            except Exception:
                pass
            if punct is not None:
                try:
                    punct.unload()
                except Exception:
                    pass

        if args.json:
            payload: object = results[0] if len(results) == 1 else results
            print(json.dumps(payload, ensure_ascii=False))
        elif len(results) == 1:
            print(results[0]["text"])
        else:
            for result in results:
                print(f"==> {result['file']} <==")
                print(result["text"])
                print()
        return 0
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        return 1
    except Exception as error:
        print(f"Ошибка транскрипции: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        for tmp in tmp_files:
            tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qaztriber-backend", description="QazTriber: локальная офлайн-транскрипция."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("transcribe", help="Расшифровать аудиофайл (wav/mp3/m4a/flac/ogg/webm или `-` для stdin).")
    p.add_argument("files", nargs="+", help="Аудиофайл(ы); `-` читает WAV из stdin.")
    p.add_argument("--model", choices=sorted(MODELS), default="220m", help="220M — быстрая, 600M — точная.")
    p.add_argument("--no-punct", action="store_true", help="Не восстанавливать пунктуацию.")
    p.add_argument("--json", action="store_true", help="JSON в stdout вместо простого текста.")
    p.add_argument("--download", action="store_true", help="Скачать модель, если её нет на диске.")
    args = parser.parse_args(argv)
    return _transcribe_files(args)


if __name__ == "__main__":
    sys.exit(main())
