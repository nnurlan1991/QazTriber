# QazTriber CLI — транскрипция из терминала

Локальная офлайн-транскрипция аудио в текст из командной строки. Тот же движок и модели, что у desktop-приложения, без HTTP-сервера и без облака.

## Установка

CLI живёт внутри установленного приложения:

- **macOS:** `/Applications/QazTriber.app/Contents/Resources/binaries/qaztriber-backend/qaztriber-backend`
- **Windows:** `C:\Program Files\QazTriber\binaries\qaztriber-backend\qaztriber-backend.exe`

Удобный алиас (macOS / Linux, добавить в `~/.zshrc`):

```bash
alias qaztriber='/Applications/QazTriber.app/Contents/Resources/binaries/qaztriber-backend/qaztriber-backend'
```

Windows (PowerShell, добавить в `$PROFILE`):

```powershell
function qaztriber { & "C:\Program Files\QazTriber\binaries\qaztriber-backend\qaztriber-backend.exe" @args }
```

В dev-режиме (из корня репо):

```bash
backend/.venv/bin/python -m backend.app.cli transcribe file.mp3
```

## Использование

```bash
qaztriber-backend transcribe meeting.mp3              # текст в stdout
qaztriber-backend transcribe rec.wav --model 600m     # точная модель
qaztriber-backend transcribe rec.mp3 --no-punct       # без восстановления пунктуации
qaztriber-backend transcribe rec.mp3 --json           # {"file","text","duration_seconds"}
qaztriber-backend transcribe a.mp3 b.wav              # пачка файлов (модель грузится один раз)
cat record.wav | qaztriber-backend transcribe -       # stdin
qaztriber-backend transcribe rec.mp3 --download       # скачать модель, если её нет
```

Форматы: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm` (нормализуются через ffmpeg автоматически).

Флаги:

| Флаг | По умолчанию | Описание |
|------|--------------|----------|
| `--model` | `220m` | `220m` — быстрая (~3x realtime), `600m` — точная (~1x realtime) |
| `--no-punct` | выкл. | Пропустить восстановление пунктуации/регистра |
| `--json` | выкл. | JSON вместо простого текста |
| `--download` | выкл. | Скачать модель при отсутствии (880 МБ / 2.3 ГБ) |

## Контракт для агентов (Claude Code, opencode и т.п.)

- **stdout** — только текст расшифровки (или один JSON-объект с `--json`; для нескольких файлов — JSON-массив).
- **stderr** — прогресс и предупреждения (читать при диагностике, игнорировать при успехе).
- **Exit codes:** `0` — успех; `1` — ошибка (нет файла, пустой stdin, сбой); `2` — модель не скачана → перезапустить с `--download` (первая загрузка ~880 МБ, терпеливо) или скачать через desktop-приложение (раздел «Модели»).
- Первое использование: модель должна быть скачана один раз (приложением или флагом `--download`), дальше всё работает офлайн.
- Модель грузится в память на время вызова и выгружается после — не держит RAM между запусками.
- Модель хранится в: macOS `~/Library/Application Support/QazTriber/models`, Windows `%LOCALAPPDATA%\QazTriber\models`.

### Примеры для агентов

```bash
# быстрая расшифровка и вывод в терминал
qaztriber-backend transcribe audio.mp3 2>/dev/null

# сохранить в файл
qaztriber-backend transcribe audio.m4a --json 2>/dev/null > result.json

# обработать все mp3 в папке
for f in *.mp3; do qaztriber-backend transcribe "$f" 2>/dev/null > "${f%.mp3}.txt"; done
```

## Проверка работоспособности

```bash
backend/.venv/bin/python backend/scripts/cli_smoke_test.py
```

Скрипт кроссплатформенный (macOS/Windows): синтезирует WAV, проверяет exit codes и JSON-вывод. В репо без скачанной модели тест №2 (exit 2) тоже отработает.

### Чеклист Windows

1. `backend\.venv\Scripts\python.exe backend\scripts\cli_smoke_test.py` (env `PYTHONUTF8=1` обязателен).
2. Проверить `qaztriber-backend.exe transcribe <файл с кириллицей в имени>` — кириллица в путях и stdout.
3. Проверить `|` piping (stdin) в PowerShell: `Get-Content file.wav -AsByteStream | ...` — либо просто файлом, stdin в PowerShell неудобен.
