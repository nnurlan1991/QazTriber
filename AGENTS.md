# QazTriber — Project Guide

> **Главный контекстный файл для AI-агентов. Читай перед любой работой.**
> Если в задаче упоминаются продукты/релизы/сервер/диктовка — всё нужное ниже.

## Overview

**QazTriber** — десктопное приложение (macOS/Windows) для **офлайн**-расшифровки казахской/русской/смешанной речи. Аудио обрабатывается локально (PyTorch, MPS на Apple Silicon, CPU на Windows). Никаких облаков, аккаунтов для транскрипции, телеметрии. Текущая версия: `1.4.1` (см. `src-tauri/tauri.conf.json`).

## Products

| Продукт | Платформа | Где |
|---------|-----------|-----|
| Desktop | Tauri 2.x + Python sidecar | этот репозиторий |
| Mobile | нативный Kotlin/Compose | `/Users/market/Documents/проекты программиста/qaztribber_mobile/` |
| Landing | статичный HTML на nginx | `landing/index.html` + сервер |
| Models | nginx статика | `https://qaztribber.aidi-lab.kz/models/desktop/` |

## Tech Stack

- **Frontend:** Tauri 2.x + React 18 + TypeScript + Vite. Дизайн-система obsidian+gold в `frontend/src/styles.css`
- **Backend (sidecar):** Python 3.11, FastAPI, Uvicorn, PyInstaller (`--onedir`) → `src-tauri/binaries/qaztriber-backend/`. venv: `backend/.venv/bin/python`
- **ИИ модель:** PyTorch (не ONNX, не CoreML/ANE). GigaAM CTC: 220M (~880 МБ, быстрая) + 600M (~2.3 ГБ, точная). Скачиваются при первом запуске. Инференс: MPS на Apple Silicon, CPU на Intel/Windows

## Build Pipeline

```bash
# 1. Frontend → frontend/dist/
cd frontend && npm run build
# 2. Sidecar (PyInstaller + strip) → src-tauri/binaries/qaztriber-backend/
backend/.venv/bin/python packaging/build_release.py
# 3. Tauri app → src-tauri/target/release/bundle/
npm run tauri build
```

## Architecture (критичные детали)

- **Frontend вшит в Python sidecar:** PyInstaller копирует `frontend/dist/` в `_internal/frontend/dist/`. Пересборка только Tauri НЕ обновляет frontend — нужно пересобирать sidecar.
- **Модели НЕ вшиты в .app** — качаются при первом запуске (~3 ГБ).
- **Polling вместо SSE:** `watchJob()` в `api.ts` — `setTimeout`-polling каждые 500мс (не EventSource).
- **`/api/system`** — device, CPU brand, memory, speed_multiplier для ETA.
- **`strip_sidecar()` в `build_release.py`** — удаляет дубликаты libtorch (`.dylib`/`.dll`) из `torch/lib/`, экономит ~350 МБ. Кроссплатформенный.
- **Tauri ACL:** команды sidecar/dictation открыты через `remote` origin в capabilities (v1.4.1) — без этого release-сборки падают с отказом ACL.

## Dictation (v1.4.0+) — push-to-talk в любом приложении

Поток: глобальный хоткей → захват микрофона → WAV → локальное распознавание → вставка текста в активное окно. Звуковое сопровождение старт/стоп.

- **Хоткей** `alt+shift+KeyD` (настраивается в настройках): через `tauri-plugin-global-shortcut`. ⚠️ `GlobalHotKeyManager` на Windows `!Send` — им владеет плагин (регистрация в main thread), сам `dictation.rs` только хранит зарегистрированный shortcut.
- **Захват микрофона:** `cpal` (Windows-фикс `!Send` в v1.4.1).
- **Rust-сторона:** `src-tauri/src/dictation.rs` — хоткей, захват, звук, HTTP в sidecar, вставка.
- **API (синхронные, без job):**
  - `POST /api/dictate` — WAV → текст
  - `POST /api/dictate/warmup?model=<name>` — прогрев модели в фоне
  - `POST /api/dictate/unload` — выгрузка модели из памяти (вызывается при выключении диктовки в настройках)
- **Гайд:** `docs/DICTATION.md`

## Punct-restore (v1.3.7+) — восстановление пунктуации и регистра

- **Модель:** XLM-RoBERTa с двумя головами (punct + case), локальная директория из `settings.punct_model_dir` (складывается из `punct_case_model_v2_final/`). Поддержка int8 (загрузка вручную, минуя `from_pretrained`).
- **Файлы:** `backend/app/services/punct_restore.py` (сервис), `quantize_punct.py` (int8-квантование).
- **Память:** модель выгружается сразу после транскрипции (как GigaAM, v1.3.7).
- **Импорт сессий работает без сети** (v1.4.1 — убрана зависимость от HF-загрузки при импорте).

## Versioning (ВАЖНО для релизов)

- **Единый источник версии** — поле `version` в `src-tauri/tauri.conf.json` (дублируется в `src-tauri/Cargo.toml` `[package] version`).
- **CI синхронизирует version-файлы из тега** (шаг "Sync all version files from tag" в `release.yml`) — версия в теге главнее, чем в коммите; файлы в репо должны совпадать, иначе CI их перепишет.
- **В UI (раздел «О приложении»)** версия берётся через `getVersion()` из `@tauri-apps/api/app` (`frontend/src/views/SettingsView.tsx`) — читает `tauri.conf.json`. **Не хардкодить версию в коде.**
- **Эта версия === версия GitHub релиза**: release workflow триггерится тегом `v{version}` (например `v1.4.1`), CI собирает `.dmg`/`.exe` и публикует в Release с той же версией.
- **Ассеты релиза — без версии в имени** (`QazTriber_aarch64.dmg`, `QazTriber_x64-setup.exe`): CI переименовывает при теге, лендинг использует стабильные ссылки `releases/latest/download/...`.
- **При релизе:** обновить `version` в `tauri.conf.json` + `Cargo.toml` → коммит → тег `v{version}` → push. UI обновится автоматически, делать сетевой запрос к GitHub API **не нужно** (офлайн-приложение).

## Auth & User Management (v1.3.0+)

- **Firebase Auth** (email/password + Google Sign-In через системный браузер — Tauri WebView блокирует OAuth popups: desktop → VPS `google.html` → customToken → desktop)
- **Admin approval gate:** регистрация → `approved=false` (Firestore rules) → доступ только после одобрения админом
- **VPS admin panel** (`https://qaztribber.aidi-lab.kz/admin/`, порт 3003, pm2 `qaztriber-admin`) — Express + Firebase Admin SDK, обходит rules. Вход: Telegram bot `@qaztriberbot` → `/login` → magic link → JWT cookie
- **Telegram bot** `@qaztriberbot` — уведомления + inline кнопки (approve/reject/revoke)
- **Whitelist** — bulk Excel/CSV импорт email с auto-approve
- **Без Cloud Functions** (Spark план) — всё на Firestore rules + VPS Admin SDK
- Ключевые файлы: `frontend/src/lib/{firebase,auth}.ts*`, `frontend/src/views/{Auth,PendingApproval}View.tsx`, `admin-panel/server/src/`, `admin-panel/web/google.html`, `firestore.rules`, `docs/AUTH_SKILLS.md`
- Service account на VPS: `/home/ai/.config/qaztriber/firebase-service-account.json`

## Deployment

- **Desktop CI:** `.github/workflows/release.yml` — триггер push тега `v*` или ручной. Платформы: `macos-14` + `windows-latest` параллельно → `.dmg` + `.exe` в GitHub Releases
- **Landing:** `.github/workflows/pages.yml` авто-деплой при пуше в `landing/`. Сервер: `/var/www/qaztriber/index.html`. Зеркало: `https://nnurlan1991.github.io/qaztriber2.0/`. Ссылки на репо — `QazTriber` (исправлено в v1.3.10)
- **Models:** `multilingual_ctc.ckpt`, `multilingual_large_ctc.ckpt`, `manifest.json` на `https://qaztribber.aidi-lab.kz/models/desktop/`. Range requests включены

## Server Access

| Параметр | Значение |
|----------|----------|
| Host | `46.224.176.8` (он же `aidi-lab.kz`) |
| User | `ai` |
| SSH | `ssh -i ~/.ssh/ai_project1 ai@46.224.176.8` (sudo без пароля; `id_rsa_aidi` на этом Mac НЕ работает) |
| Webroot | `/var/www/qaztriber/` |
| nginx | `/etc/nginx/sites-enabled/qaztribber.aidi-lab.kz` |
| SSL | Let's Encrypt (Certbot) |

Другие сервисы на сервере: `aidi-lab.kz` (:3000), `dastarkhan.online` (:3001), `slidegen`, `tapsiramin`, `tapsirubot`, `vp`, `oyau`.

## GitHub

- **Desktop repo:** `https://github.com/nnurlan1991/QazTriber` (public, переименован из `qaztribber2.0` в v1.3.10). Releases: `/releases`. Assets: `QazTriber_aarch64.dmg` (195 МБ), `QazTriber_x64-setup.exe` (150 МБ)
- **Mobile repo:** `https://github.com/nnurlan1991/qaztriber_mobile` (public). Asset: `QazTriber.apk` (без версии в имени)
- **Прямые ссылки:** macOS — `.../releases/latest/download/QazTriber_aarch64.dmg`, Windows — `.../releases/latest/download/QazTriber_x64-setup.exe`, Android — через GitHub API `assets[0].browser_download_url`

## Release Workflow

### Desktop (CI автоматически)
1. Обновить `version` в `src-tauri/tauri.conf.json` и `src-tauri/Cargo.toml`
2. `git commit -am "v1.x.0: <описание>"` → `git push origin main`
3. `git tag -a v1.x.0 -m "v1.x.0: <описание>" && git push origin v1.x.0`
4. CI синхронизирует version-файлы из тега, соберёт `.dmg` + `.exe` за ~10 мин, переименует ассеты и опубликует в Release
5. Дождаться: `gh run watch` → проверить: `gh release view v1.x.0`

### Android (вручную из мобильного репо)
1. `cd /Users/market/Documents/проекты программиста/qaztribber_mobile/android_app && ./gradlew assembleDebug`
2. `gh release upload v1.x.0 ".../app-debug.apk#QazTriber.apk" --repo nnurlan1991/qaztriber_mobile --clobber`

### Лендинг обновлять НЕ нужно — ссылки динамические (`releases/latest/download/` + GitHub API)

### Финальная проверка
- `gh release view v1.x.0` → `.dmg` + `.exe`
- `gh release view v1.x.0 --repo nnurlan1991/qaztriber_mobile` → APK
- `curl -sI https://qaztribber.aidi-lab.kz/` → 200 OK

## Key Files

```
frontend/src/
  views/          HomeView, HistoryView, SessionView, ModelsView, SettingsView, AuthView, PendingApprovalView
  components/     Sidebar, TopBar, RecordButton, ProgressBar, Waveform, Modal, StatusBadge
  store.tsx       React context (theme, language, systemInfo)
  api.ts          polling watchJob, getSystemInfo, getLogs
  styles.css      obsidian+gold дизайн-система
  i18n.ts         RU/KZ переводы
  storage.ts      localStorage сессии
  lib/            firebase.ts, auth.tsx (auth flow)

backend/app/
  api/transcriptions.py     /transcribe, /system, /sessions, /logs
  api/dictation.py          POST /api/dictate, /api/dictate/warmup, /api/dictate/unload (синхронная диктовка, без job)
  services/gigaam.py        wrapper ИИ модели, MODEL_DOWNLOAD_BASE, device() → mps|cpu
  services/punct_restore.py punct+case модель (XLM-RoBERTa), int8, unload после транскрипции
  services/quantize_punct.py int8-квантование punct-модели
  schemas.py                Pydantic schemas
  config.py                 settings, punct_model_dir

src-tauri/src/dictation.rs   диктовка push-to-talk (хоткей+захват+вставка, см. docs/DICTATION.md)
packaging/build_release.py   PyInstaller + strip_sidecar()
src-tauri/tauri.conf.json    Tauri config + ЕДИНЫЙ ИСТОЧНИК ВЕРСИИ (поле version)
src-tauri/Cargo.toml         дублирует version
.github/workflows/release.yml  CI сборка desktop + sync version files + rename assets
.github/workflows/pages.yml    CI деплой лендинга
landing/index.html            лендинг
firestore.rules               security rules (deployed)
docs/DICTATION.md             гайд по диктовке
docs/AUTH_SKILLS.md           гайд по auth-архитектуре
```

## Important Notes

- **Windows CI:** env `PYTHONUTF8=1` обязателен — иначе `UnicodeEncodeError` на кириллице
- **Windows Rust:** `cpal` и `GlobalHotKeyManager` — `!Send`; не таскать через потоки, всё регистрировать в main thread (плагин global-shortcut)
- **DMG локально:** `bundle_dmg.sh` падает на macOS, но `.app` собирается. CI собирает `.dmg` успешно
- **ИИ модель:** PyTorch (не ONNX, не CoreML). `device()` в `gigaam.py` определяет `mps` или `cpu`. 220M ~3x realtime на MPS, 600M ~1x realtime
- **`.swarm/` и `.opencode/`** в `.gitignore` — не коммитить
- **Языки интерфейса:** RU, KZ. **Языки распознавания:** kazakh, russian, mixed
