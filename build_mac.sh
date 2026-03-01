#!/bin/bash
set -e
export MACOSX_DEPLOYMENT_TARGET=14.0

echo "🔧 Сборка QazTriber..."

# Используем .spec файл, который содержит NSMicrophoneUsageDescription
./build_env/bin/pyinstaller --noconfirm --clean QazTriber.spec

echo "📦 Копирование моделей (актуализация)..."
# Путь к моделям внутри бандла
mkdir -p dist/QazTriber.app/Contents/Resources/models
cp -RL models/whisper-turbo-ksc2-ct2-v3-clean dist/QazTriber.app/Contents/Resources/models/

# Совместимость с предыдущей структурой, если она требовалась (опционально)
# mkdir -p dist/QazTriber.app/Contents/Resources/dist_models
# cp -RL models/whisper-turbo-ksc2-ct2-v3-clean dist/QazTriber.app/Contents/Resources/dist_models/

echo ""
echo "✅ Готово! Приложение: dist/QazTriber.app"
