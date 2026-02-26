#!/bin/bash
set -e
export MACOSX_DEPLOYMENT_TARGET=14.0

echo "🔧 Сборка QazTriber..."

# Используем .spec файл, который содержит NSMicrophoneUsageDescription
./build_env/bin/pyinstaller --noconfirm --clean QazTriber.spec

echo "📦 Копирование моделей..."
mkdir -p dist/QazTriber.app/Contents/Resources/dist_models
cp -RL dist/models/models--abilmansplus--whisper-turbo-ksc2 dist/QazTriber.app/Contents/Resources/dist_models/
ln -sf ../Resources/dist_models dist/QazTriber.app/Contents/Frameworks/dist_models

echo ""
echo "✅ Готово! Приложение: dist/QazTriber.app"
