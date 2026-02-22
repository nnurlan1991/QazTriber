#!/bin/bash
set -e
export MACOSX_DEPLOYMENT_TARGET=14.0

echo "🔧 Сборка QazTriber..."

# Используем .spec файл, который содержит NSMicrophoneUsageDescription
./build_env/bin/pyinstaller --noconfirm --clean QazTriber.spec

echo ""
echo "✅ Готово! Приложение: dist/QazTriber.app"
