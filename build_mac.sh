#!/bin/bash
set -e
export MACOSX_DEPLOYMENT_TARGET=14.0

echo "🔧 Сборка QazTriber..."

# Build via PyInstaller (spec handles all assets and metadata)
./build_env/bin/pyinstaller --noconfirm --clean QazTriber.spec

echo ""
echo "✅ Готово! Приложение: dist/QazTriber.app"
