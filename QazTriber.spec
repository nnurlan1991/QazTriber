# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import customtkinter
import ctranslate2
import sounddevice

# Paths for assets
ctk_path = os.path.dirname(customtkinter.__file__)
ct2_path = os.path.dirname(ctranslate2.__file__)
# Fix sounddevice data path: it's a sibling or in same site-packages
sd_data_path = os.path.join(os.path.dirname(sounddevice.__file__), '_sounddevice_data')
# Also check for .dylibs in ctranslate2
ct2_dylibs = os.path.join(ct2_path, '.dylibs')

block_cipher = None

# Collect hidden imports
from PyInstaller.utils.hooks import collect_submodules
hidden_imports = collect_submodules('faster_whisper')
hidden_imports += collect_submodules('ctranslate2')
hidden_imports += [
    'PIL._tkinter_finder',
    'sounddevice',
    'soundfile',
    'librosa',
    'torch',
    'torchaudio',
    'customtkinter',
    'numpy',
    'engine.whisper_engine'
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        (ctk_path, 'customtkinter/'),
        (ct2_dylibs, 'ctranslate2/.dylibs') if os.path.exists(ct2_dylibs) else (ct2_path, 'ctranslate2/'),
        (sd_data_path, '_sounddevice_data/') if os.path.exists(sd_data_path) else None,
        ('engine', 'engine'),
        ('models', 'models'),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'notebook', 'scipy.stats'],
    noarchive=False,
)
# Filter out None from datas
a.datas = [d for d in a.datas if d is not None]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QazTriber',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='QazTriber',
)
app = BUNDLE(
    coll,
    name='QazTriber.app',
    icon=None,
    bundle_identifier='com.nurlan.qaztriber',
    info_plist={
        'NSMicrophoneUsageDescription': 'QazTriber needs access to your microphone to record audio for transcription.',
        'LSMinimumSystemVersion': '14.0',
        'CFBundleShortVersionString': '5.0.0',
        'CFBundleVersion': '5.0.0',
        'NSHighResolutionCapable': True,
    },
)
