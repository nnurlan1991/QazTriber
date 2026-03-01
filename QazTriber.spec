# -*- mode: python ; coding: utf-8 -*-
import os
import customtkinter

# Get customtkinter path for data files
ctk_path = os.path.dirname(customtkinter.__file__)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        (ctk_path, 'customtkinter/'),
        ('engine', 'engine'),
        ('models', 'models'),
    ],
    hiddenimports=[
        'PIL._tkinter_finder',
        'sounddevice',
        'soundfile',
        'librosa',
        'torch',
        'torchaudio',
        'faster_whisper',
        'ctranslate2',
        'customtkinter',
        'numpy'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QazTriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    upx=True,
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
