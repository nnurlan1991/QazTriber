#!/usr/bin/env python3
"""Квантизация punct-restore модели: создаёт fp16 и int8 копии.

Запуск:
    cd backend && .venv/bin/python -m app.services.quantize_punct

Создаёт:
    <models_dir>/punct_v2_fp16/  — float16 (530 MB, для MPS/CUDA)
    <models_dir>/punct_v2_int8/  — int8 dynamic (280 MB, для CPU)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from app.config import settings
from app.services.punct_restore import PunctCaseModel


def quantize_fp16(src: Path, dst: Path) -> None:
    """Сохраняет модель в float16 — вдвое меньше RAM, без потери качества."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    print(f"[fp16] Загрузка из {src} ...")
    config = AutoConfig.from_pretrained(src)
    model = PunctCaseModel.from_pretrained(src, config=config)
    model = model.half().eval()
    print(f"[fp16] Сохранение в {dst} ...")
    model.save_pretrained(dst, safe_serialization=True)
    # Копируем токенайзер
    for f in src.iterdir():
        if f.name.startswith("tokenizer") or f.name.startswith("sentencepiece") or f.name.startswith("special_tokens"):
            shutil.copy2(f, dst / f.name)
    size = sum(f.stat().st_size for f in dst.iterdir()) / 1024 / 1024
    print(f"[fp16] Готово: {size:.0f} MB")


def quantize_int8(src: Path, dst: Path) -> None:
    """Ручная int8 квантизация ВСЕХ весов (включая embeddings).

    PyTorch dynamic quantization на macOS не работает (нет FBGEMM), а ONNX
    dynamic quantization не трогает embeddings (768 MB из 1060 MB). Поэтому
    квантизуем вручную: каждый тензор → int8 + scale, сохраняем через safetensors.
    При загрузке PunctRestoreService конвертирует обратно в float16.
    """
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    print(f"[int8] Загрузка из {src} ...")
    config = AutoConfig.from_pretrained(src)
    model = PunctCaseModel.from_pretrained(src, config=config).eval()

    state_dict = model.state_dict()
    quantized_sd: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    total_orig = 0
    total_quant = 0

    for name, param in state_dict.items():
        if param.dtype == torch.float32 and param.numel() > 1000:
            # Симметричная int8 квантизация
            abs_max = param.abs().max().item()
            if abs_max == 0:
                abs_max = 1.0
            scale = abs_max / 127.0
            q = torch.round(param / scale).to(torch.int8)
            quantized_sd[name] = q
            scales[name] = scale
            total_orig += param.numel() * 4
            total_quant += param.numel() * 1
        else:
            quantized_sd[name] = param
            total_orig += param.numel() * param.element_size()
            total_quant += param.numel() * param.element_size()

    print(f"[int8] Сохранение в {dst} ...")
    from safetensors.torch import save_file
    save_file(quantized_sd, str(dst / "model.safetensors"))

    # Сохраняем scales
    import json
    (dst / "quant_scales.json").write_text(json.dumps(scales), encoding="utf-8")

    # Копируем токенайзер и config
    for f in src.iterdir():
        if f.name.startswith("tokenizer") or f.name.startswith("sentencepiece") or f.name.startswith("special_tokens") or f.name == "config.json":
            shutil.copy2(f, dst / f.name)

    size = sum(f.stat().st_size for f in dst.iterdir() if f.is_file()) / 1024 / 1024
    print(f"[int8] Готово: {size:.0f} MB (веса: {total_orig/1024/1024:.0f}→{total_quant/1024/1024:.0f} MB)")


def main() -> None:
    src = settings.punct_model_dir
    if not (src / "config.json").is_file():
        print(f"ОШИБКА: модель не найдена в {src}")
        sys.exit(1)

    models_dir = settings.models_dir
    fp16_dst = models_dir / "punct_v2_fp16"
    int8_dst = models_dir / "punct_v2_int8"

    print(f"Источник: {src}")
    print(f"Размер оригинала: {sum(f.stat().st_size for f in src.iterdir() if f.is_file()) / 1024 / 1024:.0f} MB")
    print()

    quantize_fp16(src, fp16_dst)
    print()
    quantize_int8(src, int8_dst)

    print()
    print("=== ИТОГ ===")
    for name, path in [("orig (fp32)", src), ("fp16", fp16_dst), ("int8", int8_dst)]:
        size = sum(f.stat().st_size for f in path.iterdir() if f.is_file()) / 1024 / 1024
        print(f"  {name:15s}: {size:6.0f} MB  ({path})")


if __name__ == "__main__":
    main()
