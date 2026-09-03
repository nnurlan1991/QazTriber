from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

PUNCT_LABELS = ["O", "COMMA", "PERIOD", "QUESTION", "EXCLAM"]
ID2PUNCT = {i: label for i, label in enumerate(PUNCT_LABELS)}
PUNCT_SUFFIX = {"COMMA": ",", "PERIOD": ".", "QUESTION": "?", "EXCLAM": "!"}

MAX_LEN = 128           # лимит сабвордов, с которым модель обучалась
MAX_WORDS = 50          # максимум слов на окно (50 слов → ~118 сабвордов, влезает в 128)
OVERLAP_WORDS = 10      # перекрытие между окнами
BATCH_WINDOWS = 8       # сколько окон гонять за один forward (L2.2)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _empty_cache() -> None:
    """Освобождает кэш MPS/CUDA (L1.4)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass
    gc.collect()


class PunctCaseModel(PreTrainedModel):
    """Энкодер + две головы (пунктуация, регистр).

    Архитектура 1:1 с обучающей (inference_v2.py): энкодер назван ``roberta``,
    ``base_model_prefix = "roberta"`` — иначе ``from_pretrained`` не сопоставит
    ключи state_dict (веса лежат под ``roberta.*``).
    """

    config_class = AutoConfig.from_pretrained("xlm-roberta-base").__class__
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = AutoModel.from_config(config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.punct_head = nn.Linear(hidden, len(PUNCT_LABELS))
        self.case_head = nn.Linear(hidden, 2)

    def forward(self, input_ids=None, attention_mask=None, punct_labels=None, case_labels=None, **_):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        pl = self.punct_head(seq)
        cl = self.case_head(seq)
        loss = None
        if punct_labels is not None and case_labels is not None:
            lf = nn.CrossEntropyLoss(ignore_index=-100)
            loss = lf(pl.view(-1, len(PUNCT_LABELS)), punct_labels.view(-1)) + lf(cl.view(-1, 2), case_labels.view(-1))
        return {"loss": loss, "punct_logits": pl, "case_logits": cl}


class PunctRestoreService:
    """Восстановление пунктуации и регистра в тексте, полученном от GigaAM.

    Модель загружается лениво (fp16 на MPS/CUDA), выгружается явно сразу после
    использования (как GigaAM). Инференс батчится по BATCH_WINDOWS окон за forward,
    используется ``torch.inference_mode`` для экономии памяти.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._model: PunctCaseModel | None = None
        self._tokenizer = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return (self.model_dir / "config.json").is_file()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.is_available:
                raise FileNotFoundError(
                    f"Модель восстановления пунктуации не найдена в {self.model_dir}. "
                    "Скопируйте туда содержимое punct_case_model_v2_final/."
                )
            logger.info("Загрузка punct-restore модели из %s", self.model_dir)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            config = AutoConfig.from_pretrained(self.model_dir)
            device = _pick_device()

            # int8: загружаем вручную, минуя from_pretrained (не принимает int8)
            scales_path = self.model_dir / "quant_scales.json"
            if scales_path.is_file():
                import json
                from safetensors.torch import load_file
                logger.info("Punct-restore: загрузка int8-квантованной модели")
                model = PunctCaseModel(config)
                sd = load_file(str(self.model_dir / "model.safetensors"))
                scales = json.loads(scales_path.read_text(encoding="utf-8"))
                for name, scale in scales.items():
                    if name in sd and sd[name].dtype == torch.int8:
                        sd[name] = sd[name].to(torch.float32) * scale
                model.load_state_dict(sd, strict=True)
                model = model.to(device).eval()
                logger.info("Punct-restore: int8 веса деквантованы (%d тензоров)", len(scales))
            else:
                model = PunctCaseModel.from_pretrained(self.model_dir, config=config).to(device).eval()

            # L1.1: float16 на GPU/MPS — вдвое меньше памяти, без потери качества на inference
            if device.type in {"cuda", "mps"}:
                model = model.half()
            self._model = model
            self._device = device
            logger.info("Punct-restore модель загружена на %s (fp16=%s)", device, device.type in {"cuda", "mps"})

    def unload(self) -> None:
        """L1.2: выгружает модель из памяти."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._device = None
        _empty_cache()

    def _restore_batch(self, windows: list[list[str]]) -> list[list[str]]:
        """L2.2: батчит несколько окон в один forward pass."""
        tok = self._tokenizer
        device = self._device
        # Токенизируем все окна в один батч
        encodings = [
            tok(
                w,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
            )
            for w in windows
        ]
        # padding до одинаковой длины
        batch_input_ids = []
        batch_attention_mask = []
        all_word_ids = []
        max_len_batch = max(e["input_ids"].shape[1] for e in encodings)
        for enc, w in zip(encodings, windows):
            seq_len = enc["input_ids"].shape[1]
            pad_len = max_len_batch - seq_len
            batch_input_ids.append(torch.cat([enc["input_ids"][0], torch.zeros(pad_len, dtype=torch.long)]))
            batch_attention_mask.append(torch.cat([enc["attention_mask"][0], torch.zeros(pad_len, dtype=torch.long)]))
            all_word_ids.append(enc.word_ids())

        input_ids = torch.stack(batch_input_ids).to(device)
        attention_mask = torch.stack(batch_attention_mask).to(device)
        if device.type in {"cuda", "mps"}:
            input_ids = input_ids.half() if False else input_ids  # input_ids остаются long

        # L2.3: inference_mode — легче чем no_grad
        with torch.inference_mode():
            out = self._model(input_ids=input_ids, attention_mask=attention_mask)

        punct_logits = out["punct_logits"]  # [batch, seq, 5]
        case_logits = out["case_logits"]    # [batch, seq, 2]

        results: list[list[str]] = []
        for b, (word_ids, words) in enumerate(zip(all_word_ids, windows)):
            punct_ids = punct_logits[b].argmax(-1).cpu().tolist()
            case_ids = case_logits[b].argmax(-1).cpu().tolist()
            result: list[str] = []
            seen: set[int] = set()
            for idx, wid in enumerate(word_ids):
                if wid is None or wid in seen or wid >= len(words):
                    continue
                if idx >= len(case_ids):
                    break
                seen.add(wid)
                word = words[wid]
                if case_ids[idx] == 1:
                    word = word[0:1].upper() + word[1:]
                label = ID2PUNCT[int(punct_ids[idx])]
                word += PUNCT_SUFFIX.get(label, "")
                result.append(word)
            results.append(result)
        return results

    def _split_windows(self, words: list[str]) -> list[tuple[int, list[str]]]:
        """Нарезает слова на окна, гарантируя что каждое окно ≤ MAX_LEN сабвордов.
        Казахские слова длинные → 100 слов могут дать 200+ сабвордов и обрезаться
        truncation. Динамически подбираем размер окна по реальному числу сабвордов.
        """
        tok = self._tokenizer
        windows: list[tuple[int, list[str]]] = []
        i = 0
        while i < len(words):
            # Бинарный поиск: сколько слов влезет в MAX_LEN сабвордов
            lo, hi = 1, min(MAX_WORDS, len(words) - i)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                n_sub = len(tok(words[i : i + mid], is_split_into_words=True)["input_ids"])
                if n_sub <= MAX_LEN:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            window = words[i : i + best]
            windows.append((i, window))
            step = max(1, best - OVERLAP_WORDS)
            i += step
        return windows

    def restore(self, raw_text: str) -> str:
        """Восстанавливает пунктуацию/регистр в тексте произвольной длины,
        нарезая его окнами со скользящим перекрытием, батчит по BATCH_WINDOWS
        окон за forward pass (L2.2)."""
        self._ensure_loaded()
        words = raw_text.strip().split()
        if not words:
            return raw_text

        windows = self._split_windows(words)

        # L2.2: обрабатываем батчами
        out_words: list[str] = []
        for batch_start in range(0, len(windows), BATCH_WINDOWS):
            batch = windows[batch_start : batch_start + BATCH_WINDOWS]
            batch_windows = [w for _, w in batch]
            restored_batch = self._restore_batch(batch_windows)

            for j, (start_idx, _) in enumerate(batch):
                restored = restored_batch[j]
                skip = 0 if start_idx == 0 else OVERLAP_WORDS
                out_words.extend(restored[skip:])

        text = " ".join(out_words)
        if text.endswith(","):
            text = text[:-1] + "."
        elif text and text[-1] not in ".!?":
            text += "."
        return text
