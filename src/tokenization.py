"""
Tokenizer and collate function for TinyDetector.
"""

from typing import Dict, List, Callable
import torch
from transformers import AutoTokenizer

from .config import cfg

_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(cfg.model_name)
    return _TOKENIZER


def make_collate_fn(tokenizer) -> Callable:
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return collate
