"""
Tokenizer and collate function for TinyDetector.
"""

from typing import Dict, List, Callable

import torch
from transformers import AutoTokenizer

from .config import cfg


_tokenizer = None


def get_tokenizer():
    """
    Lazy-load the tokenizer so it's only created once.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return _tokenizer


def make_collate_fn(tokenizer) -> Callable:
    """
    Returns a collate_fn that:
    - tokenizes a batch of texts
    - packs them into tensors
    - attaches labels
    """
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )

        encoded["labels"] = labels
        return encoded

    return collate
