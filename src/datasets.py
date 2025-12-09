"""
Data loading utilities for TinyDetector.
"""

from typing import List, Dict
from torch.utils.data import Dataset


class SafetyDataset(Dataset):
    """Placeholder dataset implementation."""

    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }
