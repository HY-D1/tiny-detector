"""
Data loading utilities for TinyDetector.
"""

from typing import List, Dict, Tuple
import pandas as pd
from torch.utils.data import Dataset

from .config import cfg


class SafetyDataset(Dataset):
    """Dataset storing raw texts and integer labels."""

    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        return {"text": self.texts[idx], "label": int(self.labels[idx])}


def load_split(split: str = "train") -> SafetyDataset:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    csv_path = cfg.processed_data_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing processed file: {csv_path}. "
            "Generate it from notebooks/exploration.ipynb."
        )

    df = pd.read_csv(csv_path)
    texts = df["comment_text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return SafetyDataset(texts=texts, labels=labels)


def get_label_mappings() -> Tuple[dict, dict]:
    return cfg.id2label, cfg.label2id
