from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from .config import cfg


class SafetyDataset(Dataset):
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


def load_split(split: str = "train") -> SafetyDataset:
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    csv_path = cfg.processed_data_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed split file not found: {csv_path}. "
            "Did you run the data preparation notebook?"
        )

    df = pd.read_csv(csv_path)
    texts = df["comment_text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return SafetyDataset(texts=texts, labels=labels)


def get_label_mappings() -> Tuple[dict, dict]:
    id2label = {0: "safe", 1: "toxic", 2: "hate"}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id

def get_label_mappings():
    return cfg.id2label, cfg.label2id
