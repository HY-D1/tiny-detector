"""
Model helpers for TinyDetector.
"""

from transformers import AutoModelForSequenceClassification
from .config import cfg


def build_model(num_labels: int = cfg.num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
    )
