"""
Model definition and loading helpers for TinyDetector.
"""

from transformers import AutoModelForSequenceClassification
from .config import cfg


def build_model(num_labels: int = cfg.num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
    )
    return model
