"""
Evaluation script for TinyDetector.

Run:
  python -m src.eval
  python -m src.eval --checkpoint checkpoints/best_model.pt
  python -m src.eval --checkpoint checkpoints/best_model.pt --tag baseline
  python -m src.eval --checkpoint checkpoints/best_model.pt --out-dir reports

Outputs (by default to reports/):
- metrics_{tag}.json
- confusion_matrix_{tag}.png
- val_predictions_{tag}.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from .config import cfg
from .datasets import load_split
from .model import build_model
from .tokenization import get_tokenizer, make_collate_fn


def load_model(device: torch.device, checkpoint_path: Path) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(num_labels=cfg.num_labels)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Pred
