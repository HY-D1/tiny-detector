"""
Evaluation script for TinyDetector.

- Loads the best saved model checkpoint.
- Evaluates on the validation split.
- Prints metrics (accuracy, precision, recall, F1).
- Saves a confusion matrix plot.
- Saves a CSV of predictions for failure analysis.

Run with:
    python -m src.eval
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from .config import cfg
from .datasets import load_split
from .model import build_model
from .tokenization import get_tokenizer, make_collate_fn


def load_best_model(device: torch.device) -> torch.nn.Module:
    """
    Load the best saved model checkpoint.
    """
    checkpoint_path = cfg.checkpoints_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Have you run training and saved a model?"
        )

    model = build_model(num_labels=cfg.num_labels)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_on_val() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    val_ds = load_split("val")
    texts: List[str] = val_ds.texts
    true_labels: List[int] = val_ds.labels

    # Model + tokenizer
    model = load_best_model(device)
    tokenizer = get_tokenizer()
    collate_fn = make_collate_fn(tokenizer)

    # DataLoader
    from torch.utils.data import DataLoader

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_preds: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # Convert to numpy arrays
    y_true = np.array(true_labels)
    y_pred = np.array(all_preds)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\nValidation accuracy: {acc:.4f}\n")

    target_names = [cfg.id2label[i] for i in range(cfg.num_labels)]
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.num_labels)))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save confusion matrix plot
    reports_dir = cfg.project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm_path = reports_dir / "confusion_matrix_val.png"

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cfg.num_labels),
        yticks=np.arange(cfg.num_labels),
        xticklabels=target_names,
        yticklabels=target_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Validation)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"\nSaved confusion matrix plot to: {cm_path}")

    # Save predictions CSV for failure analysis
    preds_path = reports_dir / "val_predictions.csv"

    rows = []
    for text, true_id, pred_id, probs in zip(
        texts, true_labels, all_preds, all_probs
    ):
        true_name = cfg.id2label[true_id]
        pred_name = cfg.id2label[pred_id]
        row = {
            "text": text,
            "true_label_id": true_id,
            "true_label_name": true_name,
            "pred_label_id": pred_id,
            "pred_label_name": pred_name,
        }
        # Add probability columns
        for i, p in enumerate(probs):
            row[f"prob_{cfg.id2label[i]}"] = float(p)
        rows.append(row)

    preds_df = pd.DataFrame(rows)
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved validation predictions to: {preds_path}")


if __name__ == "__main__":
    evaluate_on_val()
