"""
Evaluation script for TinyDetector.

Run:
  python -m src.eval
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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


def load_best_model(device: torch.device):
    ckpt = cfg.checkpoints_dir / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}. Train first (python -m src.train).")

    model = build_model(num_labels=cfg.num_labels)
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_ds = load_split("val")
    tokenizer = get_tokenizer()
    collate_fn = make_collate_fn(tokenizer)

    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    model = load_best_model(device)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true = np.array(val_ds.labels)
    y_pred = np.array(all_preds)

    target_names = [cfg.id2label[i] for i in range(cfg.num_labels)]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    recalls = recall_score(y_true, y_pred, average=None, labels=list(range(cfg.num_labels)))

    print(f"\nValidation accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.num_labels)))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # save metrics.json
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "recall_safe": float(recalls[0]),
        "recall_toxic": float(recalls[1]),
        "recall_hate": float(recalls[2]),
    }
    metrics_path = cfg.reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    # save confusion matrix png
    cm_path = cfg.reports_dir / "confusion_matrix_val.png"
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

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {cm_path}")

    # save val_predictions.csv
    preds_path = cfg.reports_dir / "val_predictions.csv"
    rows = []
    for text, t, p, prob in zip(val_ds.texts, y_true.tolist(), y_pred.tolist(), all_probs):
        row = {
            "text": text,
            "true_label_id": int(t),
            "true_label_name": cfg.id2label[int(t)],
            "pred_label_id": int(p),
            "pred_label_name": cfg.id2label[int(p)],
        }
        for i, pr in enumerate(prob):
            row[f"prob_{cfg.id2label[i]}"] = float(pr)
        rows.append(row)

    pd.DataFrame(rows).to_csv(preds_path, index=False)
    print(f"Saved validation predictions to: {preds_path}")


if __name__ == "__main__":
    main()
