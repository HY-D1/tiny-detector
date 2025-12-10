"""
Inference utilities for TinyDetector.
"""

from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import cfg


def load_model_and_tokenizer(
    checkpoint_path: str | None = None,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load tokenizer + model.
    If checkpoint_path is None, load from cfg.checkpoints_dir / 'best_model.pt'.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
    )

    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoints_dir / "best_model.pt"

    checkpoint_path = str(checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def predict(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: str | torch.device = "cpu",
) -> List[Dict]:
    """
    Predict labels for a list of texts.
    Returns a list of dicts: { 'text', 'label_id', 'label_name', 'probs' }
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    ).to(device)

    outputs = model(**encoded)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    preds = torch.argmax(probs, dim=-1).cpu().tolist()
    probs = probs.cpu().tolist()

    results = []
    for text, pred_id, prob_vec in zip(texts, preds, probs):
        label_name = cfg.id2label[pred_id]
        probs_dict = {
            cfg.id2label[i]: float(p)
            for i, p in enumerate(prob_vec)
        }
        results.append(
            {
                "text": text,
                "label_id": pred_id,
                "label_name": label_name,
                "probs": probs_dict,
            }
        )
    return results
