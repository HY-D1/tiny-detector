"""
Inference utilities for TinyDetector.
"""

from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer

from .config import cfg
from .model import build_model


def load_model_and_tokenizer(checkpoint_path: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = build_model(num_labels=cfg.num_labels)

    if checkpoint_path is None:
        checkpoint_path = str(cfg.checkpoints_dir / "best_model.pt")

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device | str = "cpu",
    threshold: float = 0.0,
) -> List[Dict]:
    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.eval()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    ).to(device)

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)

    max_prob, pred = probs.max(dim=-1)

    results = []
    for i, text in enumerate(texts):
        pred_id = int(pred[i].item())
        conf = float(max_prob[i].item())

        label_name = cfg.id2label[pred_id]
        if threshold > 0.0 and conf < threshold:
            label_name = "uncertain"

        probs_dict = {cfg.id2label[j]: float(probs[i, j].item()) for j in range(cfg.num_labels)}

        results.append(
            {
                "text": text,
                "label_id": pred_id,
                "label_name": label_name,
                "confidence": conf,
                "probs": probs_dict,
            }
        )

    return results
