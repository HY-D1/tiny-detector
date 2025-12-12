import os
import pytest

import torch

from src.config import cfg
from src.inference import load_model_and_tokenizer, predict


def test_predict_probabilities_sum_to_one():
    ckpt = cfg.checkpoints_dir / "best_model.pt"
    if not ckpt.exists():
        pytest.skip("No checkpoint found; run training first to create checkpoints/best_model.pt")

    tokenizer, model = load_model_and_tokenizer()
    out = predict(["hello world"], tokenizer, model, device="cpu")[0]
    probs = out["probs"]

    assert set(probs.keys()) == {"safe", "toxic", "hate"}
    s = sum(probs.values())
    assert abs(s - 1.0) < 1e-4
