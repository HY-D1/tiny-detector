"""
Global configuration for TinyDetector.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    checkpoints_dir: Path = project_root / "checkpoints"

    # Model
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128

    # Labels
    id2label: Dict[int, str] = None
    label2id: Dict[str, int] = None
    num_labels: int = 3  # safe, toxic, hate

    # Training hyperparameters
    seed: int = 42
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_train_samples: int | None = 2000  # for quick debug runs


cfg = Config()
cfg.id2label = {0: "safe", 1: "toxic", 2: "hate"}
cfg.label2id = {v: k for k, v in cfg.id2label.items()}

# Ensure checkpoint dir exists
cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
