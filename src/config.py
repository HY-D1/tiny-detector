"""
Global configuration for TinyDetector.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"

    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    num_labels: int = 3  # safe, toxic, hate

    seed: int = 42


cfg = Config()
