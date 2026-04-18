import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return numerator / (denominator + eps)


def save_json(data: Any, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

