"""A few simple helper functions for Model 2."""

import random
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seeds so the results are more stable from run to run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    """Create a folder if it does not exist yet."""
    path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Use GPU when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Return accuracy for one batch as a percentage."""
    predictions = torch.argmax(outputs, dim=1)
    correct_predictions = (predictions == labels).sum().item()
    accuracy = (correct_predictions / labels.size(0)) * 100
    return accuracy
