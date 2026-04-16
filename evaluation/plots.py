"""Plot helpers for Part 2 reporting."""

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    """Create an output folder if it is missing."""
    path.mkdir(parents=True, exist_ok=True)


def plot_accuracy_vs_iterations(
    train_accuracies: List[float],
    val_accuracies: List[float],
    save_path: Path,
) -> None:
    """Save the required accuracy-vs-iterations curve."""
    ensure_dir(save_path.parent)

    iterations = list(range(1, len(train_accuracies) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, train_accuracies, marker="o", label="Training Accuracy")
    plt.plot(iterations, val_accuracies, marker="o", label="Validation Accuracy")
    plt.title("Accuracy vs Number of Iterations")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(iterations)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_kfold_accuracies(
    fold_accuracies: List[float],
    average_accuracy: float,
    save_path: Path,
) -> None:
    """Save a bar chart showing each K-fold accuracy."""
    ensure_dir(save_path.parent)

    fold_numbers = list(range(1, len(fold_accuracies) + 1))

    plt.figure(figsize=(8, 5))
    plt.bar(fold_numbers, fold_accuracies)
    plt.axhline(
        average_accuracy,
        color="red",
        linestyle="--",
        label=f"Average = {average_accuracy:.2f}%",
    )
    plt.title("K-Fold Cross Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.xticks(fold_numbers)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
