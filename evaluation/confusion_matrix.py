"""Confusion matrix utilities for final evaluation."""

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import (
    class_accuracy_from_confusion_matrix,
    overall_accuracy_from_confusion_matrix,
    overall_error_rate_from_confusion_matrix,
)


def ensure_dir(path: Path) -> None:
    """Create a folder if it is missing."""
    path.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a labeled confusion matrix figure."""
    ensure_dir(save_path.parent)

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_positions = np.arange(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=45)
    plt.yticks(tick_positions, class_names)

    threshold = confusion_matrix.max() / 2 if confusion_matrix.size > 0 else 0

    for row_index in range(confusion_matrix.shape[0]):
        for col_index in range(confusion_matrix.shape[1]):
            value = confusion_matrix[row_index, col_index]
            plt.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def write_confusion_matrix_report(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> None:
    """Write a short report explaining the confusion matrix results."""
    ensure_dir(save_path.parent)

    class_accuracies = class_accuracy_from_confusion_matrix(
        confusion_matrix,
        class_names,
    )
    overall_accuracy = overall_accuracy_from_confusion_matrix(confusion_matrix)
    overall_error_rate = overall_error_rate_from_confusion_matrix(confusion_matrix)

    lines = [
        "Confusion Matrix Report",
        "=======================",
        "",
        "Meaning:",
        "- Main diagonal values are correct classifications.",
        "- Off-diagonal values are wrong classifications.",
        "- Rows represent actual classes.",
        "- Columns represent predicted classes.",
        "",
        "Class accuracies:",
    ]

    for class_name, accuracy in class_accuracies.items():
        lines.append(f"- {class_name}: {accuracy:.2f}%")

    lines.extend(
        [
            "",
            f"Overall accuracy: {overall_accuracy:.2f}%",
            f"Overall error rate: {overall_error_rate:.2f}%",
        ]
    )

    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
