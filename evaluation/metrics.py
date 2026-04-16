"""Shared metric helpers for the assignment.

These functions are kept small on purpose so they are easy to explain in
the report and presentation.
"""

from typing import Dict, List

import numpy as np


def calculate_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate overall accuracy as a percentage."""
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)

    if len(y_true_array) == 0:
        return 0.0

    correct = np.sum(y_true_array == y_pred_array)
    return (correct / len(y_true_array)) * 100


def calculate_error_rate(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate overall error rate as a percentage."""
    return 100.0 - calculate_accuracy(y_true, y_pred)


def class_accuracy_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Calculate accuracy for each class from the confusion matrix."""
    class_accuracies = {}

    for index, class_name in enumerate(class_names):
        class_total = np.sum(confusion_matrix[index, :])
        correct = confusion_matrix[index, index]

        if class_total == 0:
            class_accuracies[class_name] = 0.0
        else:
            class_accuracies[class_name] = (correct / class_total) * 100

    return class_accuracies


def overall_accuracy_from_confusion_matrix(confusion_matrix: np.ndarray) -> float:
    """Calculate overall accuracy using the diagonal of the confusion matrix."""
    total_samples = np.sum(confusion_matrix)

    if total_samples == 0:
        return 0.0

    correct_samples = np.trace(confusion_matrix)
    return (correct_samples / total_samples) * 100


def overall_error_rate_from_confusion_matrix(confusion_matrix: np.ndarray) -> float:
    """Calculate error rate from the confusion matrix."""
    return 100.0 - overall_accuracy_from_confusion_matrix(confusion_matrix)
