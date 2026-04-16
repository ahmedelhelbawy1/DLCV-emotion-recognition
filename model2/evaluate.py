"""Evaluate the saved Model 2 CNN on the test set."""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.confusion_matrix import write_confusion_matrix_report
from model2.model2_emotion_cnn import EmotionCNN
from model2.utils import ensure_dir, get_device


IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_CLASSES = 4
HIDDEN_FEATURES = 128


def parse_args() -> argparse.Namespace:
    """Read evaluation settings from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate Model 2 CNN.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "best_model2.pth",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def get_test_transform(image_size: int) -> transforms.Compose:
    """Use the same image preprocessing during testing."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def limit_dataset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    """Use a smaller balanced subset when we need a quick local test."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    if hasattr(dataset, "targets"):
        targets = dataset.targets
        class_indices = {}

        for index, target in enumerate(targets):
            class_indices.setdefault(target, []).append(index)

        samples_per_class = max_samples // len(class_indices)
        selected_indices = []

        for target in sorted(class_indices):
            selected_indices.extend(class_indices[target][:samples_per_class])

        remaining = max_samples - len(selected_indices)
        if remaining > 0:
            used_indices = set(selected_indices)
            for index in range(len(dataset)):
                if index not in used_indices:
                    selected_indices.append(index)
                    remaining -= 1
                if remaining == 0:
                    break

        return Subset(dataset, selected_indices)

    return Subset(dataset, list(range(max_samples)))


def load_test_dataset(
    image_size: int,
    max_test_samples: Optional[int],
) -> Tuple[Dataset, List[str], int]:
    """Load the test dataset and print some basic information."""
    test_dir = PROJECT_ROOT / "dataset" / "test"

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Required dataset folder was not found: {test_dir}\n"
            "Please make sure dataset/test exists."
        )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_test_transform(image_size),
    )

    print(f"Test classes found: {test_dataset.classes}")
    print(f"Number of test images: {len(test_dataset)}")

    if len(test_dataset) == 0:
        raise ValueError("The test dataset is empty.")

    class_names = test_dataset.classes
    full_test_size = len(test_dataset)
    test_dataset = limit_dataset(test_dataset, max_test_samples)

    if max_test_samples is not None:
        print(f"Quick run test images used: {len(test_dataset)}")

    return test_dataset, class_names, full_test_size


def load_trained_model(
    model_path: Path,
    device: torch.device,
) -> Tuple[EmotionCNN, List[str], int]:
    """Load the saved best checkpoint from training."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model was not found: {model_path}\n"
            "Please train the model first before running evaluation."
        )

    checkpoint = torch.load(model_path, map_location=device)

    class_names = checkpoint.get(
        "class_names",
        ["angry", "happy", "sad", "surprise"],
    )
    hidden_features = checkpoint.get("hidden_features", HIDDEN_FEATURES)
    input_size = checkpoint.get("input_size", IMAGE_SIZE)

    model = EmotionCNN(
        num_classes=NUM_CLASSES,
        input_size=input_size,
        hidden_features=hidden_features,
    ).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, class_names, input_size


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
) -> None:
    """Draw and save the confusion matrix using matplotlib only."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Model 2")
    plt.colorbar()

    tick_positions = np.arange(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=45)
    plt.yticks(tick_positions, class_names)

    threshold = cm.max() / 2.0 if cm.size > 0 else 0
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            plt.text(
                col_index,
                row_index,
                str(cm[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if cm[row_index, col_index] > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    """Run evaluation and save the final reports."""
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    model, checkpoint_classes, input_size = load_trained_model(
        model_path=args.model_path,
        device=device,
    )
    print("Saved model loaded successfully.")
    print(f"Checkpoint image size: {input_size}x{input_size}")

    print("Loading test data...")
    test_dataset, test_class_names, full_test_size = load_test_dataset(
        image_size=input_size,
        max_test_samples=args.max_test_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if checkpoint_classes != test_class_names:
        print("Warning: Class names in the saved model and test dataset do not match.")
        print(f"Checkpoint classes: {checkpoint_classes}")
        print(f"Test dataset classes: {test_class_names}")

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # We only need the class with the highest score.
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    test_accuracy = (all_predictions == all_labels).mean() * 100
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=test_class_names,
        digits=4,
        zero_division=0,
    )

    plots_output_dir = PROJECT_ROOT / "outputs" / "plots"
    reports_output_dir = PROJECT_ROOT / "outputs" / "reports"
    ensure_dir(plots_output_dir)
    ensure_dir(reports_output_dir)

    confusion_matrix_path = plots_output_dir / "confusion_matrix_model2.png"
    classification_report_path = reports_output_dir / "classification_report.txt"
    test_results_path = reports_output_dir / "test_results.txt"
    confusion_matrix_report_path = reports_output_dir / "confusion_matrix_report_model2.txt"

    plot_confusion_matrix(
        cm=cm,
        class_names=test_class_names,
        save_path=confusion_matrix_path,
    )

    classification_report_text = (
        "Model 2 Classification Report\n"
        "=============================\n\n"
        f"{report}"
    )
    classification_report_path.write_text(classification_report_text, encoding="utf-8")
    write_confusion_matrix_report(
        confusion_matrix=cm,
        class_names=test_class_names,
        save_path=confusion_matrix_report_path,
    )

    test_results_text = (
        "Model 2 Test Results\n"
        "====================\n"
        f"Test Accuracy: {test_accuracy:.2f}%\n"
        f"Number of Test Images Used: {len(test_dataset)}\n"
        f"Full Test Dataset Size: {full_test_size}\n"
        f"Classes: {', '.join(test_class_names)}\n"
        f"Saved Model Path: {args.model_path}\n"
        f"Confusion Matrix Path: {confusion_matrix_path}\n"
        f"Confusion Matrix Report Path: {confusion_matrix_report_path}\n"
        f"Classification Report Path: {classification_report_path}\n"
    )
    test_results_path.write_text(test_results_text, encoding="utf-8")

    print("\nEvaluation finished.")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print("\nClassification Report:")
    print(report)
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print(f"Confusion matrix report saved to: {confusion_matrix_report_path}")
    print(f"Classification report saved to: {classification_report_path}")
    print(f"Test summary saved to: {test_results_path}")


if __name__ == "__main__":
    main()
