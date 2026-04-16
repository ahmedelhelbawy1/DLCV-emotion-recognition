"""K-fold validation for Model 2.

This script trains Model 2 several times using different validation folds.
The assignment says the minimum K value is 4, so the default here is 4.
"""

from pathlib import Path
import argparse
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.plots import plot_kfold_accuracies
from model2.model2_emotion_cnn import EmotionCNN
from model2.train import train_one_epoch, validate_one_epoch
from model2.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Read K-fold settings from the command line."""
    parser = argparse.ArgumentParser(description="Run K-fold validation for Model 2.")
    parser.add_argument("--k", type=int, default=4, help="Number of folds. Minimum is 4.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=512, help="Input image size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden-features", type=int, default=128, help="Hidden layer size.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional balanced subset limit. Leave empty for full dataset.",
    )
    parser.add_argument(
        "--subset-justification",
        type=str,
        default=(
            "A subset was used only when full K-fold training was too slow on CPU. "
            "The subset is balanced across the four emotion classes."
        ),
        help="Explanation written into the report when --max-samples is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def get_transform(image_size: int) -> transforms.Compose:
    """Use the same preprocessing idea as Model 2 training."""
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


def write_kfold_report(
    fold_accuracies: List[float],
    average_accuracy: float,
    save_path: Path,
    k: int,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
    total_dataset_size: int,
    samples_used: int,
    max_samples: Optional[int],
    subset_justification: str,
) -> None:
    """Save fold accuracies and their average."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Model 2 K-Fold Validation Results",
        "=================================",
        "",
        "Settings:",
        f"- K value: {k}",
        f"- epochs per fold: {epochs}",
        f"- batch size: {batch_size}",
        f"- image size: {image_size} x {image_size}",
        f"- learning rate: {learning_rate}",
        f"- total available training samples: {total_dataset_size}",
        f"- samples used in K-fold: {samples_used}",
        "",
    ]

    if max_samples is not None:
        lines.extend(
            [
                "Subset justification:",
                subset_justification,
                "",
            ]
        )

    for index, accuracy in enumerate(fold_accuracies, start=1):
        lines.append(f"Fold {index} accuracy: {accuracy:.2f}%")

    lines.extend(
        [
            "",
            f"Average K-fold accuracy: {average_accuracy:.2f}%",
        ]
    )

    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_balanced_indices(dataset: datasets.ImageFolder, max_samples: Optional[int]) -> np.ndarray:
    """Return all indices or a balanced subset across classes."""
    if max_samples is None or max_samples >= len(dataset):
        return np.arange(len(dataset))

    class_indices = {}
    for index, target in enumerate(dataset.targets):
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

    return np.array(selected_indices)


def main() -> None:
    """Run K-fold cross-validation."""
    args = parse_args()

    if args.k < 4:
        raise ValueError("The assignment requires K to be at least 4.")

    set_seed(args.seed)
    device = get_device()

    dataset_dir = PROJECT_ROOT / "dataset" / "train"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    dataset = datasets.ImageFolder(
        root=dataset_dir,
        transform=get_transform(args.image_size),
    )

    indices = get_balanced_indices(dataset, args.max_samples)

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    criterion = nn.CrossEntropyLoss()
    fold_accuracies = []

    print(f"Running {args.k}-fold validation on device: {device}")
    print(f"Total samples used: {len(indices)}")

    for fold_number, (train_indices, val_indices) in enumerate(kfold.split(indices), start=1):
        print(f"\nFold {fold_number}/{args.k}")

        train_subset = Subset(dataset, indices[train_indices])
        val_subset = Subset(dataset, indices[val_indices])

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = EmotionCNN(
            num_classes=len(dataset.classes),
            input_size=args.image_size,
            hidden_features=args.hidden_features,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.epochs):
            train_loss, train_accuracy = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            val_loss, val_accuracy = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Acc: {train_accuracy:.2f}% | "
                f"Val Acc: {val_accuracy:.2f}%"
            )

        fold_accuracies.append(val_accuracy)

    average_accuracy = float(np.mean(fold_accuracies))

    report_path = PROJECT_ROOT / "outputs" / "reports" / "kfold_results_model2.txt"
    plot_path = PROJECT_ROOT / "outputs" / "plots" / "model2" / "kfold_accuracy_model2.png"

    write_kfold_report(
        fold_accuracies=fold_accuracies,
        average_accuracy=average_accuracy,
        save_path=report_path,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        total_dataset_size=len(dataset),
        samples_used=len(indices),
        max_samples=args.max_samples,
        subset_justification=args.subset_justification,
    )
    plot_kfold_accuracies(fold_accuracies, average_accuracy, plot_path)

    print("\nK-fold validation finished.")
    print(f"Average accuracy: {average_accuracy:.2f}%")
    print(f"Report saved to: {report_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
