"""Train Model 2 for emotion classification."""

import argparse
from pathlib import Path
import random
import sys
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.plots import plot_accuracy_vs_iterations
from model2.model2_emotion_cnn import EmotionCNN
from model2.utils import (
    calculate_accuracy,
    ensure_dir,
    get_device,
    set_seed,
)


# Main training settings from the assignment.
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4
HIDDEN_FEATURES = 128
SEED = 42


def parse_args() -> argparse.Namespace:
    """Read training settings from the command line."""
    parser = argparse.ArgumentParser(description="Train Model 2 CNN.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden-features", type=int, default=HIDDEN_FEATURES)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "best_model2.pth",
    )
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Use simple training-only augmentation to help the CNN generalize.",
    )
    return parser.parse_args()


def get_data_transforms(image_size: int, is_training: bool, use_augmentation: bool) -> transforms.Compose:
    """Build the image preprocessing steps for train or validation."""
    transform_steps = [
        # The assignment says images should be 512x512.
        transforms.Resize((image_size, image_size)),
    ]

    if is_training and use_augmentation:
        # These light changes make training images slightly different each epoch.
        # This can help the model avoid memorizing one simple pattern.
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            # Standard normalization values for RGB images.
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(transform_steps)


def validate_dataset_folder(folder_path: Path, folder_name: str) -> None:
    """Stop early with a clear message if a folder is missing."""
    if not folder_path.exists():
        raise FileNotFoundError(
            f"Required dataset folder was not found: {folder_path}\n"
            f"Please make sure dataset/{folder_name} exists."
        )


def limit_dataset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    """Use a smaller balanced subset when we need a quick local run."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    if hasattr(dataset, "targets"):
        targets = dataset.targets
        class_indices = {}

        for index, target in enumerate(targets):
            class_indices.setdefault(target, []).append(index)

        random_generator = random.Random(SEED)
        for indices in class_indices.values():
            random_generator.shuffle(indices)

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


def load_datasets(
    image_size: int,
    max_train_samples: Optional[int],
    max_val_samples: Optional[int],
    use_augmentation: bool,
) -> Tuple[Dataset, Dataset, List[str]]:
    """Load train and validation data from the dataset folder."""
    train_dir = PROJECT_ROOT / "dataset" / "train"
    val_dir = PROJECT_ROOT / "dataset" / "val"

    validate_dataset_folder(train_dir, "train")
    validate_dataset_folder(val_dir, "val")

    train_transform = get_data_transforms(
        image_size=image_size,
        is_training=True,
        use_augmentation=use_augmentation,
    )
    val_transform = get_data_transforms(
        image_size=image_size,
        is_training=False,
        use_augmentation=False,
    )

    # ImageFolder works well here because the dataset is already split
    # into class-based folders like angry/, happy/, sad/, surprise/.
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    print(f"Training classes found: {train_dataset.classes}")
    print(f"Validation classes found: {val_dataset.classes}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty.")

    if len(val_dataset) == 0:
        raise ValueError("The validation dataset is empty.")

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Class names in training and validation folders do not match."
        )

    class_names = train_dataset.classes
    train_dataset = limit_dataset(train_dataset, max_train_samples)
    val_dataset = limit_dataset(val_dataset, max_val_samples)

    if max_train_samples is not None:
        print(f"Quick run training images used: {len(train_dataset)}")

    if max_val_samples is not None:
        print(f"Quick run validation images used: {len(val_dataset)}")

    return train_dataset, val_dataset, class_names


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap datasets in dataloaders so batches can be read easily."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def save_metric_plot(
    values: List[float],
    title: str,
    y_label: str,
    save_path: Path,
) -> None:
    """Save one metric plot after training."""
    epochs = list(range(1, len(values) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_one_epoch(
    model: EmotionCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one full training epoch."""
    model.train()

    running_loss = 0.0
    running_accuracy = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Start this batch with zero gradients from the previous step.
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        batch_accuracy = calculate_accuracy(outputs, labels)
        running_accuracy += batch_accuracy * labels.size(0)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_accuracy / total_samples

    return epoch_loss, epoch_accuracy


def validate_one_epoch(
    model: EmotionCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int]]:
    """Check model performance on the validation set."""
    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0
    total_samples = 0
    prediction_counts = [0 for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Validation is only for checking performance, not updating weights.
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            batch_accuracy = calculate_accuracy(outputs, labels)
            running_accuracy += batch_accuracy * labels.size(0)
            total_samples += labels.size(0)

            predictions = torch.argmax(outputs, dim=1)
            batch_counts = torch.bincount(
                predictions.cpu(),
                minlength=NUM_CLASSES,
            )
            prediction_counts = [
                old_count + int(new_count)
                for old_count, new_count in zip(prediction_counts, batch_counts)
            ]

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_accuracy / total_samples

    return epoch_loss, epoch_accuracy, prediction_counts


def save_best_model(
    model: EmotionCNN,
    class_names: List[str],
    save_path: Path,
    input_size: int,
    hidden_features: int,
) -> None:
    """Save the current best model for testing later."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "input_size": input_size,
        "hidden_features": hidden_features,
    }
    torch.save(checkpoint, save_path)


def main() -> None:
    """Run the full training process."""
    args = parse_args()
    set_seed(SEED)
    device = get_device()

    print(f"Using device: {device}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hidden features: {args.hidden_features}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Training augmentation: {'on' if args.use_augmentation else 'off'}")
    print("Loading training and validation data...")

    train_dataset, val_dataset, class_names = load_datasets(
        image_size=args.image_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        use_augmentation=args.use_augmentation,
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
    )

    model = EmotionCNN(
        num_classes=NUM_CLASSES,
        input_size=args.image_size,
        hidden_features=args.hidden_features,
    ).to(device)

    # CrossEntropyLoss fits this task because the model predicts
    # one class out of 4 possible emotion classes.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    plots_output_dir = PROJECT_ROOT / "outputs" / "plots"
    ensure_dir(args.model_output_path.parent)
    ensure_dir(plots_output_dir)

    best_model_path = args.model_output_path
    best_val_accuracy = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    print("\nTraining started...\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy, val_prediction_counts = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )
        prediction_summary = ", ".join(
            f"{class_name}: {count}"
            for class_name, count in zip(class_names, val_prediction_counts)
        )
        print(f"Validation predictions: {prediction_summary}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_best_model(
                model=model,
                class_names=class_names,
                save_path=best_model_path,
                input_size=args.image_size,
                hidden_features=args.hidden_features,
            )
            print("Validation accuracy improved, so I saved this model.")
            print(f"Saved to: {best_model_path}")
        else:
            print("Validation accuracy did not improve this epoch.")

        print()

    print("Saving training plots...")

    # I kept the plotting directly in train.py so the full training flow
    # stays in one place and is easier to explain.
    save_metric_plot(
        values=history["train_loss"],
        title="Training Loss vs Epochs",
        y_label="Loss",
        save_path=plots_output_dir / "train_loss.png",
    )
    save_metric_plot(
        values=history["val_loss"],
        title="Validation Loss vs Epochs",
        y_label="Loss",
        save_path=plots_output_dir / "val_loss.png",
    )
    save_metric_plot(
        values=history["train_accuracy"],
        title="Training Accuracy vs Epochs",
        y_label="Accuracy (%)",
        save_path=plots_output_dir / "train_accuracy.png",
    )
    save_metric_plot(
        values=history["val_accuracy"],
        title="Validation Accuracy vs Epochs",
        y_label="Accuracy (%)",
        save_path=plots_output_dir / "val_accuracy.png",
    )
    plot_accuracy_vs_iterations(
        train_accuracies=history["train_accuracy"],
        val_accuracies=history["val_accuracy"],
        save_path=plots_output_dir / "model2" / "accuracy_vs_iterations_model2.png",
    )

    print("Training finished successfully.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best model path: {best_model_path}")
    print(f"Plots saved in: {plots_output_dir}")


if __name__ == "__main__":
    main()
