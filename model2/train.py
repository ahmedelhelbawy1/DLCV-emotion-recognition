"""Train Model 2 for emotion classification."""

from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def get_data_transforms() -> transforms.Compose:
    """Use the same preprocessing for train and validation images."""
    return transforms.Compose(
        [
            # The assignment says images should be 512x512.
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            # Standard normalization values for RGB images.
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def validate_dataset_folder(folder_path: Path, folder_name: str) -> None:
    """Stop early with a clear message if a folder is missing."""
    if not folder_path.exists():
        raise FileNotFoundError(
            f"Required dataset folder was not found: {folder_path}\n"
            f"Please make sure dataset/{folder_name} exists."
        )


def load_datasets() -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Load train and validation data from the dataset folder."""
    train_dir = PROJECT_ROOT / "dataset" / "train"
    val_dir = PROJECT_ROOT / "dataset" / "val"

    validate_dataset_folder(train_dir, "train")
    validate_dataset_folder(val_dir, "val")

    transform = get_data_transforms()

    # ImageFolder works well here because the dataset is already split
    # into class-based folders like angry/, happy/, sad/, surprise/.
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

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

    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap datasets in dataloaders so batches can be read easily."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
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
) -> Tuple[float, float]:
    """Check model performance on the validation set."""
    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0
    total_samples = 0

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

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_accuracy / total_samples

    return epoch_loss, epoch_accuracy


def save_best_model(
    model: EmotionCNN,
    class_names: List[str],
    save_path: Path,
) -> None:
    """Save the current best model for testing later."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "input_size": IMAGE_SIZE,
        "hidden_features": HIDDEN_FEATURES,
    }
    torch.save(checkpoint, save_path)


def main() -> None:
    """Run the full training process."""
    set_seed(SEED)
    device = get_device()

    print(f"Using device: {device}")
    print("Loading training and validation data...")

    train_dataset, val_dataset = load_datasets()
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)

    model = EmotionCNN(
        num_classes=NUM_CLASSES,
        input_size=IMAGE_SIZE,
        hidden_features=HIDDEN_FEATURES,
    ).to(device)

    # CrossEntropyLoss fits this task because the model predicts
    # one class out of 4 possible emotion classes.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_output_dir = PROJECT_ROOT / "outputs" / "models"
    plots_output_dir = PROJECT_ROOT / "outputs" / "plots"
    ensure_dir(model_output_dir)
    ensure_dir(plots_output_dir)

    best_model_path = model_output_dir / "best_model2.pth"
    best_val_accuracy = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    print("\nTraining started...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_best_model(
                model=model,
                class_names=train_dataset.classes,
                save_path=best_model_path,
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

    print("Training finished successfully.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best model path: {best_model_path}")
    print(f"Plots saved in: {plots_output_dir}")


if __name__ == "__main__":
    main()
