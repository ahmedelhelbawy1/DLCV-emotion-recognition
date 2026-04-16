"""Create a small visual sheet of Model 2 predictions.

This figure is useful for the final report or presentation because it shows
real input samples together with the model output.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model2.model2_emotion_cnn import EmotionCNN


MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model2.pth"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "diagrams" / "model2_sample_predictions.png"


def get_transform(image_size: int) -> transforms.Compose:
    """Use the same preprocessing that was used during testing."""
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


def load_model(device: torch.device):
    """Load the trained Model 2 checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint.get("class_names", ["angry", "happy", "sad", "surprise"])
    image_size = checkpoint.get("input_size", 512)
    hidden_features = checkpoint.get("hidden_features", 256)

    model = EmotionCNN(
        num_classes=len(class_names),
        input_size=image_size,
        hidden_features=hidden_features,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, image_size


def pick_samples(dataset: datasets.ImageFolder, samples_per_class: int = 2):
    """Pick a few samples from each class so the figure is balanced."""
    selected_indices = []
    class_counts = {class_index: 0 for class_index in range(len(dataset.classes))}

    for index, (_, class_index) in enumerate(dataset.samples):
        if class_counts[class_index] < samples_per_class:
            selected_indices.append(index)
            class_counts[class_index] += 1

        if all(count == samples_per_class for count in class_counts.values()):
            break

    return selected_indices


def main() -> None:
    """Save the prediction sheet as a PNG image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_model(device)
    dataset = datasets.ImageFolder(root=TEST_DIR, transform=get_transform(image_size))
    selected_indices = pick_samples(dataset)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    with torch.no_grad():
        for ax, index in zip(axes, selected_indices):
            image_tensor, label_index = dataset[index]
            image_path, _ = dataset.samples[index]

            input_tensor = image_tensor.unsqueeze(0).to(device)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_index].item()) * 100

            original_image = Image.open(image_path).convert("RGB")
            ax.imshow(original_image)
            ax.axis("off")

            true_label = class_names[label_index]
            predicted_label = class_names[predicted_index]
            title_color = "green" if true_label == predicted_label else "red"
            ax.set_title(
                f"True: {true_label}\nPred: {predicted_label} ({confidence:.1f}%)",
                color=title_color,
                fontsize=10,
            )

    fig.suptitle("Model 2 Sample Test Predictions", fontsize=16, fontweight="bold")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    print(f"Sample prediction figure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
