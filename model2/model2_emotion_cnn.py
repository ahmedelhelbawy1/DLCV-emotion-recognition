"""Model 2 CNN used for emotion classification."""

import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    """Simple CNN for the 4 emotion classes in the assignment."""

    def __init__(
        self,
        num_classes: int = 4,
        input_size: int = 512,
        hidden_features: int = 128,
    ) -> None:
        super().__init__()

        # This is the exact convolution setup requested in the assignment.
        # Only the first 3 conv layers are followed by max pooling.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7),
            nn.ReLU(),
        )

        # Instead of manually calculating the flatten size, I pass a dummy
        # image through the conv part once and use the resulting shape.
        flattened_features = self._get_flattened_size(input_size)

        # After extracting features, the model uses one hidden layer
        # with Sigmoid, then the final output layer gives 4 logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, hidden_features),
            nn.Sigmoid(),
            nn.Linear(hidden_features, num_classes),
        )

    def _get_flattened_size(self, input_size: int) -> int:
        """Find the flatten size automatically from a dummy input."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            dummy_output = self.features(dummy_input)
            flattened_features = dummy_output.view(1, -1).shape[1]
        return flattened_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x
