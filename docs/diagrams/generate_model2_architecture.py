"""Generate a simple block diagram for Model 2.

The assignment asks for a block diagram and the chosen hyperparameters.
This script creates a clean PNG diagram using matplotlib only.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "diagrams" / "model2_architecture.png"


def draw_block(ax, x, y, text, width=3.8, height=0.8):
    """Draw one architecture block."""
    rectangle = plt.Rectangle(
        (x, y),
        width,
        height,
        fill=True,
        edgecolor="black",
        facecolor="#e9f2ff",
        linewidth=1.5,
    )
    ax.add_patch(rectangle)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
    )


def draw_arrow(ax, x1, y1, x2, y2):
    """Draw an arrow from one block to the next."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5},
    )


def main() -> None:
    """Create and save the Model 2 architecture diagram."""
    blocks = [
        "Input image\n512 x 512 x 3",
        "Conv1 valid 3x3, 32 filters\nOutput: 510 x 510 x 32\nReLU + MaxPool 2x2 -> 255 x 255 x 32",
        "Conv2 valid 3x3, 64 filters\nOutput: 253 x 253 x 64\nReLU + MaxPool 2x2 -> 126 x 126 x 64",
        "Conv3 valid 3x3, 64 filters\nOutput: 124 x 124 x 64\nReLU + MaxPool 2x2 -> 62 x 62 x 64",
        "Conv4 valid 5x5, 32 filters\nOutput: 58 x 58 x 32\nReLU",
        "Conv5 valid 7x7, 16 filters\nOutput: 52 x 52 x 16\nReLU",
        "Flatten\n52 x 52 x 16 = 43,264 features",
        "Fully connected hidden layer\n43,264 -> 256\nSigmoid",
        "Output layer\n256 -> 4 logits\nSoftmax used for probabilities",
    ]

    fig, ax = plt.subplots(figsize=(9, 13))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 14)
    ax.axis("off")

    y_position = 12.2
    block_x = 0.65
    block_width = 4.7
    block_height = 1.0
    gap = 1.35

    for index, block_text in enumerate(blocks):
        draw_block(
            ax,
            block_x,
            y_position,
            block_text,
            width=block_width,
            height=block_height,
        )

        if index < len(blocks) - 1:
            draw_arrow(
                ax,
                block_x + block_width / 2,
                y_position,
                block_x + block_width / 2,
                y_position - 0.35,
            )

        y_position -= gap

    ax.text(
        3,
        13.35,
        "Model 2 CNN Architecture",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    print(f"Model 2 architecture diagram saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
