"""Download selected FER2013 classes into class folders.

This script downloads the FER2013 benchmark dataset from Hugging Face
and saves only the classes needed for this assignment:
- angry
- happy
- sad
- surprise

The downloaded images are stored in class-based folders so they can be
passed directly to prepare_dataset.py for duplicate removal, resizing,
and 70/20/10 splitting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET_NAME = "Piro17/balancednumber-affecthqnet-fer2013"
TARGET_CLASSES = {"anger": "angry", "angry": "angry", "happy": "happy", "happiness": "happy", "sad": "sad", "sadness": "sad", "surprise": "surprise"}


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download selected FER2013 classes into separate folders."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Folder where the downloaded class folders will be saved.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to download.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_label(label_name: str) -> str | None:
    """Map dataset labels to the assignment class names."""
    return TARGET_CLASSES.get(label_name.lower())


def main() -> None:
    """Download FER2013 and export only the selected classes."""
    args = parse_args()
    output_dir = args.output.resolve()
    ensure_dir(output_dir)

    print("Downloading FER2013 benchmark dataset...")
    print(f"Dataset source: {args.dataset_name}")
    print(f"Saving selected classes to: {output_dir}")

    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    label_names = dataset.features["label"].names
    saved_counts = defaultdict(int)

    for class_name in ["angry", "happy", "sad", "surprise"]:
        ensure_dir(output_dir / class_name)

    print("Streaming dataset rows and saving selected classes...")

    for index, item in enumerate(dataset, start=1):
        label_name = label_names[item["label"]]
        target_class = normalize_label(label_name)

        if target_class is None:
            continue

        image = item["image"].convert("RGB")
        saved_counts[target_class] += 1

        image_path = output_dir / target_class / f"{target_class}_{saved_counts[target_class]:05d}.jpg"
        image.save(image_path, format="JPEG", quality=95)

        if index % 5000 == 0:
            print(f"Processed {index} source rows...")

    print("\nDownload and export finished.")
    for class_name in ["angry", "happy", "sad", "surprise"]:
        print(f"{class_name}: {saved_counts[class_name]} image(s)")


if __name__ == "__main__":
    main()
