"""Prepare the emotion dataset for the assignment.

This script does the practical data preparation steps needed before training:
1. Read images from a source folder that already contains class subfolders
2. Keep only the 4 selected emotion classes
3. Remove exact duplicate images inside each class
4. Resize every image to 512x512 RGB
5. Split the cleaned images into train / val / test using 70 / 20 / 10

Expected source structure:
source_folder/
├── angry/
├── happy/
├── sad/
└── surprise/

Expected output structure:
dataset/
├── train/
├── val/
└── test/
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = ["angry", "happy", "sad", "surprise"]
IMAGE_SIZE = (512, 512)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
SEED = 42
DEFAULT_BENCHMARK = "FER2013"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
REPORTS_ROOT = PROJECT_ROOT / "outputs" / "reports"
SUMMARY_REPORT_PATH = REPORTS_ROOT / "dataset_preparation_summary.txt"


def parse_args() -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare the emotion dataset for train/val/test."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the original dataset folder that contains class subfolders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for shuffling before splitting.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=DEFAULT_BENCHMARK,
        help="Name of the public benchmark dataset being used.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear existing images inside dataset/train, dataset/val and dataset/test before saving new files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a folder if it is missing."""
    path.mkdir(parents=True, exist_ok=True)


def clear_split_folders() -> None:
    """Remove old prepared images so the new split is clean."""
    for split_name in ["train", "val", "test"]:
        split_dir = DATASET_ROOT / split_name
        if not split_dir.exists():
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                shutil.rmtree(class_dir)


def validate_source_folder(source_dir: Path) -> None:
    """Make sure the source folder and class folders exist."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder was not found: {source_dir}")

    missing_classes = [class_name for class_name in CLASS_NAMES if not (source_dir / class_name).exists()]
    if missing_classes:
        missing_text = ", ".join(missing_classes)
        raise FileNotFoundError(
            f"These required class folders are missing in the source dataset: {missing_text}"
        )


def list_images(class_dir: Path) -> List[Path]:
    """Collect image paths from one class folder."""
    image_paths = []
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)
    return image_paths


def get_image_hash(image_path: Path) -> str | None:
    """Create a hash from image pixels after converting to RGB.

    This helps remove exact duplicate images even if file names differ.
    If an image cannot be opened, it is skipped.
    """
    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            hash_input = rgb_image.tobytes() + str(rgb_image.size).encode("utf-8")
            return hashlib.sha256(hash_input).hexdigest()
    except (UnidentifiedImageError, OSError):
        print(f"Skipping unreadable image: {image_path}")
        return None


def remove_duplicates(image_paths: List[Path]) -> List[Path]:
    """Keep only one copy of each exact image."""
    unique_images = []
    seen_hashes = set()

    for image_path in image_paths:
        image_hash = get_image_hash(image_path)
        if image_hash is None:
            continue

        if image_hash not in seen_hashes:
            seen_hashes.add(image_hash)
            unique_images.append(image_path)

    return unique_images


def split_paths(image_paths: List[Path], seed: int) -> Dict[str, List[Path]]:
    """Split a class into train / val / test lists."""
    random_generator = random.Random(seed)
    shuffled_paths = image_paths[:]
    random_generator.shuffle(shuffled_paths)

    total_images = len(shuffled_paths)
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)
    test_count = total_images - train_count - val_count

    train_paths = shuffled_paths[:train_count]
    val_paths = shuffled_paths[train_count:train_count + val_count]
    test_paths = shuffled_paths[train_count + val_count:]

    print(
        f"Split sizes -> train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}"
    )

    if len(train_paths) + len(val_paths) + len(test_paths) != total_images:
        raise ValueError("Split counts do not add up correctly.")

    if test_count < 0:
        raise ValueError("Test count became negative. Please check the split logic.")

    return {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }


def save_resized_images(split_data: Dict[str, List[Path]], class_name: str) -> None:
    """Resize images and save them into the assignment folder structure."""
    for split_name, paths in split_data.items():
        target_dir = DATASET_ROOT / split_name / class_name
        ensure_dir(target_dir)

        for index, source_path in enumerate(paths, start=1):
            try:
                with Image.open(source_path) as image:
                    rgb_image = image.convert("RGB")
                    resized_image = rgb_image.resize(IMAGE_SIZE)

                    # New names keep the class name and running index.
                    target_name = f"{class_name}_{index:04d}.jpg"
                    target_path = target_dir / target_name
                    resized_image.save(target_path, format="JPEG", quality=95)
            except (UnidentifiedImageError, OSError):
                print(f"Skipping unreadable image during save: {source_path}")


def prepare_class(source_dir: Path, class_name: str, seed: int) -> Dict[str, int]:
    """Prepare one class from start to finish."""
    class_dir = source_dir / class_name
    original_images = list_images(class_dir)

    print(f"\nPreparing class: {class_name}")
    print(f"Found {len(original_images)} image(s) before cleaning.")

    unique_images = remove_duplicates(original_images)
    duplicates_removed = len(original_images) - len(unique_images)

    print(f"Removed {duplicates_removed} duplicate image(s).")
    print(f"{len(unique_images)} image(s) left after cleaning.")

    if len(unique_images) == 0:
        raise ValueError(f"No valid images were found for class: {class_name}")

    split_data = split_paths(unique_images, seed=seed)
    save_resized_images(split_data, class_name)

    return {
        "original_count": len(original_images),
        "duplicates_removed": duplicates_removed,
        "clean_count": len(unique_images),
        "train_count": len(split_data["train"]),
        "val_count": len(split_data["val"]),
        "test_count": len(split_data["test"]),
    }


def create_output_structure() -> None:
    """Make sure train / val / test folders exist for all classes."""
    for split_name in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            ensure_dir(DATASET_ROOT / split_name / class_name)


def write_summary_report(
    benchmark_name: str,
    source_dir: Path,
    summary_data: Dict[str, Dict[str, int]],
) -> None:
    """Save a short dataset preparation report for the assignment."""
    ensure_dir(REPORTS_ROOT)

    total_original = sum(class_info["original_count"] for class_info in summary_data.values())
    total_duplicates_removed = sum(
        class_info["duplicates_removed"] for class_info in summary_data.values()
    )
    total_clean = sum(class_info["clean_count"] for class_info in summary_data.values())
    total_train = sum(class_info["train_count"] for class_info in summary_data.values())
    total_val = sum(class_info["val_count"] for class_info in summary_data.values())
    total_test = sum(class_info["test_count"] for class_info in summary_data.values())

    report_lines = [
        "Dataset Preparation Summary",
        "===========================",
        "",
        f"Benchmark dataset: {benchmark_name}",
        f"Source folder: {source_dir}",
        f"Selected classes: {', '.join(CLASS_NAMES)}",
        f"Target size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} x 3",
        f"Split ratio: {int(TRAIN_RATIO * 100)}/{int(VAL_RATIO * 100)}/{int(TEST_RATIO * 100)}",
        "",
        "Per-class summary:",
    ]

    for class_name in CLASS_NAMES:
        class_info = summary_data[class_name]
        report_lines.extend(
            [
                f"- {class_name}",
                f"  original images: {class_info['original_count']}",
                f"  duplicates removed: {class_info['duplicates_removed']}",
                f"  images after cleaning: {class_info['clean_count']}",
                f"  train images: {class_info['train_count']}",
                f"  val images: {class_info['val_count']}",
                f"  test images: {class_info['test_count']}",
            ]
        )

    report_lines.extend(
        [
            "",
            "Overall summary:",
            f"- total original images: {total_original}",
            f"- total duplicates removed: {total_duplicates_removed}",
            f"- total cleaned images: {total_clean}",
            f"- total train images: {total_train}",
            f"- total val images: {total_val}",
            f"- total test images: {total_test}",
        ]
    )

    SUMMARY_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full dataset preparation process."""
    args = parse_args()
    source_dir = args.source.resolve()

    validate_source_folder(source_dir)

    print("Starting dataset preparation...")
    print(f"Benchmark dataset: {args.benchmark}")
    print(f"Source folder: {source_dir}")
    print(f"Target dataset folder: {DATASET_ROOT}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Target image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print("Split ratio: 70% train / 20% val / 10% test")

    if args.clear_output:
        print("Clearing old prepared images first...")
        clear_split_folders()

    create_output_structure()
    summary_data = {}

    for class_name in CLASS_NAMES:
        class_summary = prepare_class(
            source_dir=source_dir,
            class_name=class_name,
            seed=args.seed,
        )
        summary_data[class_name] = class_summary

    write_summary_report(
        benchmark_name=args.benchmark,
        source_dir=source_dir,
        summary_data=summary_data,
    )

    print("\nDataset preparation finished successfully.")
    print("Prepared images are now saved inside dataset/train, dataset/val and dataset/test.")
    print(f"Preparation summary saved to: {SUMMARY_REPORT_PATH}")


if __name__ == "__main__":
    main()
