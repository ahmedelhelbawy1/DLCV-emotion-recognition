"""Run final Model 2 training and testing.

This is the clean end-to-end runner for the final Model 2 workflow:
1. train and validate Model 2
2. load the best saved checkpoint
3. evaluate it on the test set

By default it uses the official assignment settings from train.py:
- image size 512
- batch size 8
- epochs 10
- full train / val / test folders

On a CPU-only machine this can take a long time. For quick local checks,
use --quick-test.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "model2"


def parse_args() -> argparse.Namespace:
    """Read final pipeline settings."""
    parser = argparse.ArgumentParser(description="Run Model 2 train + test pipeline.")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a small CPU-friendly proof run instead of the official full run.",
    )
    return parser.parse_args()


def run_command(command: list[str], log_path: Path) -> None:
    """Run one command and save its terminal output."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning: {' '.join(command)}")
    print(f"Log file: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}. Check {log_path}"
        )


def main() -> None:
    """Run the final Model 2 workflow."""
    args = parse_args()

    if args.quick_test:
        train_command = [
            sys.executable,
            "-u",
            "model2/train.py",
            "--image-size",
            "128",
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--max-train-samples",
            "400",
            "--max-val-samples",
            "120",
        ]
        evaluate_command = [
            sys.executable,
            "-u",
            "model2/evaluate.py",
            "--batch-size",
            "8",
            "--max-test-samples",
            "120",
        ]
        train_log = LOG_DIR / "quick_train_model2.log"
        eval_log = LOG_DIR / "quick_evaluate_model2.log"
    else:
        train_command = [sys.executable, "-u", "model2/train.py"]
        evaluate_command = [sys.executable, "-u", "model2/evaluate.py"]
        train_log = LOG_DIR / "official_train_model2.log"
        eval_log = LOG_DIR / "official_evaluate_model2.log"

    run_command(train_command, train_log)
    run_command(evaluate_command, eval_log)

    print("\nModel 2 pipeline finished successfully.")
    print(f"Training log: {train_log}")
    print(f"Evaluation log: {eval_log}")


if __name__ == "__main__":
    main()
