"""Train a vanilla RNN (SimpleRNN) for code autocomplete."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimpleRNN model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Directory with processed data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store trained models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Load processed sequences, build SimpleRNN model, train, save.
    print("Train placeholder. Update logic in src/train.py")


if __name__ == "__main__":
    main()
