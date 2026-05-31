"""Preprocess dataset for RNN training.

Fill in dataset loading, tokenization, and train/val split.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess C code dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/data.txt"),
        help="Path to dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Directory to store processed artifacts.",
    )
    parser.add_argument(
        "--level",
        choices=["char", "token"],
        default="char",
        help="Tokenization level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Load dataset, clean, tokenize, save vocab and sequences.
    print("Preprocess placeholder. Update logic in src/preprocess.py")


if __name__ == "__main__":
    main()
