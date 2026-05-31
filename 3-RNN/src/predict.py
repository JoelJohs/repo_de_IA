"""Run local predictions for autocomplete testing."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next token/char.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/rnn_v1"),
        help="Path to trained model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to autocomplete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: Load model and tokenizer/vocab, generate predictions.
    print(f"Predict placeholder. Prompt: {args.prompt}")


if __name__ == "__main__":
    main()
