"""Preprocess C code corpus into char-level windows for the RNN.

The corpus is read as one big string. We build:
  * a vocabulary of unique characters
  * a single sequence of integer ids
  * sliding windows of BLOCK_SIZE chars

Output (in --output-dir):
  * meta.json : {"chars": [...], "block_size": int, "vocab_size": int,
                 "corpus_chars": int, "num_windows": int,
                 "dataset_path": "...absolute path of source file..."}
  * X.npy, Y.npy : int64 arrays of shape (num_windows, BLOCK_SIZE)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess C code for char-level RNN.")
    p.add_argument("--input", type=Path, default=Path("dataset/sample/funciones.c"))
    p.add_argument("--output-dir", type=Path, default=Path("dataset/processed"))
    p.add_argument("--block-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text = args.input.read_text(encoding="utf-8", errors="ignore")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    seq = np.array([stoi[c] for c in text], dtype=np.int64)
    vocab_size = len(chars)

    if len(seq) <= args.block_size:
        raise SystemExit(
            f"Corpus too short: {len(seq)} chars (need > block_size={args.block_size})."
        )

    n_windows = len(seq) - args.block_size
    X = np.empty((n_windows, args.block_size), dtype=np.int64)
    Y = np.empty((n_windows, args.block_size), dtype=np.int64)
    for i in range(n_windows):
        X[i] = seq[i : i + args.block_size]
        Y[i] = seq[i + 1 : i + 1 + args.block_size]

    np.save(args.output_dir / "X.npy", X)
    np.save(args.output_dir / "Y.npy", Y)

    meta = {
        "chars": chars,
        "block_size": args.block_size,
        "vocab_size": vocab_size,
        "corpus_chars": len(seq),
        "num_windows": n_windows,
        "dataset_path": str(args.input.resolve()),
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"vocab_size={vocab_size} | corpus_chars={len(seq):,} | num_windows={n_windows:,}")
    print(f"X.shape={X.shape} Y.shape={Y.shape} dtype={X.dtype}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
