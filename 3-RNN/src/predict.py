"""Run predictions for char-level C code autocomplete.

Usage (CLI):
    python src/predict.py --prompt "int sum" --max-new 60 --temperature 0.5

The model + meta are loaded once. `complete()` is the public function used by
the API servers (F5 and F6) as well.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass
class RNNModel:
    model: tf.keras.Model
    stoi: dict[str, int]
    itos: dict[int, str]
    block_size: int
    vocab_size: int

    @classmethod
    def load(cls, model_path: Path, meta_path: Path | None = None) -> "RNNModel":
        if meta_path is None:
            meta_path = model_path.with_suffix(".meta.json")
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chars = meta["chars"]
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        model = tf.keras.models.load_model(model_path)
        return cls(
            model=model,
            stoi=stoi,
            itos=itos,
            block_size=meta["block_size"],
            vocab_size=meta["vocab_size"],
        )


def _encode(prefix: str, stoi: dict[str, int], block_size: int) -> np.ndarray:
    ids = [stoi[c] for c in prefix if c in stoi]
    if len(ids) >= block_size:
        ids = ids[-block_size:]
    else:
        # left-pad with the first vocab id (arbitrary; rarely used because
        # we only care about the last block_size positions)
        pad_id = 0
        ids = [pad_id] * (block_size - len(ids)) + ids
    return np.array(ids, dtype=np.int64).reshape(1, block_size)


def _sample(logits: np.ndarray, temperature: float) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    logits = logits / temperature
    # numerically stable softmax
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def complete(
    rnn: RNNModel,
    prefix: str,
    max_new: int = 60,
    temperature: float = 0.5,
    seed: int | None = None,
) -> str:
    """Generate `max_new` chars after the given prefix."""
    if seed is not None:
        np.random.seed(seed)

    state = _encode(prefix, rnn.stoi, rnn.block_size)
    generated: list[str] = []
    for _ in range(max_new):
        logits = rnn.model.predict(state, verbose=0)[0, -1]  # (vocab,)
        idx = _sample(logits, temperature)
        ch = rnn.itos[idx]
        generated.append(ch)
        # slide window
        state = np.concatenate([state[:, 1:], np.array([[idx]], dtype=np.int64)], axis=1)
    return "".join(generated)


def complete_full(rnn: RNNModel, prefix: str, max_new: int = 60, temperature: float = 0.5) -> str:
    """Returns the prefix + generated continuation."""
    return prefix + complete(rnn, prefix, max_new, temperature)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run char-level RNN autocomplete.")
    p.add_argument("--model", type=Path, default=Path("models/rnn_v1.keras"))
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--show-only-new",
        action="store_true",
        help="Print only the generated continuation (no prefix).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rnn = RNNModel.load(args.model, args.meta)
    text = complete(rnn, args.prompt, args.max_new, args.temperature, args.seed)
    if args.show_only_new:
        print(text)
    else:
        print(args.prompt + text)


if __name__ == "__main__":
    main()
