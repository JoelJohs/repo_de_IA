"""Train a vanilla RNN (SimpleRNN) on char-level C code.

Architecture (from the professor's example, "Autocompletado con RNN vanilla en Keras"):

    Embedding(vocab, embed_dim)
        -> SimpleRNN(hidden, tanh, return_sequences=True)
        -> TimeDistributed(Dense(vocab))   # logits for the *next* char at every step

Loss: SparseCategoricalCrossentropy(from_logits=True)
Optimizer: Adam(lr=1e-3)

Saved artifacts in --output-dir:
    rnn_v1.keras   - the trained model
    meta.json      - copied from --processed-dir, plus training stats
    history.json   - per-epoch loss curve
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, TimeDistributed
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_model(vocab_size: int, embed_dim: int = 48, hidden: int = 64) -> Sequential:
    return Sequential(
        [
            Input(shape=(None,)),
            Embedding(vocab_size, embed_dim),
            SimpleRNN(hidden, activation="tanh", return_sequences=True),
            TimeDistributed(Dense(vocab_size)),
        ]
    )


class PrintEvery(Callback):
    """Light progress logger so we can see training life without verbose=1 spam."""

    def __init__(self, every: int = 10) -> None:
        super().__init__()
        self.every = every

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if (epoch + 1) % self.every == 0 or epoch == 0:
            logs = logs or {}
            print(
                f"  epoch {epoch + 1:4d} | loss={logs.get('loss', 0):.4f}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a char-level vanilla RNN for code autocompletion.")
    p.add_argument("--processed-dir", type=Path, default=Path("dataset/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("models"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-windows", type=int, default=20000, help="Cap on training windows for speed.")
    p.add_argument("--embed-dim", type=int, default=48)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", type=str, default="rnn_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.processed_dir / "X.npy")
    Y = np.load(args.processed_dir / "Y.npy")
    meta = json.loads((args.processed_dir / "meta.json").read_text(encoding="utf-8"))
    vocab_size = meta["vocab_size"]
    block_size = meta["block_size"]

    if len(X) > args.max_windows:
        idx = np.random.RandomState(SEED).permutation(len(X))[: args.max_windows]
        idx.sort()
        X, Y = X[idx], Y[idx]
        print(f"Subsampled to {args.max_windows:,} windows (from {len(idx):,} original)")

    print(f"X={X.shape} Y={Y.shape} | vocab={vocab_size} | block_size={block_size}")
    print(
        f"Training {args.tag}: epochs={args.epochs} batch={args.batch_size} "
        f"embed={args.embed_dim} hidden={args.hidden} lr={args.lr}"
    )

    model = build_model(vocab_size, args.embed_dim, args.hidden)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=SparseCategoricalCrossentropy(from_logits=True),
    )
    model.summary()

    history = model.fit(
        X,
        Y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[PrintEvery(every=10)],
    )

    model_path = args.output_dir / f"{args.tag}.keras"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    losses = [float(x) for x in history.history["loss"]]
    history_path = args.output_dir / f"{args.tag}.history.json"
    history_path.write_text(
        json.dumps(
            {
                "loss": losses,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "embed_dim": args.embed_dim,
                "hidden": args.hidden,
                "lr": args.lr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"History saved to {history_path}")

    # copy meta next to the model so the server finds it
    deploy_meta = dict(meta)
    deploy_meta.update(
        {
            "model_path": str(model_path.resolve()),
            "history_path": str(history_path.resolve()),
            "params": {
                "embed_dim": args.embed_dim,
                "hidden": args.hidden,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        }
    )
    (args.output_dir / f"{args.tag}.meta.json").write_text(
        json.dumps(deploy_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Deploy meta saved to {args.output_dir / (args.tag + '.meta.json')}")


if __name__ == "__main__":
    main()
