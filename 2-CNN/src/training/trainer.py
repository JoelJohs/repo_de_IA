from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import tensorflow as tf


@dataclass(frozen=True)
class TrainingResult:
    history: tf.keras.callbacks.History
    metrics: Dict[str, float]
    model_path: str


def compile_model(model: tf.keras.Model, learning_rate: float, optimizer: str) -> None:
    optimizer_name = optimizer.lower()
    if optimizer_name == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"],
    )


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    early_stopping: bool,
    patience: int,
    checkpoint_path: str,
) -> Tuple[tf.keras.callbacks.History, str]:
    callbacks = []
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        )
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    return history, checkpoint_path


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> Dict[str, float]:
    results = model.evaluate(test_ds, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    return metrics
