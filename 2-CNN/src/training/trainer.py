from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Tuple

import tensorflow as tf


@dataclass(frozen=True)
class TrainingResult:
    history: tf.keras.callbacks.History
    metrics: Dict[str, float]
    model_path: str


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    optimizer: str,
    logger: logging.Logger,
) -> None:
    optimizer_name = optimizer.lower()
    if optimizer_name == "sgd":
        opt = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            decay=learning_rate / 100,
        )
    elif optimizer_name == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    logger.info("Compiling model")
    logger.info("Optimizer: %s", optimizer_name)
    logger.info("Learning rate: %s", learning_rate)

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
    logger: logging.Logger,
) -> Tuple[tf.keras.callbacks.History, str]:
    callbacks = []
    if early_stopping:
        logger.info("Early stopping enabled (patience=%s)", patience)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    logger.info("Model checkpoint: %s", checkpoint_path)
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        )
    )

    logger.info("Starting training")
    logger.info("Epochs: %s", epochs)
    logger.info("Train batches: %s", tf.data.experimental.cardinality(train_ds).numpy())
    logger.info("Validation batches: %s", tf.data.experimental.cardinality(val_ds).numpy())
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training complete")
    return history, checkpoint_path


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    logger: logging.Logger,
) -> Dict[str, float]:
    logger.info("Evaluating model")
    results = model.evaluate(test_ds, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    logger.info("Evaluation metrics: %s", metrics)
    return metrics
