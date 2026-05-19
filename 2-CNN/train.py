from __future__ import annotations

import argparse
import os
from typing import Tuple

import tensorflow as tf

from src.config import load_config
from src.data.datasets import build_datasets
from src.models.baseline_cnn import build_baseline_cnn
from src.training.trainer import compile_model, evaluate_model, train_model
from src.utils import ensure_dir, save_class_names, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar CNN para animales")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Ruta al archivo de configuracion",
    )
    return parser.parse_args()


def _resolve_dirs(base: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.project.seed)

    base_dir = os.path.dirname(os.path.abspath(args.config))
    dataset_dir = _resolve_dirs(base_dir, cfg.data.dataset_dir)

    artifacts_dir = ensure_dir(_resolve_dirs(base_dir, cfg.output.artifacts_dir))
    models_dir = ensure_dir(_resolve_dirs(base_dir, cfg.output.models_dir))

    image_size: Tuple[int, int] = tuple(cfg.data.image_size)

    datasets = build_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=cfg.data.batch_size,
        validation_split=cfg.data.validation_split,
        test_split=cfg.data.test_split,
        shuffle=cfg.data.shuffle,
        seed=cfg.project.seed,
    )

    class_path = os.path.join(artifacts_dir, "classes.json")
    save_class_names(class_path, datasets.class_names)

    model = build_baseline_cnn(
        input_shape=(image_size[0], image_size[1], cfg.data.channels),
        num_classes=len(datasets.class_names),
        dropout=cfg.model.dropout,
    )

    compile_model(model, cfg.training.learning_rate, cfg.training.optimizer)

    checkpoint_path = os.path.join(models_dir, "best_model.keras")
    train_model(
        model=model,
        train_ds=datasets.train,
        val_ds=datasets.val,
        epochs=cfg.training.epochs,
        early_stopping=cfg.training.early_stopping,
        patience=cfg.training.patience,
        checkpoint_path=checkpoint_path,
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    metrics = evaluate_model(best_model, datasets.test)

    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            handle.write(f"{key}: {value}\n")

    print("Entrenamiento completo")
    print(f"Modelo: {checkpoint_path}")
    print(f"Clases: {class_path}")
    print(f"Metricas: {metrics_path}")


if __name__ == "__main__":
    main()
