from __future__ import annotations

import argparse
import os
from typing import Tuple

import tensorflow as tf

from src.config import load_config
from src.data.manual_datasets import build_manual_splits
from src.models.baseline_cnn import build_baseline_cnn
from src.training.trainer import compile_model
from src.utils import ensure_dir, save_class_names, set_seed, setup_logging


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
    base_dir = os.path.dirname(os.path.abspath(args.config))
    logs_dir = ensure_dir(_resolve_dirs(base_dir, cfg.output.logs_dir))
    logger = setup_logging(logs_dir, "train")

    try:
        logger.info("Starting training pipeline")
        logger.info("Config path: %s", args.config)
        logger.info("Project: %s", cfg.project.name)
        set_seed(cfg.project.seed)
        logger.info("Seed: %s", cfg.project.seed)
        dataset_dir = _resolve_dirs(base_dir, cfg.data.dataset_dir)
        logger.info("Dataset dir: %s", dataset_dir)

        artifacts_dir = ensure_dir(_resolve_dirs(base_dir, cfg.output.artifacts_dir))
        models_dir = ensure_dir(_resolve_dirs(base_dir, cfg.output.models_dir))
        logger.info("Artifacts dir: %s", artifacts_dir)
        logger.info("Models dir: %s", models_dir)

        image_size: Tuple[int, int] = tuple(cfg.data.image_size)

        datasets = build_manual_splits(
            dataset_dir=dataset_dir,
            image_size=image_size,
            validation_split=cfg.data.validation_split,
            test_split=cfg.data.test_split,
            seed=cfg.project.seed,
            logger=logger,
        )
        logger.info("Classes: %s", datasets.class_names)

        class_path = os.path.join(artifacts_dir, "classes.json")
        save_class_names(class_path, datasets.class_names)

        model = build_baseline_cnn(
            input_shape=(image_size[0], image_size[1], cfg.data.channels),
            num_classes=len(datasets.class_names),
        )

        summary_lines: list[str] = []

        def _capture_summary(line: str) -> None:
            summary_lines.append(line)

        model.summary(print_fn=_capture_summary)
        logger.info("Model summary:\n%s", "\n".join(summary_lines))
        compile_model(model, cfg.training.learning_rate, cfg.training.optimizer, logger)

        checkpoint_path = os.path.join(models_dir, "best_model.keras")
        logger.info("Starting training")
        logger.info("Epochs: %s", cfg.training.epochs)
        history = model.fit(
            datasets.train_X,
            datasets.train_Y,
            batch_size=cfg.data.batch_size,
            epochs=cfg.training.epochs,
            verbose=1,
            validation_data=(datasets.val_X, datasets.val_Y),
        )
        model.save(checkpoint_path)

        metrics_values = model.evaluate(datasets.test_X, datasets.test_Y, verbose=1)
        metrics = dict(zip(model.metrics_names, metrics_values))
        logger.info("Evaluation metrics: %s", metrics)

        metrics_path = os.path.join(artifacts_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            for key, value in metrics.items():
                handle.write(f"{key}: {value}\n")
        logger.info("Metrics saved: %s", metrics_path)

        logger.info("Training complete")
        logger.info("Model: %s", checkpoint_path)
        logger.info("Classes: %s", class_path)
        logger.info("Metrics: %s", metrics_path)
    except Exception:
        logger.exception("Training failed with an unhandled exception")
        raise


if __name__ == "__main__":
    main()
