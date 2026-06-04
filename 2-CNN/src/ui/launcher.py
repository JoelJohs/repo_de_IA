from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QMessageBox

from src.config import load_config
from src.utils import ensure_dir, load_class_names, setup_logging

from .main_window import MainWindow
from .styles import apply as apply_styles


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UI PyQt6 para CNN de animales")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Ruta al archivo de configuracion",
    )
    parser.add_argument(
        "--model",
        default="config/models/best_model.keras",
        help="Ruta al modelo .keras",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Sobreescribe data.dataset_dir",
    )
    parser.add_argument(
        "--test-dir",
        default="test",
        help="Carpeta con imagenes pequenas para explorar",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Carga todo y sale sin abrir ventana (smoke test)",
    )
    return parser.parse_args(argv)


def _resolve(base: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


def _load_metrics(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    out: dict = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            try:
                out[key] = float(value)
            except ValueError:
                continue
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    base_dir = os.path.dirname(os.path.abspath(args.config))
    cfg = load_config(args.config)

    logs_dir = ensure_dir(_resolve(base_dir, cfg.output.logs_dir))
    logger = setup_logging(logs_dir, "cnn-ui")

    logger.info("Starting CNN UI")
    logger.info("Config: %s", args.config)
    logger.info("Model:  %s", args.model)

    if not os.path.isfile(args.model):
        msg = (
            f"No se encontro el modelo en:\n{args.model}\n\n"
            "Entrena primero con:\n"
            "  ./scripts/trains/train_cnn.fish"
        )
        logger.error(msg)
        app = QApplication.instance() or QApplication(sys.argv)
        apply_styles(app)
        QMessageBox.critical(None, "Modelo no encontrado", msg)
        return 1

    class_path = _resolve(base_dir, os.path.join(cfg.output.artifacts_dir, "classes.json"))
    if not os.path.isfile(class_path):
        logger.error("Classes file missing: %s", class_path)
        app = QApplication.instance() or QApplication(sys.argv)
        apply_styles(app)
        QMessageBox.critical(
            None,
            "classes.json no encontrado",
            f"Falta {class_path}. Entrena el modelo primero.",
        )
        return 1
    class_names = load_class_names(class_path)
    logger.info("Classes: %s", class_names)

    image_size: Tuple[int, int] = tuple(cfg.data.image_size)
    dataset_dir = args.dataset_dir or _resolve(base_dir, cfg.data.dataset_dir)
    if os.path.isabs(args.test_dir):
        test_dir = args.test_dir
    else:
        test_dir = _resolve(base_dir, args.test_dir)
        if not os.path.isdir(test_dir):
            project_root = os.path.dirname(base_dir)
            candidate = _resolve(project_root, args.test_dir)
            if os.path.isdir(candidate):
                test_dir = candidate

    logger.info("Dataset dir: %s", dataset_dir)
    logger.info("Test dir:    %s", test_dir)
    logger.info("Image size:  %s", image_size)

    logger.info("Loading model")
    model = tf.keras.models.load_model(args.model)
    logger.info("Model loaded")

    metrics_path = _resolve(base_dir, os.path.join(cfg.output.artifacts_dir, "metrics.json"))
    metrics = _load_metrics(metrics_path)

    if args.no_gui:
        logger.info("--no-gui specified, exiting after model load")
        return 0

    app = QApplication.instance() or QApplication(sys.argv)
    apply_styles(app)

    window = MainWindow(
        model=model,
        class_names=class_names,
        dataset_dir=dataset_dir,
        test_dir=test_dir,
        image_size=image_size,
        model_path=args.model,
        metrics=metrics,
        logger=logger,
    )
    window.show()
    rc = app.exec()
    logger.info("UI closed with code %s", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
