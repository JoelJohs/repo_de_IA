from __future__ import annotations

import argparse
import os
from typing import Iterable
from typing import Tuple

import tensorflow as tf

from src.config import load_config
from src.inference.predict import predict_image
from src.utils import ensure_dir, load_class_names, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia con CNN")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Ruta al archivo de configuracion",
    )
    parser.add_argument("--model", required=True, help="Ruta al modelo .keras/.h5")
    parser.add_argument("--image", required=True, help="Ruta a la imagen a clasificar")
    return parser.parse_args()


def _resolve_dirs(base: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


def _iter_image_paths(path: str) -> Iterable[str]:
    if os.path.isdir(path):
        allowed_exts = {".jpg", ".jpeg", ".png"}
        entries = sorted(os.listdir(path))
        for name in entries:
            ext = os.path.splitext(name)[1].lower()
            if ext in allowed_exts:
                yield os.path.join(path, name)
        return
    yield path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))

    logs_dir = ensure_dir(_resolve_dirs(base_dir, cfg.output.logs_dir))
    logger = setup_logging(logs_dir, "predict")

    logger.info("Starting prediction")
    logger.info("Config path: %s", args.config)
    logger.info("Model path: %s", args.model)
    logger.info("Image path: %s", args.image)

    image_size: Tuple[int, int] = tuple(cfg.data.image_size)
    class_path = _resolve_dirs(base_dir, os.path.join(cfg.output.artifacts_dir, "classes.json"))
    class_names = load_class_names(class_path)
    logger.info("Classes: %s", class_names)

    model = tf.keras.models.load_model(args.model)
    image_paths = list(_iter_image_paths(args.image))
    if not image_paths:
        logger.warning("No images found at: %s", args.image)
        print("No se encontraron imagenes para predecir.")
        return

    for image_path in image_paths:
        results = predict_image(
            model=model,
            class_names=class_names,
            image_path=image_path,
            image_size=image_size,
            top_k=3,
            logger=logger,
        )
        logger.info("Top-3 predictions for %s: %s", image_path, results)

        print(f"\nPrediccion para: {image_path}")
        for label, prob in results:
            print(f"- {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
