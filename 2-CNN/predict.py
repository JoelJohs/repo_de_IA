from __future__ import annotations

import argparse
import os
from typing import Tuple

import tensorflow as tf

from src.config import load_config
from src.inference.predict import predict_image
from src.utils import load_class_names


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))

    image_size: Tuple[int, int] = tuple(cfg.data.image_size)
    class_path = _resolve_dirs(base_dir, os.path.join(cfg.output.artifacts_dir, "classes.json"))
    class_names = load_class_names(class_path)

    model = tf.keras.models.load_model(args.model)
    results = predict_image(
        model=model,
        class_names=class_names,
        image_path=args.image,
        image_size=image_size,
        top_k=3,
    )

    print("Prediccion:")
    for label, prob in results:
        print(f"- {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
