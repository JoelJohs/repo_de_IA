from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf


def load_image(path: str, image_size: Tuple[int, int], logger: logging.Logger) -> tf.Tensor:
    logger.info("Loading image: %s", path)
    image = tf.keras.utils.load_img(path, target_size=image_size)
    array = tf.keras.utils.img_to_array(image)
    array = array / 255.0
    return tf.expand_dims(array, axis=0)


def predict_image(
    model: tf.keras.Model,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int],
    top_k: int = 3,
    logger: logging.Logger | None = None,
) -> List[Tuple[str, float]]:
    active_logger = logger or logging.getLogger(__name__)
    input_batch = load_image(image_path, image_size, active_logger)
    probs = model.predict(input_batch, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_indices]
