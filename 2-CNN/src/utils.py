from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Iterable, List

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_class_names(path: str, class_names: Iterable[str]) -> None:
    save_json(path, {"classes": list(class_names)})


def load_class_names(path: str) -> List[str]:
    data = load_json(path)
    return data["classes"]


def setup_logging(log_dir: str, name: str) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{name}-{timestamp}.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized")
    logger.info("Log file: %s", log_path)
    return logger
