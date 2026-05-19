from __future__ import annotations

import json
import os
import random
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
