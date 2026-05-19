from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset_dir: str
    image_size: Tuple[int, int]
    channels: int
    validation_split: float
    test_split: float
    batch_size: int
    shuffle: bool


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    optimizer: str
    early_stopping: bool
    patience: int


@dataclass(frozen=True)
class ModelConfig:
    type: str
    dropout: float


@dataclass(frozen=True)
class OutputConfig:
    artifacts_dir: str
    models_dir: str
    logs_dir: str


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    output: OutputConfig


def _require(value: Any, key: str) -> Any:
    if value is None:
        raise ValueError(f"Missing config key: {key}")
    return value


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    project = raw.get("project", {})
    data = raw.get("data", {})
    training = raw.get("training", {})
    model = raw.get("model", {})
    output = raw.get("output", {})

    return Config(
        project=ProjectConfig(
            name=_require(project.get("name"), "project.name"),
            seed=int(_require(project.get("seed"), "project.seed")),
        ),
        data=DataConfig(
            dataset_dir=os.path.expanduser(_require(data.get("dataset_dir"), "data.dataset_dir")),
            image_size=tuple(_require(data.get("image_size"), "data.image_size")),
            channels=int(_require(data.get("channels"), "data.channels")),
            validation_split=float(_require(data.get("validation_split"), "data.validation_split")),
            test_split=float(_require(data.get("test_split"), "data.test_split")),
            batch_size=int(_require(data.get("batch_size"), "data.batch_size")),
            shuffle=bool(_require(data.get("shuffle"), "data.shuffle")),
        ),
        training=TrainingConfig(
            epochs=int(_require(training.get("epochs"), "training.epochs")),
            learning_rate=float(_require(training.get("learning_rate"), "training.learning_rate")),
            optimizer=_require(training.get("optimizer"), "training.optimizer"),
            early_stopping=bool(_require(training.get("early_stopping"), "training.early_stopping")),
            patience=int(_require(training.get("patience"), "training.patience")),
        ),
        model=ModelConfig(
            type=_require(model.get("type"), "model.type"),
            dropout=float(_require(model.get("dropout"), "model.dropout")),
        ),
        output=OutputConfig(
            artifacts_dir=_require(output.get("artifacts_dir"), "output.artifacts_dir"),
            models_dir=_require(output.get("models_dir"), "output.models_dir"),
            logs_dir=_require(output.get("logs_dir"), "output.logs_dir"),
        ),
    )
