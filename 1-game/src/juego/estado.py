from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pygame

from core.tipos import Sample


@dataclass
class PlayerState:
    rect: pygame.Rect
    salto: bool
    en_suelo: bool
    salto_vel: float
    agachado: bool


@dataclass
class BulletState:
    rect: pygame.Rect
    velocidad: int
    disparada: bool
    arriba: bool
    dist_min: Optional[int]


@dataclass
class ScoreState:
    valor: int
    por_frame: int
    esquiva_base: int
    esquiva_umbral: int


@dataclass
class AnimationState:
    current_frame: int
    frame_speed: int
    frame_count: int


@dataclass
class RenderState:
    fondo_x1: int
    fondo_x2: int


@dataclass
class ModelState:
    datos: List[Sample]
    modelo_mlp: Optional[object]
    scaler: Optional[object]
    entrenado_mlp: bool
    clase_unica_mlp: Optional[int]
    modelo_arbol: Optional[object]
    entrenado_arbol: bool
    clase_unica_arbol: Optional[int]
    ultima_proba: Optional[float]
    accion_auto: str
