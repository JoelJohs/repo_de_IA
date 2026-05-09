from dataclasses import dataclass
from typing import Optional


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    salto: int
    bala_y: float
    bala_arriba: int
    agachado: int
    accion: int
    puntaje: int
    ataque_color: int


@dataclass
class EstadoJugador:
    x: int
    y: int
    salto: bool
    en_suelo: bool
    salto_vel: float


@dataclass
class EstadoBala:
    x: int
    y: int
    velocidad: int
    disparada: bool


@dataclass
class EstadoModelo:
    modelo_entrenado: bool
    clase_unica: Optional[int]
    ultima_proba_salto: Optional[float]
