from __future__ import annotations


def calcular_bonus_esquiva(dist_min: int | None, umbral: int, base: int) -> int:
    if dist_min is None:
        return 0
    umbral = max(1, umbral)
    bonus = max(0, umbral - int(dist_min))
    return base + bonus
