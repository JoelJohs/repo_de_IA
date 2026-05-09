from __future__ import annotations

import random

import pygame


def disparar_bala(
    bala: pygame.Rect,
    disparada: bool,
    velocidad_base: int,
    scale: float,
    bala_arriba: bool,
    ground_y: int,
    player_height: int,
    bullet_height: int,
) -> tuple[bool, int, bool]:
    if disparada:
        return True, velocidad_base, bala_arriba
    velocidad = int(random.randint(-12, -6) * scale)
    bala_arriba = random.choice([True, False])
    if bala_arriba:
        bala.y = ground_y + player_height - int(bullet_height)
    else:
        bala.y = ground_y + int(player_height * 0.35) - int(30 * scale)
    return True, velocidad, bala_arriba


def reset_bala(bala: pygame.Rect, start_x: int) -> tuple[bool, bool, None]:
    bala.x = start_x
    return False, False, None


def mover_bala(
    bala: pygame.Rect,
    velocidad: int,
    jugador_x: int,
    dist_min: int | None,
) -> int | None:
    bala.x += velocidad
    distancia = abs(jugador_x - bala.x)
    if dist_min is None:
        return distancia
    return min(dist_min, distancia)
