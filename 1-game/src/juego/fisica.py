from __future__ import annotations

import pygame

from core.constantes import DURACION_AGACHAR


def iniciar_salto(player: pygame.Rect, en_suelo: bool, agachado: bool) -> bool:
    if en_suelo and not agachado:
        return True
    return False


def aplicar_salto(
    player: pygame.Rect, salto: bool, salto_vel: float, gravedad: float, ground_y: int
) -> tuple[bool, float, bool]:
    if salto:
        player.y -= int(salto_vel)
        salto_vel -= gravedad
        if player.y >= ground_y:
            player.y = ground_y
            salto = False
            salto_vel = 0.0
            en_suelo = True
        else:
            en_suelo = False
        return salto, salto_vel, en_suelo
    return salto, salto_vel, True


def iniciar_agacharse(en_suelo: bool) -> tuple[bool, int]:
    if not en_suelo:
        return False, 0
    return True, DURACION_AGACHAR


def actualizar_agacharse(agachado: bool, timer: int) -> tuple[bool, int]:
    if not agachado:
        return False, 0
    timer -= 1
    if timer <= 0:
        return False, 0
    return True, timer
