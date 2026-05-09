from __future__ import annotations

import pygame


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


def iniciar_agacharse(
    player: pygame.Rect, agachado: bool, en_suelo: bool, altura_base: int
) -> bool:
    if agachado or not en_suelo:
        return agachado
    nueva_altura = max(1, int(altura_base * 0.1))
    player.height = nueva_altura
    return True


def terminar_agacharse(player: pygame.Rect, agachado: bool, altura_base: int) -> bool:
    if not agachado:
        return False
    player.height = altura_base
    return False
