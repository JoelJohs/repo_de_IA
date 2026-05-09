import os
from typing import List, Tuple

import pygame


def cargar_superficie_segura(
    path: str, size: Tuple[int, int], fallback_color=(200, 200, 200, 255)
) -> pygame.Surface:
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, size)
    except Exception:
        surf = pygame.Surface(size, pygame.SRCALPHA)
        surf.fill(fallback_color)
        return surf


def cargar_activos(
    base_dir: str,
    player_size: Tuple[int, int],
    bullet_size: Tuple[int, int],
    ship_size: Tuple[int, int],
    screen_size: Tuple[int, int],
) -> dict:
    jugador_frames: List[pygame.Surface] = [
        cargar_superficie_segura(
            os.path.join(base_dir, "assets/sprites/walk_frame1.png"), player_size
        ),
        cargar_superficie_segura(
            os.path.join(base_dir, "assets/sprites/walk_frame2.png"), player_size
        ),
        cargar_superficie_segura(
            os.path.join(base_dir, "assets/sprites/walk_frame3.png"), player_size
        ),
        cargar_superficie_segura(
            os.path.join(base_dir, "assets/sprites/walk_frame4.png"), player_size
        ),
    ]
    jugador_jump = cargar_superficie_segura(
        os.path.join(base_dir, "assets/sprites/jump.png"), player_size
    )
    jugador_down = cargar_superficie_segura(
        os.path.join(base_dir, "assets/sprites/down.png"), player_size
    )
    bala_img_arriba = cargar_superficie_segura(
        os.path.join(base_dir, "assets/game/red_atack.png"),
        bullet_size,
        (200, 60, 60, 255),
    )
    bala_img_abajo = cargar_superficie_segura(
        os.path.join(base_dir, "assets/game/yellow_atack.png"),
        bullet_size,
        (240, 200, 80, 255),
    )
    fondo_img = cargar_superficie_segura(
        os.path.join(base_dir, "assets/game/background.png"),
        screen_size,
        (40, 40, 40, 255),
    )
    nave_img = cargar_superficie_segura(
        os.path.join(base_dir, "assets/game/enemy.png"),
        ship_size,
        (140, 255, 200, 255),
    )

    return {
        "jugador_frames": jugador_frames,
        "jugador_jump": jugador_jump,
        "jugador_down": jugador_down,
        "bala_img_arriba": bala_img_arriba,
        "bala_img_abajo": bala_img_abajo,
        "fondo_img": fondo_img,
        "nave_img": nave_img,
    }
