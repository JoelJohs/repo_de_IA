from typing import Optional

import pygame

from core.constantes import AMARILLO


def dibujar_fondo(
    pantalla: pygame.Surface, fondo_img: pygame.Surface, fondo_x1: int, fondo_x2: int
) -> None:
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))


def dibujar_personaje(
    pantalla: pygame.Surface, frames, frame_actual: int, x: int, y: int
) -> None:
    pantalla.blit(frames[frame_actual], (x, y))


def dibujar_nave(
    pantalla: pygame.Surface, nave_img: pygame.Surface, x: int, y: int
) -> None:
    pantalla.blit(nave_img, (x, y))


def dibujar_bala(
    pantalla: pygame.Surface, bala_img: pygame.Surface, x: int, y: int
) -> None:
    pantalla.blit(bala_img, (x, y))


def dibujar_info_modelo(
    pantalla: pygame.Surface,
    fuente_chica: pygame.font.Font,
    proba_salto: Optional[float],
) -> None:
    if proba_salto is None:
        return
    txt = fuente_chica.render(f"proba_salto≈{proba_salto:.2f}", True, AMARILLO)
    pantalla.blit(txt, (10, 10))
