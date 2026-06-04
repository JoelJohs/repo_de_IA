from __future__ import annotations

import pygame


def procesar_eventos(
    eventos,
    modo_auto: bool,
    en_suelo: bool,
    on_quit,
    on_menu,
    on_fullscreen,
    on_resize,
    on_jump,
    on_crouch_press,
):
    for e in eventos:
        if e.type == pygame.QUIT:
            on_quit()
        elif e.type == pygame.VIDEORESIZE:
            on_resize(e.w, e.h)
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q:
                on_quit()
            elif e.key in (pygame.K_ESCAPE, pygame.K_p):
                on_menu()
            elif e.key == pygame.K_f:
                on_fullscreen()
            elif e.key == pygame.K_SPACE and (not modo_auto) and en_suelo:
                on_jump()
            elif e.key in (pygame.K_DOWN, pygame.K_s):
                if not modo_auto:
                    on_crouch_press()
