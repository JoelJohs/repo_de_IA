import os
from typing import Optional

import pygame

from core.constantes import AMARILLO, BLANCO, GRIS, NEGRO

_TITLE_CACHE: Optional[pygame.Surface] = None


def _cargar_titulo() -> Optional[pygame.Surface]:
    global _TITLE_CACHE
    if _TITLE_CACHE is not None:
        return _TITLE_CACHE
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ruta = os.path.join(base, "assets", "game", "title.png")
    try:
        _TITLE_CACHE = pygame.image.load(ruta).convert_alpha()
    except Exception:
        _TITLE_CACHE = None
    return _TITLE_CACHE


def _dibujar_fondo(
    pantalla: pygame.Surface, w: int, h: int, titulo_img: Optional[pygame.Surface]
) -> None:
    if titulo_img is None:
        pantalla.fill(NEGRO)
        return
    tw, th = titulo_img.get_size()
    s = max(w / tw, h / th)
    sw, sh = int(tw * s), int(th * s)
    fondo = pygame.transform.smoothscale(titulo_img, (sw, sh))
    x = (w - sw) // 2
    y = (h - sh) // 2
    pantalla.blit(fondo, (x, y))

    vignette = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(vignette, (0, 0, 0, 120), vignette.get_rect(), border_radius=0)
    pygame.draw.rect(vignette, (0, 0, 0, 80), vignette.get_rect(), 40)
    pantalla.blit(vignette, (0, 0))


def _dibujar_panel(
    pantalla: pygame.Surface, rect: pygame.Rect, color=(10, 8, 10, 180)
) -> None:
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    panel.fill(color)
    pygame.draw.rect(panel, (180, 120, 70, 120), panel.get_rect(), 2)
    pantalla.blit(panel, rect.topleft)


def dibujar_menu(
    pantalla: pygame.Surface,
    fuente: pygame.font.Font,
    fuente_chica: pygame.font.Font,
    w: int,
    h: int,
    scale: float,
    datos_modelo_len: int,
    modelo_entrenado: bool,
    decision_window: int,
    msg: str = "",
) -> None:
    titulo_img = _cargar_titulo()
    _dibujar_fondo(pantalla, w, h, titulo_img)

    opciones = [
        ("M", "Manual (reinicia dataset y borra modelo)"),
        ("A", "Auto (usa MLP; sin modelo NO salta)"),
        ("T", "Entrenar MLP"),
        ("C", "Exportar datos a CSV"),
        ("F", "Fullscreen (toggle)"),
        ("Q", "Salir"),
    ]
    line_h = fuente.get_linesize()
    pad = max(8, int(8 * scale))
    estado_lines = 2
    msg_lines = 1 if msg else 0
    content_h = len(opciones) * (line_h + pad) - pad
    content_h += int(8 * scale)
    content_h += estado_lines * fuente_chica.get_linesize()
    if msg_lines:
        content_h += int(10 * scale) + fuente_chica.get_linesize()

    menu_pad_x = int(80 * scale)
    menu_pad_y = int(14 * scale)
    menu_w = w - menu_pad_x * 2
    menu_h = content_h + menu_pad_y * 2
    menu_bottom = h - int(18 * scale)
    menu_top = max(int(h * 0.48), menu_bottom - menu_h)
    menu_rect = pygame.Rect(menu_pad_x, menu_top, menu_w, menu_h)
    _dibujar_panel(pantalla, menu_rect)

    x0 = menu_rect.x + int(28 * scale)
    y = menu_rect.y + menu_pad_y
    accent = (210, 160, 90)
    for tecla, texto in opciones:
        key = fuente.render(tecla, True, accent)
        label = fuente.render(f"- {texto}", True, BLANCO)
        pantalla.blit(key, (x0, y))
        pantalla.blit(label, (x0 + int(32 * scale), y))
        y += line_h + pad

    y += int(6 * scale)
    estado = [
        f"Memoria: {datos_modelo_len} | Modelo: {'sí' if modelo_entrenado else 'no'}",
        f"Resolución: {w}x{h} | scale≈{scale:.2f} | ventana_decisión≈{decision_window}",
    ]
    for line in estado:
        t = fuente_chica.render(line, True, GRIS)
        pantalla.blit(t, (x0, y))
        y += fuente_chica.get_linesize()

    if msg:
        mm = fuente_chica.render(msg, True, AMARILLO)
        pantalla.blit(mm, (x0, y + int(10 * scale)))

    pygame.display.flip()
