import pygame

from core.constantes import AMARILLO, BLANCO, GRIS, NEGRO


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
    pantalla.fill(NEGRO)
    titulo = fuente.render("MENÚ", True, BLANCO)
    pantalla.blit(titulo, (w // 2 - titulo.get_width() // 2, int(60 * scale)))

    opciones = [
        "M - Manual (reinicia dataset y borra modelo)",
        "A - Auto (usa MLP; sin modelo NO salta)",
        "T - Entrenar MLP",
        "C - Exportar datos a CSV",
        "F - Fullscreen (toggle)",
        "Q - Salir",
    ]
    x0 = int(80 * scale)
    y = int(140 * scale)
    line_h = fuente.get_linesize()
    pad = max(6, int(6 * scale))
    for op in opciones:
        t = fuente.render(op, True, BLANCO)
        pantalla.blit(t, (x0, y))
        y += line_h + pad

    y += int(8 * scale)
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
        pantalla.blit(mm, (x0, y + int(12 * scale)))

    pygame.display.flip()
