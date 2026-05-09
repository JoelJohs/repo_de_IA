# Render y assets

## Carga de assets

Archivo: `src/render/activos.py`

Funcion principal:

- `cargar_activos(base_dir, player_size, bullet_size, ship_size, screen_size)`

Esta funcion carga y escala:

- Animacion del jugador (4 frames).
- Imagen de bala.
- Fondo.
- Nave.

Si un asset falla al cargar, se usa un placeholder coloreado para evitar que el juego se rompa.

## Dibujo en pantalla

Archivo: `src/render/dibujar.py`

Funciones:

- `dibujar_fondo`: blit del fondo en dos posiciones para efecto scroll.
- `dibujar_personaje`: dibuja frame actual del jugador.
- `dibujar_bala`: dibuja la bala.
- `dibujar_nave`: dibuja la nave.
- `dibujar_info_modelo`: texto con probabilidad de salto.

## Colores de UI

Archivo: `src/core/constantes.py`

- Colores como `AMARILLO`, `BLANCO`, `GRIS` afectan textos y HUD.
