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

## Cambios requeridos por la entrega

### Agacharse

Se recomienda agregar un sprite para el jugador agachado:

- `assets/sprites/mono_crouch.png` (nombre sugerido).

Si no hay sprite dedicado, se puede reescalar la imagen del jugador para simular agacharse.

### Balas arriba/abajo

La bala puede reutilizar el mismo sprite. Solo cambia la posicion `y`.

No se requieren nuevos assets, pero si se quieren variantes visuales:

- `assets/sprites/purple_ball_up.png`
- `assets/sprites/purple_ball_down.png`

## Cambios por mejoras personales

### Ataque frontal

Se recomienda agregar un sprite de ataque:

- `assets/sprites/mono_attack.png`

### Enemigo y jefes

Assets recomendados:

- `assets/sprites/enemy.png`
- `assets/sprites/boss.png` (opcional si quieres variar).
- `assets/ui/hp_bar.png` o dibujar la barra por codigo.

## Colores de UI

Archivo: `src/core/constantes.py`

- Colores como `AMARILLO`, `BLANCO`, `GRIS` afectan textos y HUD.
- Para cambiar tematica visual, ajustar la paleta aqui.
