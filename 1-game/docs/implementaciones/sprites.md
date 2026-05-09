# Sprites de jugador (walk/jump/down)

Fecha: 2026-05-08

## Resumen

Se reemplazaron los sprites de caminata `mono_frame_*` por `walk_frame*`, y se agregaron sprites dedicados para salto (`jump.png`) y agacharse (`down.png`). El sprite de agacharse queda preparado aunque la mecanica no este implementada aun.

## Cambios realizados

### Carga de assets

- Se reemplazaron los frames de caminata del jugador:
  - Antes: `assets/sprites/mono_frame_1.png` a `mono_frame_4.png`
  - Ahora: `assets/sprites/walk_frame1.png` a `walk_frame4.png`
- Se agregaron superficies nuevas:
  - `assets/sprites/jump.png` (uso durante salto)
  - `assets/sprites/down.png` (reservado para agacharse)

### Escala del jugador

- Se aumentó el tamaño base del jugador para que el sprite se vea más grande en pantalla:
  - Antes: 32x48
  - Ahora: 38x58
- El escalado por resolución se mantiene, solo cambia el tamaño base.

Archivos afectados:

- `src/juego/juego.py`
- `juego_pygame_mlp1.py`

Archivo afectado:

- `src/render/activos.py`

### Logica de render

- Cuando `self.salto` es True, se renderiza `jugador_jump`.
- En caso contrario, se mantiene la animacion cíclica de `jugador_frames`.

Archivos afectados:

- `src/juego/juego.py`
- `juego_pygame_mlp1.py`

## Notas

- `jugador_down` se carga pero no se usa aun, porque la mecanica de agacharse no esta implementada.
- Se actualizo el inventario de assets para reflejar las nuevas rutas.

Archivo de inventario:

- `docs/activos.md`
