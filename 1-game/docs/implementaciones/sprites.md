# Sprites de jugador (walk/jump/down)

Fecha: 2026-05-09

## Resumen

Se reemplazaron los sprites de caminata `mono_frame_*` por `walk_frame*`, y se agregaron sprites dedicados para salto (`jump.png`) y agacharse (`down.png`). Se ajusto la logica de salto/agacharse, alturas de bala, puntaje y nuevos ataques por color.

## Cambios realizados

### Carga de assets

- Se reemplazaron los frames de caminata del jugador:
  - Antes: `assets/sprites/mono_frame_1.png` a `mono_frame_4.png`
  - Ahora: `assets/sprites/walk_frame1.png` a `walk_frame4.png`
- Se agregaron superficies nuevas:
  - `assets/sprites/jump.png` (uso durante salto)
  - `assets/sprites/down.png` (reservado para agacharse)

### Escala del jugador

- Se fijó el tamaño base del jugador para que coincida con el sprite:
  - Tamaño actual: 60x96
- El escalado por resolución se mantiene, solo cambia el tamaño base.

Archivos afectados:

- `src/juego/juego.py`

### Agacharse y balas arriba/abajo

- Se agregó el estado `agachado` y el ajuste de hitbox al agacharse.
- Se cargó el sprite `down.png` para renderizar el agachado.
- Las balas ahora salen aleatoriamente desde arriba o abajo.
- En modo manual: el jugador se agacha con `ABAJO/S` y salta con `ESPACIO` o `ENTER`.
- Se desactivó el auto-agacharse/saltar para evitar controles bloqueados.
- Al agacharse la hitbox se reduce al 10% (mínimo 1px) y no se mueve `jugador.y`.
- Alturas de bala coherentes con el sprite 60x96:
  - Arriba (rojo): `ground_y + player_size[1] - bullet_size[1]` (al ras del suelo)
  - Abajo (amarillo): `ground_y + player_size[1] * 0.35 - 20 * scale`

Archivos afectados:

- `src/juego/juego.py`

### Enemigo

- Se reemplazó el sprite del UFO por `enemy.png`.
- El tamaño del enemigo se aumentó a 3x respecto al tamaño base.
- La posicion del enemigo se ajusta para mantenerlo dentro de la pantalla.

Archivos afectados:

- `src/render/activos.py`
- `src/juego/juego.py`

### Ataques por altura

- Se agregaron sprites de ataque por altura:
  - Arriba: `assets/game/red_atack.png`
  - Abajo: `assets/game/yellow_atack.png`
- Se dibuja el ataque rojo cuando la bala viene por arriba y el amarillo cuando viene por debajo.
- Se aumento el tamaño de la bala para que los ataques sean visibles (64x64).

### Fondo

- Se actualizo el fondo para usar `assets/game/background.png`.

Archivos afectados:

- `src/render/activos.py`
- `src/juego/juego.py`

### Dataset ampliado

- Se extendió `Sample` con variables de contexto y acción:
  - `bala_y`, `bala_arriba`, `agachado`, `accion`, `puntaje`, `ataque_color`.
- El CSV exportado incluye estas nuevas columnas.
- El entrenamiento usa ahora `accion` como etiqueta multiclase.

Detalles de columnas:

- `velocidad_bala`: velocidad horizontal de la bala (negativa si va de derecha a izquierda).
- `distancia`: distancia horizontal jugador-bala en el frame de registro.
- `salto`: etiqueta binaria legacy (1 si esta en el aire, 0 si esta en suelo).
- `bala_y`: altura actual de la bala en pantalla.
- `bala_arriba`: 1 si la bala viene por arriba (roja), 0 si viene por abajo (amarilla).
- `agachado`: 1 si el jugador esta agachado en ese frame, 0 si no.
- `accion`: etiqueta multiclase actual:
  - 0: no accion
  - 1: salto
  - 2: agacharse
- `puntaje`: puntaje acumulado del jugador en ese frame (incluye bonus por cercania).
- `ataque_color`: alias de color de ataque (1=rojo/arriba, 0=amarillo/abajo).

Como se usa en el entrenamiento:

- Features X:
  - `velocidad_bala`
  - `distancia`
  - `bala_y`
  - `bala_arriba`
  - `puntaje`
  - `ataque_color`
- Etiqueta y: `accion` (multiclase).
- Entrenamiento usa `train_test_split` y `StandardScaler` antes del MLP.
- El modelo aprende a distinguir tres acciones (0/1/2) en función de la altura/color de la bala.

Como se usa en el juego (modo auto):

- Se calcula la acción con `decision_auto_saltar()`.
- Acción predicha:
  - 0: no hacer nada
  - 1: saltar (`iniciar_salto()`)
  - 2: agacharse (`iniciar_agacharse()`)
- Si la acción es 0 y el jugador estaba agachado, se llama `terminar_agacharse()`.

Persistencia / recolección:

- La recolección de datos ocurre en modo manual, frame a frame, mientras la bala está activa.
- `accion` refleja la decisión real tomada en ese frame (salto/agacharse/ninguna).
- `puntaje` se registra para que el modelo pueda correlacionar decisiones con desempeño.

Correcciones aplicadas:

- Se corrigió la exportación CSV moviendo la escritura dentro del `with`.
- Se eliminó el uso de `agachado` como feature para evitar sesgo hacia el estado actual.
- Se agregó `ataque_color` como feature explícita para aprender cuándo agacharse.

Salida en HUD:

- Se muestra `proba_salto` cuando el modelo tiene probabilidades.
- Se muestra la accion recomendada (`salta`, `agacha` o `none`).

Archivos clave para explicar al profesor:

- `src/core/tipos.py` (nuevas columnas en `Sample`).
- `src/ml/dataset.py` (registro de columnas y export CSV).
- `src/ml/modelo.py` (features X, etiqueta y, predicción multiclase).
- `src/juego/juego.py` (uso de la acción en modo auto y HUD).

Archivos afectados:

- `src/core/tipos.py`
- `src/ml/dataset.py`
- `src/ml/modelo.py`
- `src/juego/juego.py`

### Sistema de puntaje

- Se agregó puntaje por tiempo vivo (+1 por frame).
- Se agregó puntaje extra al esquivar una bala.
- El bonus aumenta cuanto más cerca pasa la bala del jugador.
- El umbral de cercanía se escala con la resolución (`200 * scale`) para mantener el comportamiento consistente.
- El puntaje se muestra en HUD y se registra en el dataset.

### Suelo y salto

- El piso se subio 100px para alinearlo con el puente del fondo.
- El salto se incremento 15% para compensar el cambio de altura.

Archivos afectados:

- `src/juego/juego.py`

### Hitbox visual (debug)

- Se agrego un toggle con tecla `H` para dibujar hitbox de jugador y ataques.
- La hitbox de los ataques se reduce a 40% del sprite para evitar colisiones injustas.

Archivos afectados:

- `src/juego/juego.py`

Archivos afectados:

- `src/juego/juego.py`
- `src/ml/dataset.py`
- `src/core/tipos.py`

### Correcciones

- Se eliminó un `pygame.quit()` duplicado que causaba `IndentationError` al ejecutar `src/main.py`.

Archivos afectados:

- `src/juego/juego.py`

### Logica de render

- Cuando `self.salto` es True, se renderiza `jugador_jump`.
- En caso contrario, se mantiene la animacion cíclica de `jugador_frames`.
- Cuando `self.agachado` es True, se renderiza `jugador_down`.

Archivos afectados:

- `src/juego/juego.py`

## Notas

- `jugador_down` se usa en modo manual al agacharse.
- Se actualizo el inventario de assets para reflejar las nuevas rutas.

Archivo de inventario:

- `docs/activos.md`
