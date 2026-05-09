# Logica del juego

Este documento describe los cambios necesarios para cumplir la entrega y las mejoras personales.

## Clase principal

La logica vive en `src/juego/juego.py` dentro de la clase `Juego`.

Responsabilidades:

- Inicializar Pygame y pantalla.
- Cargar assets segun el tamanio de pantalla.
- Mantener el estado del jugador, bala y nave.
- Controlar el salto, agacharse y colisiones.
- Registrar datos para el modelo y ejecutar decisiones automaticas.

## Ciclo principal

Metodo: `Juego.loop()`

Flujo resumido:

1. Muestra menu.
2. Entra en loop mientras `corriendo` sea True.
3. Procesa eventos de teclado.
4. Decide salto o agacharse (manual o automatico).
5. Actualiza salto y bala.
6. Dibuja la escena y actualiza pantalla.

## Controles (propuesto)

- `M`: modo manual, reinicia dataset y modelo.
- `A`: modo automatico (requiere modelo entrenado).
- `T`: entrena MLP con datos actuales.
- `C`: exporta CSV.
- `F`: fullscreen.
- `Q`: salir.
- `ESPACIO`: salto manual.
- `ABAJO` o `S`: agacharse.
- `ESC` o `P`: volver al menu.
- `X` o `K`: ataque frontal (mejora personal).

## Nuevo estado: agacharse (requerido)

### Que se agrega

- Variable `agachado` en `Juego`.
- Ajuste de hitbox cuando el jugador se agacha.
- Bloqueo de salto mientras este agachado.

### Comportamiento esperado

- Si `agachado` es True, la altura del rect del jugador se reduce.
- La colision con balas bajas se mantiene, pero puede evitar balas medias.
- Al soltar la tecla, se restaura la hitbox original.

### Puntos del codigo

- Entrada de teclado en `loop()`.
- Metodo nuevo: `iniciar_agacharse()` y `terminar_agacharse()`.
- Ajustes en `registrar_decision_manual()` para registrar el estado `agachado`.

## Balas desde arriba o desde abajo (requerido)

### Que se agrega

- Variantes de bala con altura distinta.
- Eleccion aleatoria de altura al disparar.

### Propuesta de implementacion

1. Definir `bala_y` con tres posibles valores:
   - Arriba (por ejemplo `ground_y - 120 * scale`).
   - Medio (altura actual).
   - Abajo (por ejemplo `ground_y + 10 * scale`).
2. En `disparar_bala()`, seleccionar una altura al azar.
3. En `reset_bala()`, volver a elegir altura.

### Colisiones

La colision sigue siendo `jugador.colliderect(bala)`, pero ahora la posicion vertical de la bala cambia.

## Dataset para IA (requerido)

Como el jugador ahora puede agacharse, el dataset debe incluir nuevos campos:

- `agachado`: 0/1.
- `bala_y` o `altura_bala`: para que el modelo sepa donde viene la bala.
- `tipo_bala`: opcional (arriba/medio/abajo).

Sin estas variables, el modelo no puede aprender a decidir entre saltar o agacharse.

## Mejora personal: ataque frontal

### Idea

Permitir que el jugador ataque hacia la derecha para golpear al enemigo.

### Elementos minimos

- Estado `atacando` con duracion corta (por ejemplo 200 ms).
- Hitbox de ataque frente al jugador.
- Enemigo recibe dano si colisiona con la hitbox.

## Mejora personal: enemigo fijo y jefes infinitos

### Idea

- Un enemigo siempre ubicado a la derecha.
- Barra de vida visible.
- Sistema infinito de jefes con vida escalada.

### Escalado propuesto

Vida base por jefe:

- Jefe 1: 1
- Jefe 2: 1.5
- Jefe 3: 2
- Jefe 4: 4
- Jefe 5: 8
- Luego: duplicar cada jefe

### Puntos a documentar en `enemigos.md`

- Definir clase `Enemigo`.
- Barra de vida y renderizado.
- Logica de cambio de jefe.
- Puntaje por derrota.
