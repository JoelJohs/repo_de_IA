# Logica del juego

Este documento explica el funcionamiento del juego por componentes y el flujo del loop.

## Componentes principales

- `src/juego/juego.py`: orquestador. Coordina input, estado, IA y render.
- `src/juego/estado.py`: estructuras de estado (jugador, bala, puntaje, render, modelo).
- `src/juego/fisica.py`: reglas de salto y agacharse.
- `src/juego/bala.py`: disparo, movimiento y reinicio de la bala.
- `src/juego/puntaje.py`: bonus por esquiva.
- `src/juego/loop.py`: lectura de eventos y callbacks.

## Flujo general del loop

1. Se muestra el menu.
2. Se procesa input (manual o auto).
3. Se actualizan estados (salto, bala, colision, puntaje).
4. Se dibuja la escena.

## Estado del juego (resumen)

- Jugador: posicion, salto, en_suelo, salto_vel, agachado.
- Bala: posicion, velocidad, disparada, arriba, distancia minima.
- Puntaje: valor total, incremento por frame, bonus por esquiva.
- Render: posiciones de scroll del fondo.
- Modelo: dataset, modelo entrenado, ultima probabilidad.

## Controles

- `M`: modo manual (reinicia dataset y modelo).
- `A`: modo automatico (usa MLP entrenado).
- `T`: entrena MLP.
- `C`: exporta CSV.
- `F`: fullscreen.
- `Q`: salir.
- `ESPACIO`: saltar.
- `ABAJO` o `S`: agacharse.
- `ESC` o `P`: volver al menu.

## Decisiones clave para explicar

- El loop no decide IA; solo orquesta.
- Las reglas de salto y agacharse estan separadas en `fisica.py`.
- La bala y el puntaje se actualizan con funciones puras.
- El render no toca logica de juego, solo dibuja.

## Preguntas tipo examen (resumen)

- Que responsabilidades tiene `Juego` y que no deberia hacer?
- Donde se actualiza el estado del jugador y por que esta separado?
- Que condiciones permiten iniciar un salto?
- Como se calcula el bonus por esquiva y cuando se aplica?
- Por que el loop necesita separar input, update y render?
