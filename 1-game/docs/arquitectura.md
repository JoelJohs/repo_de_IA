# Arquitectura del proyecto

## Objetivo general

Implementar un juego simple en Pygame que recolecta datos en modo manual y entrena un MLP para decidir acciones en modo automatico. La estructura esta separada por componentes para poder explicar el flujo con claridad.

## Capas y responsabilidades

- Entrada: `src/juego/loop.py` (eventos y callbacks).
- Estado: `src/juego/estado.py` (dataclasses).
- Reglas: `src/juego/fisica.py`, `src/juego/bala.py`, `src/juego/puntaje.py`.
- IA: `src/ml/` (dataset, entrenamiento, decision).
- Render: `src/render/` (assets y dibujo).
- UI: `src/ui/menu.py` (menu y mensajes).

## Flujo de ejecucion (macro)

1. `src/main.py` crea `Juego` y llama a `loop()`.
2. `Juego` inicializa pantalla, assets y estados.
3. Muestra menu y espera eleccion de modo.
4. Loop principal:
   - Input -> actualiza estado del jugador.
   - IA -> decide accion (si modo automatico).
   - Update -> bala, puntaje, colisiones.
   - Render -> dibuja fondo, jugador, bala, UI.

## Datos principales

- Dataset: `datos_modelo` (lista de `Sample`).
- Modelo: MLP + scaler + clase unica (si aplica).
- Estado jugador: salto, en_suelo, agachado.
- Estado bala: velocidad, disparada, altura.

## Puntos de modificacion frecuentes

- Logica de juego: `src/juego/juego.py`.
- Reglas de movimiento: `src/juego/fisica.py`.
- Cambios de IA: `src/ml/`.
- Render y assets: `src/render/`.
- UI: `src/ui/menu.py`.

## Diagrama simplificado

```
Input -> Estado -> IA -> Update -> Render
          ^                 |
          |-----------------|
```
