# Arquitectura del proyecto

## Objetivo general

El proyecto implementa un juego simple en Pygame que permite entrenar un modelo MLP para decidir acciones (saltar o agacharse). El codigo se divide en modulos para separar: logica del juego, render, interfaz y Machine Learning.

## Estructura principal

Rutas relativas a `1-game/`:

- `src/main.py`: punto de entrada. Ajusta el `sys.path` y crea `Juego().loop()`.
- `src/juego/juego.py`: clase principal del juego. Contiene el loop, estados, colisiones y reglas.
- `src/render/activos.py`: carga y escala de imagenes (assets).
- `src/render/dibujar.py`: funciones de dibujo en pantalla.
- `src/ui/menu.py`: menu de inicio y mensajes.
- `src/ml/`: dataset, entrenamiento, decision automatica y visualizaciones.
- `src/core/`: constantes y tipos base.

## Flujo de ejecucion

1. `src/main.py` llama a `Juego().loop()`.
2. `Juego` inicializa Pygame, resolucion y assets.
3. Se muestra el menu inicial (`src/ui/menu.py`).
4. Se ejecuta el loop principal:
   - Entrada de teclado.
   - Registro de datos (modo manual) o decision automatica (modo auto).
   - Actualizacion de salto, agacharse y bala.
   - Dibujo de fondo, jugador, bala y nave.

## Estados y datos principales

- Estado de juego: `modo_auto`, `corriendo`, `salto`, `agachado`, `en_suelo`.
- Estado del modelo: `modelo_entrenado`, `modelo`, `scaler`, `clase_unica`.
- Dataset en memoria: `datos_modelo` (lista de `Sample`).

## Dependencias clave

- Pygame: loop, input y render.
- scikit-learn: `MLPClassifier`, `StandardScaler` y division train/test.
- matplotlib: visualizacion 2D/3D (opcional).

## Puntos de modificacion habituales

- Cambios de mecanica: `src/juego/juego.py`.
- Cambios de interfaz: `src/ui/menu.py`.
- Cambios de IA: `src/ml/`.
- Assets y render: `src/render/`.
