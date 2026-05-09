# Guia tipo examen

Preguntas y respuestas cortas para repasar el funcionamiento del codigo.

## Arquitectura general

1) Cual es el flujo principal del juego?
Respuesta: Input -> IA -> Update de estados -> Render, dentro del loop de `Juego`.

2) Por que separar estado y reglas?
Respuesta: Para explicar cada parte por separado y evitar mezclar logica con render.

3) Que responsabilidades tiene `Juego`?
Respuesta: Orquestar componentes, no decidir reglas internas ni renderizar detalles.

4) Que archivo es el punto de entrada?
Respuesta: `src/main.py`.

## Estado del jugador

5) Cuando se puede iniciar un salto?
Respuesta: Solo si `en_suelo` es True y no esta agachado.

6) Que pasa cuando el salto termina?
Respuesta: El jugador vuelve a `ground_y`, `salto` pasa a False y `en_suelo` a True.

7) Como se implementa el agacharse?
Respuesta: Se reduce la altura del rect y se bloquea el salto mientras esta activo.

## Bala y puntaje

8) Cuando se dispara una bala?
Respuesta: Cuando no hay bala activa (`disparada=False`) el loop la genera.

9) Como se decide la altura de la bala?
Respuesta: Aleatoria, `arriba` o `abajo` en `bala.py`.

10) Como se calcula el bonus por esquiva?
Respuesta: En `puntaje.py`, depende de la distancia minima a la bala.

## IA y dataset

11) Que datos guarda el dataset?
Respuesta: velocidad, distancia, salto, bala_y, bala_arriba, agachado, accion, puntaje.

12) Cuantas muestras se requieren para entrenar?
Respuesta: Al menos 80.

13) Que pasa si el dataset solo tiene una clase?
Respuesta: Se devuelve un modelo trivial que siempre hace la misma accion.

14) Que hace el scaler?
Respuesta: Normaliza las variables antes de entrenar el MLP.

15) Que representa `proba_salto`?
Respuesta: La probabilidad estimada de elegir la accion salto.

## Render y UI

16) Donde se cargan los assets?
Respuesta: `src/render/activos.py`.

17) Que hace `dibujar_info_modelo`?
Respuesta: Muestra la probabilidad de salto en pantalla.

18) Que muestra el menu?
Respuesta: Opciones de modo, estado del modelo y mensajes.

## Explicacion corta (2-3 frases)

19) Explica el proyecto en dos frases.
Respuesta: Es un juego 2D en Pygame que registra decisiones del jugador para entrenar un MLP. Luego el modelo decide automaticamente cuando saltar o agacharse, y todo el codigo esta separado por componentes.

20) Explica por que se usa un loop con etapas.
Respuesta: Separar input, update y render facilita el mantenimiento y la explicacion del flujo.
