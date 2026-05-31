# Machine Learning y dataset

Esta seccion explica como se registran datos y como se entrena el MLP para decidir acciones.

## Dataset en memoria

Archivo: `src/ml/dataset.py`

La funcion `registrar_decision_manual()` agrega muestras a `datos_modelo` cuando la bala ya fue disparada.

Cambio clave: la etiqueta `accion` se toma del input manual (intencion del jugador), no del estado fisico.
Esto permite entrenar el estilo real de juego (saltos constantes, agacharse continuo o no hacer nada),
independiente de si el jugador juega bien o mal.

Variables registradas:

- `velocidad_bala`: velocidad del proyectil.
- `distancia`: distancia absoluta entre jugador y bala.
- `salto`: etiqueta (0 si esta en suelo, 1 si esta saltando).
- `bala_y`: altura de la bala.
- `bala_arriba`: 1 si viene arriba, 0 si viene abajo.
- `agachado`: 1 si el jugador se agacha, 0 si no.
- `accion`: 0 no hace nada, 1 salta (input), 2 agacha (hold).
- `puntaje`: puntaje acumulado.
- `ataque_color`: alias de `bala_arriba` (para mostrar el color de la bala).

## Exportacion de CSV

La funcion `exportar_datos_csv()` guarda los datos en `datos_mlp.csv` con esas columnas.

## Entrenamiento

Archivos: `src/ml/modelo.py` (MLP) y `src/ml/arbol.py` (Arbol de Decision)

`entrenar_modelo(datos_modelo)`:

- Si hay menos de 80 muestras, devuelve un modelo trivial que siempre predice 0 (no hacer nada).
- Separa train/test con `train_test_split` estratificado.
- Escala los datos con `StandardScaler`.
- Entrena un `MLPClassifier`.
- Devuelve accuracy en test.

Si el dataset solo tiene una clase, se devuelve un modelo trivial que siempre predice esa accion.

`entrenar_arbol(datos_modelo)`:

- Mismo criterio de datos y clases que el MLP.
- Entrena un `DecisionTreeClassifier` con profundidad moderada.
- Devuelve accuracy en test.

## Decision automatica

Archivos: `src/ml/modelo.py` (MLP) y `src/ml/arbol.py` (Arbol)

`decision_auto_saltar(...)` y `decision_auto_arbol(...)`:

- Si no hay bala o el jugador no esta en suelo, devuelve accion 0.
- Si hay modelo, calcula probabilidades (cuando aplica) y predice accion.

## Sesgo de estilo (imitar al jugador)

Archivo: `src/juego/juego.py`

La IA aplica un "sesgo de estilo" en modo automatico:

- Se guarda un historial reciente de acciones manuales (`_estilo_hist`).
- Si una accion domina con un ratio >= 0.55 y hay suficientes muestras, la IA la prioriza.
- Esto garantiza que si el jugador juega saltando todo el tiempo o agachado todo el tiempo,
  la IA imita ese estilo, aunque no sea el mas optimo.

## Limpieza de "cache" al entrenar

Cuando se entrena, se limpia solo el estado temporal de la IA (cache), no el dataset:

- Historial de estilo (`_estilo_hist`).
- Ultima probabilidad y accion mostrada en pantalla.
- Flags de hold manual/auto (para no heredar estado anterior).

Esto evita que el comportamiento auto herede residuos de sesiones anteriores, pero conserva
los datos para seguir entrenando con mas muestras.

## Visualizacion (opcional)

Archivo: `src/ml/visualizacion.py`

- Grafica relaciones de variables para inspeccionar el dataset.
