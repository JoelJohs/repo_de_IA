# Machine Learning y dataset

Esta seccion se ajusta a los cambios requeridos: agacharse y balas con alturas variables. El objetivo es mantener el dataset actualizado para que la IA pueda decidir **saltar o agacharse**.

## Dataset en memoria

Archivo: `src/ml/dataset.py`

La funcion `registrar_decision_manual()` agrega muestras a `datos_modelo` cuando la bala ya fue disparada.

### Variables actuales

- `velocidad_bala`: velocidad del proyectil.
- `distancia`: distancia absoluta entre jugador y bala.
- `salto`: etiqueta (0 si esta en suelo, 1 si esta saltando).

### Variables nuevas requeridas

Para soportar agacharse y balas arriba/abajo, se recomienda agregar:

- `agachado`: 0/1.
- `bala_y` o `altura_bala`: posicion vertical de la bala.
- `tipo_bala`: opcional, valores `arriba`, `medio`, `abajo`.

Estas variables permiten que el modelo distinga entre saltar o agacharse segun la altura de la bala.

## Exportacion de CSV

La funcion `exportar_datos_csv()` debe actualizarse para incluir las nuevas columnas. Esto es obligatorio para poder analizar y entrenar con los datos correctos.

Formato recomendado de columnas:

- velocidad_bala
- distancia
- salto
- agachado
- bala_y

Si se usa `tipo_bala`, agregar esa columna al final.

## Entrenamiento

Archivo: `src/ml/modelo.py`

Funcion `entrenar_modelo(datos_modelo)`:

- Requiere al menos 80 muestras.
- Separa train/test con `train_test_split` estratificado.
- Escala los datos con `StandardScaler`.
- Entrena un `MLPClassifier` con arquitectura simple.
- Devuelve accuracy en test.

Si el dataset solo tiene una clase (solo salto o solo no salto), se devuelve un modelo trivial.

## Decision automatica

Funcion `decision_auto_saltar(...)`:

- Debe evolucionar a una decision doble: saltar o agacharse.
- Se recomienda cambiar a `decision_auto_accion()` que devuelva una accion enum:
  - `"saltar"`
  - `"agacharse"`
  - `"nada"`

## Visualizacion (opcional)

Archivo: `src/ml/visualizacion.py`

- Se puede extender para graficar `bala_y` o `agachado`.
- No es obligatorio para la entrega, pero ayuda a validar el dataset.
