# Proyecto 2 - CNN para clasificar animales

Este proyecto toma como referencia los notebooks entregados por el profesor y los convierte en una base organizada para un flujo de trabajo moderno (dataset configurable, entrenamiento reproducible, guardado de modelo e inferencia). Los notebooks **no se modifican** y se mantienen como material referencial.

Referencias originales (no tocar):
- `2-CNN/CNN.ipynb`
- `2-CNN/CNNriesgo.ipynb`

## Objetivo

Clasificar imágenes de 5 clases:
- ranas
- aranas
- pajaros
- ballenas
- changos

La primera etapa es **clasificacion** (por carpetas). Deteccion con bounding boxes no aplica porque no hay etiquetas de cajas.

## Estructura de dataset (clasificacion por carpeta)

Se espera una carpeta con subcarpetas por clase:

```
dataset/
  ranas/
  aranas/
  pajaros/
  ballenas/
  changos/
```

## Lo que aportan los notebooks

Los notebooks muestran el flujo base que se va a conservar y mejorar:

- Lectura de imagenes desde subdirectorios
- Etiquetado por carpeta
- Normalizacion y redimensionado a `(21, 28, 3)`
- Entrenamiento de un CNN sencillo con Keras
- Evaluacion y guardado del modelo
- Inferencia desde imagenes nuevas

Fragmentos clave (referencia):

- CNN base con `Conv2D`, `MaxPooling2D`, `Dropout`, `Dense` y `softmax`.
- Parametros iniciales: `INIT_LR=1e-3`, `epochs=20`, `batch_size=64`.
- Guardado de modelo en formato `.h5`.

## Estructura propuesta del proyecto

```
2-CNN/
  config/
    default.yaml
  src/
    data/
      datasets.py
    models/
      baseline_cnn.py
    training/
      trainer.py
    inference/
      predict.py
    utils.py
    config.py
  train.py
  predict.py
  requirements.txt
  README.md
```

## Uso rapido

Instalar dependencias:

```
pip install -r requirements.txt
```

Entrenar:

```
python train.py --config config/default.yaml
```

Predecir:

```
python predict.py --config config/default.yaml --model models/best_model.keras --image ruta/a/imagen.jpg
```

## Siguientes pasos (implementacion)

1. Extraer el flujo del notebook a un proyecto estructurado (sin tocar notebooks).
2. Reemplazar rutas absolutas por rutas configurables.
3. Usar `tf.data` o `image_dataset_from_directory` para lectura eficiente en CPU.
4. Implementar GUI en PyQt6 para entrenamiento e inferencia.

## Notas

- Solo CPU (sin GPU).
- Se mantendra el CNN base para cumplir el requerimiento y se podra agregar una opcion de transfer learning si se necesita mas precision.
