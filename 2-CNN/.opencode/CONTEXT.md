# Contexto actual - Proyecto 2-CNN

## Estado general

Se creo un esqueleto de proyecto en `2-CNN/` basado en los notebooks del profesor, sin modificarlos. El objetivo es clasificar 5 clases de animales con TensorFlow en CPU. La GUI con PyQt6 aun no esta implementada.

## Estructura actual

- `config/default.yaml`: parametros de datos, entrenamiento y salida.
- `src/`: modulos de datos, modelo, entrenamiento e inferencia.
- `train.py`: entrenamiento con CNN base.
- `predict.py`: inferencia sobre una imagen.
- `requirements.txt`: dependencias.
- `README.md`: documentacion del proyecto.
- `CNN.ipynb`, `CNNriesgo.ipynb`: referencias del profesor (no tocar).

## Flujo de datos

- Dataset por carpetas: `dataset/ranas`, `dataset/aranas`, `dataset/pajaros`, `dataset/ballenas`, `dataset/changos`.
- Carga con `image_dataset_from_directory` y split train/val/test.
- Imagenes redimensionadas a `(21, 28)` y normalizadas a [0,1].

## Modelo base

- CNN simple: `Conv2D -> LeakyReLU -> MaxPool -> Dropout -> Dense -> Softmax`.
- Parametros iniciales: `learning_rate=0.001`, `epochs=20`, `batch_size=64`.

## Pendientes

- Implementar GUI en PyQt6 para entrenamiento e inferencia.
- Agregar evaluacion avanzada (F1, matriz de confusion).
- Opcion de transfer learning si la precision es baja.
