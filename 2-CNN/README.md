# Proyecto 2 - CNN para clasificar animales

Este proyecto toma como referencia el notebook entregado por el profesor y lo convierte en una base organizada para un flujo de trabajo moderno (dataset configurable, entrenamiento reproducible, guardado de modelo e inferencia). Los notebooks **no se modifican** y se mantienen como material referencial.

Referencia original (no tocar):
- `2-CNN/CNN.ipynb`

## Objetivo

Clasificar imágenes de 5 clases:
- ranas
- aranas
- pajaros
- ballenas
- monos

La primera etapa es **clasificacion** (por carpetas). Deteccion con bounding boxes no aplica porque no hay etiquetas de cajas.

## Estructura de dataset (clasificacion por carpeta)

Se espera una carpeta con subcarpetas por clase:

```
dataset/
  ranas/
  aranas/
  pajaros/
  ballenas/
  monos/
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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparacion del dataset

Las imagenes **no** se versionan en el repo (estan ignoradas en
`.gitignore` por peso). Se distribuyen por separado y hay que dejarlas
en `dataset/<clase>/` antes de entrenar. El formato esperado es una
carpeta por clase con JPGs 32x32, nombrados
`<clase>__<origen>__<6-digit-index>.jpg`.

Hay dos formas de obtener el dataset:

### 1. Descargar el dataset ya procesado

Ubicarlo en `2-CNN/dataset/<clase>/` (la estructura exacta se ve en
"Estructura de dataset" mas arriba). Si viene en un zip, descomprimirlo
alli.

### 2. Regenerarlo desde las imagenes crudas

Las imagenes crudas viven en una carpeta externa (configurable en
`data.raw_source.path`, por defecto `/home/jojo/Imágenes/dataset_crudo`)
con una subcarpeta por especie. El script `build_processed_dataset` las
lee, las redimensiona a `data.image_size`, las recodifica como JPG y las
guarda en `dataset/<clase>/`.

Comando:

```
python -m src.data.build_processed_dataset --config config/default.yaml
```

Flags utiles:

- `--source PATH` sobreescribe `data.raw_source.path`.
- `--output PATH` sobreescribe `data.dataset_dir`.
- `--jpeg-quality N` sobreescribe `data.processing.jpeg_quality` (1-100).
- `--dry-run` cuenta y reporta cuantas imagenes procesaria por clase sin escribir nada.

Mapeo por defecto (`data.raw_source.class_mapping`):

| Carpeta origen | Carpeta final |
| --- | --- |
| aranas   | aranas   |
| ballenas | ballenas |
| pajaros  | pajaros  |
| ranas    | ranas    |
| simios   | monos    |

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
