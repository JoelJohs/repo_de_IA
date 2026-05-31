# Explicaciones del codigo (para defender en clase)

## Flujo general

1) `train.py` carga configuracion y crea datasets (train/val/test) de forma manual.
2) Se construye la CNN y se entrena con `trainer.py`.
3) Se guarda el mejor modelo en `config/models/best_model.keras`.
4) Se guardan clases y metricas en `config/artifacts/`.
5) `predict.py` carga el modelo y predice imagenes (archivo o carpeta).

## Por que 32x32

- Mis imagenes reales son 32x32, asi que entrenar con 21x28 borra info.
- 32x32 es barato de entrenar y ya esta en el dataset.

## Por que 1 bloque conv

- El notebook usa 1 conv + dense y la pauta pide seguirlo.
- Es mas facil explicar y comparar con el material del curso.

## Sin data augmentation

- El notebook no lo usa.
- Se mantiene la comparacion directa con los resultados del curso.

## Optimizador y epochs

- SGD con `learning_rate=0.001` y `decay=lr/100`, igual al notebook.
- 20 epochs, igual al notebook.

## Mejora A (hyperparams)

- Se suben epochs a 40 y learning rate a 0.005.
- No se cambia la arquitectura, solo se intenta que aprenda mas.

## Script para muestrear test

- `sample_to_test.py` mueve N imagenes por clase desde `dataset/` a `test/`.
- Las renombra como `<clase>__<archivo>` para saber de donde salieron.
- Esto es solo para pruebas rapidas con `predict.fish`.

## Archivos clave

- `config/default.yaml`: parametros de entrenamiento y data.
- `src/data/datasets.py`: carga y augmentation.
- `src/models/baseline_cnn.py`: arquitectura.
- `train.py`: pipeline de entrenamiento.
- `predict.py`: inferencia.
-
## Carga manual de imagenes

- Se recorre `dataset/` con `os.walk` y se lee cada imagen, igual que en el notebook.
- Se hace resize a `image_size` si la imagen no coincide.
- Se normaliza con `/255` y se hace one-hot de las etiquetas.
- Los splits se hacen con `train_test_split`.
