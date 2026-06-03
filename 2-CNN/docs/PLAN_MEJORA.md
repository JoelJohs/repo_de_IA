# Plan de Mejora - 2-CNN (clasificacion 5 animales)

## Estado actual (diagnostico)

- **Accuracy en test**: 0.5696 (loss 1.04). Azar = 20%. Target razonable = 75-85%.
- **Dataset**: 40,808 imagenes, ~8,100 por clase. Balanceado y suficiente.
- **Imagenes**: 32x32 RGB. Pequenas pero validas si se sube el modelo.
- **Arquitectura** (`src/models/baseline_cnn.py:8-31`): **muy debil**.
  - 1 sola capa `Conv2D(32)` con activacion `linear` + LeakyReLU
  - `MaxPool(2,2)` aplasta a 16x16
  - `Flatten` directo + `Dense(32)` + `Dropout(0.5)` doble
  - Sin BatchNorm, sin segunda capa conv
- **Optimizador** (`config/default.yaml:28`): `SGD lr=0.005` con decay. Lento y sin momentum.
- **Entrenamiento** (`config/default.yaml:29`): 40 epochs, sin early stopping.
- **Sin data augmentation** (documentado en `docs/decisiones.md:9` como decision consciente).
- **Sin learning rate scheduling** real.
- **Sin diagnostico de errores** (no hay confusion matrix, no hay classification report).

## Causa raiz del 56%

1. La red es **demasiado pequena** para 5 clases con imagenes naturales: un solo `Conv2D(32)` despues de un solo pooling pierde casi todo el contenido espacial antes del `Flatten`.
2. **Sin augmentation**, con 8k imagenes por clase y red chica, sobreajusta rapido y generaliza mal.
3. **SGD sin momentum** converge muy lento en 40 epochs.
4. **Sin normalizacion ni early stopping**: no hay forma de saber si el modelo sigue aprendiendo o se esta pasando de rosca.

## Plan de cambios (orden de impacto esperado)

### Fase 1 - Quick wins (1-2 horas, +15-25% accuracy esperado)

Cambios minimos en archivos existentes, sin agregar modulos nuevos.

**1.1) Arquitectura nueva** -> reemplazar `src/models/baseline_cnn.py` entero.
- 3 bloques conv: `Conv2D(32) -> BN -> ReLU -> Conv2D(32) -> BN -> ReLU -> MaxPool -> Dropout(0.25)`
- Mismo patron con 64 y 128 filtros
- `Flatten -> Dense(128) -> BN -> ReLU -> Dropout(0.5) -> Dense(num_classes, softmax)`
- Mantener la firma `build_baseline_cnn(input_shape, num_classes)` para no romper `train.py`.
- Justificacion: una sola conv 32f sobre 32x32 -> 16x16 -> flatten destruye informacion. Tres bloques conservan jerarquia de features.

**1.2) Optimizador y learning rate** -> editar `config/default.yaml`.
- `optimizer: adam`
- `learning_rate: 0.001`
- `epochs: 80`
- `early_stopping: true`
- `patience: 10`
- `dropout: 0.3` (en `model:`)
- Justificacion: Adam converge ~3x mas rapido que SGD sin momentum en este tipo de tareas. 80 epochs con early stopping permite explorar sin sobreajustar.

**1.3) Activar augmentation** -> modificar `src/data/manual_datasets.py`.
- Anadir un `tf.keras.Sequential` de augmentation aplicado **solo en train**:
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.1)`
  - `RandomZoom(0.1)`
- El split ya esta hecho: `train_X`, `val_X`, `test_X`. Aplicar `augment(train_X)` antes de devolver el bundle, o envolver con `tf.data.Dataset.map` despues.
- Decision recomendada: hacer augment **fuera del bundle** (despues de `build_manual_splits`), asi no contamina el cache de test.
- Justificacion: animales pequenos (ranas, aranas, pajaros) tienen varianza de pose y encuadre muy alta. Augment artificial regulariza y sube accuracy ~5-10%.

**1.4) Subir image size** -> editar `config/default.yaml`.
- `image_size: [64, 64]`
- Riesgo: cuadruplica el compute. Si CPU va muy lento, dejar 48x48. Probar 64x64 primero.
- Justificacion: pasar de 32x32 a 64x64 es 4x la informacion espacial. En tareas con texturas (pelaje, escamas) ayuda mucho.

**Verificacion esperada Fase 1**: accuracy 70-78%.

---

### Fase 2 - Diagnostico y tuning (1-2 horas, +5-10%)

**2.1) Agregar callbacks utiles** -> modificar `src/training/trainer.py`.
- Ademas de `EarlyStopping` y `ModelCheckpoint`, anadir:
  - `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)`
- Loguear learning rate por epoch (custom callback o atributo del history).

**2.2) Confusion matrix + classification report** -> nuevo script `src/evaluation/evaluate.py` o agregar al final de `train.py`.
- Despues de `model.evaluate(test_X, test_Y)`, calcular:
  - `sklearn.metrics.confusion_matrix(test_Y_argmax, pred_argmax)`
  - `sklearn.metrics.classification_report(...)` con `target_names=class_names`
- Guardar en `config/artifacts/confusion_matrix.png` y `config/artifacts/classification_report.txt`.
- Justificacion: saber si confunde pajaros vs monos, o ranas vs nada, indica donde apuntar Fase 3.

**2.3) Verificar leakage y duplicados** -> nuevo script `scripts/check_dataset_integrity.py` o paso en `build_processed_dataset.py`.
- Detectar duplicados por hash MD5 entre train y test.
- Verificar que cada imagen tenga 3 canales.
- Reportar imagenes corruptas o fuera de tamano.
- Justificacion: si hay 5% de duplicados, el accuracy esta artificialmente inflado o el test es "facil".

**Verificacion esperada Fase 2**: accuracy 78-85% o saber exactamente por que no sube.

---

### Fase 3 - Transfer learning (2-3 horas, +5-15% si Fase 1+2 no alcanza)

Solo si las Fases 1+2 quedan por debajo de 80%.

**3.1) Anadir modelo preentrenado** -> nuevo archivo `src/models/pretrained_cnn.py`.
- Wrapper con `tf.keras.applications.MobileNetV2(input_shape=(96,96,3), include_top=False, weights="imagenet")`.
- Parametrizable: `model.type: pretrained_mobilenet` en yaml.
- Estrategia en 2 etapas:
  1. Congelar backbone, entrenar solo cabeza (`Dense(128) -> Dropout -> Dense(num_classes)`) por 10 epochs con `lr=1e-3`.
  2. Descongelar ultimas 20 capas, fine-tune con `lr=1e-5` por 20 epochs mas.
- Subir `image_size` a 96x96 (MobileNetV2 fue entrenado en 96x96 o 224x224; 96x96 es compromiso CPU).
- Actualizar `Config` en `src/config.py` para aceptar `model.type: pretrained_mobilenet` ademas de `baseline_cnn`.
- Actualizar `train.py:11` (`from src.models.pretrained_cnn import build_pretrained_cnn`) y la logica de seleccion.

**3.2) Justificacion academica** (para defender en clase).
- Animales pequenos (32x32) son exactamente el caso donde transfer learning brilla: el backbone preentrenado en ImageNet ya sabe extraer texturas y formas.
- Es un **bonus**, no reemplaza al baseline. Mantener ambos modelos entrenados y comparar en `config/artifacts/`.

**Verificacion esperada Fase 3**: accuracy 85-92%.

---

### Fase 4 - Robustez y entrega (30 min)

**4.1) Actualizar `docs/decisiones.md` y `docs/explicaciones.md`** con:
- Por que se eligio la nueva arquitectura.
- Por que Adam > SGD en este caso.
- Por que augmentation sube accuracy.
- Si se aplico transfer learning: por que MobileNetV2, por que 2 etapas.

**4.2) Actualizar `README.md`** (seccion "Estructura propuesta") con la nueva arquitectura y el flujo actualizado.

**4.3) Verificar reproducibilidad**: el seed ya esta en `set_seed(42)`. Confirmar que con la nueva config el accuracy es estable ±2% entre 2 corridas.

**4.4) Actualizar `metrics.json` con el mejor run** y commit solo eso como evidencia final.

---

## Archivos a tocar (resumen)

| Archivo | Cambio |
|---|---|
| `config/default.yaml` | optimizer=adam, lr=1e-3, epochs=80, early_stopping=true, dropout=0.3, image_size=64x64 |
| `src/models/baseline_cnn.py` | Reescribir con 3 bloques conv + BN |
| `src/data/manual_datasets.py` | Anadir augment en train (o en pipeline posterior) |
| `src/training/trainer.py` | Anadir ReduceLROnPlateau |
| `train.py` | Calcular y guardar confusion matrix + classification report |
| `src/config.py` | Aceptar `model.type: pretrained_mobilenet` (solo si se hace Fase 3) |
| `src/models/pretrained_cnn.py` | **Nuevo**, solo si Fase 3 |
| `docs/decisiones.md` | Documentar Fase 1+2+3 |
| `docs/explicaciones.md` | Idem |
| `README.md` | Actualizar arquitectura |

## Riesgos / cosas a chequear

- **Tiempo de entrenamiento en CPU**: Fase 1 con 64x64 + 3 conv sube el tiempo. Si pasa de 30 min, bajar a 48x48 o reducir filtros (64->48, 128->96).
- **Memoria**: cargar 40k imagenes como `np.array` en RAM puede pasar de 8 GB. Si falla, mover a `tf.data.Dataset.from_tensor_slices` con `batch_size=64` y `prefetch`.
- **Activacion `linear` + LeakyReLU** en el modelo actual: es una decision rara del original. En la nueva arquitectura usar `Conv2D(..., activation="relu")` directamente.
- **No romper el notebook** `CNN.ipynb` y `CNNriesgo.ipynb` (segun README, son material de referencia, no se tocan).

## Orden de ejecucion recomendado

1. Hacer Fase 1.1, 1.2, 1.4. Entrenar. Medir.
2. Si accuracy < 70%: agregar Fase 1.3 (augment).
3. Si accuracy < 80%: hacer Fase 2 entera (diagnostico).
4. Si sigue < 80%: Fase 3 (transfer learning).
5. Siempre cerrar con Fase 4 (docs + entrega).
