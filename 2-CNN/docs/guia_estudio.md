# Guia de estudio — 2-CNN

Documento corto para defender el proyecto en clase. Cubre arquitectura,
flujo de datos, modelo, inferencia, la UI PyQt6 y preguntas tipo examen.

## 1. Arbol de archivos

```
2-CNN/
├── app.py                       # entry point CLI: python app.py
├── train.py                     # entrenamiento
├── predict.py                   # inferencia por CLI
├── config/
│   ├── default.yaml             # hiperparametros y paths
│   ├── artifacts/
│   │   ├── classes.json         # lista de clases en orden alfabetico
│   │   └── metrics.json         # loss + accuracy del test
│   └── models/
│       └── best_model.keras     # modelo entrenado
├── dataset/
│   ├── aranas/  ballenas/  monos/  pajaros/  ranas/
├── test/                        # muestras pequenas para probar rapido
├── src/
│   ├── config.py                # dataclasses + loader YAML
│   ├── utils.py                 # logging, seed, json, paths
│   ├── data/
│   │   ├── manual_datasets.py   # carga manual con os.walk + split
│   │   ├── datasets.py          # variante tf.data (image_dataset_from_directory)
│   │   ├── build_processed_dataset.py  # resize + recodifica JPG
│   │   └── sample_to_test.py    # mueve N imagenes de dataset/ a test/
│   ├── models/
│   │   └── baseline_cnn.py      # build_baseline_cnn()
│   ├── training/
│   │   └── trainer.py           # compile + fit + early stopping
│   ├── inference/
│   │   └── predict.py           # predict_image() -> top-k
│   └── ui/                      # GUI PyQt6
│       ├── launcher.py          # main(): carga modelo + crea ventana
│       ├── main_window.py       # QMainWindow con 3 paneles
│       ├── styles.py            # QSS dark theme
│       └── widgets/
│           ├── class_card.py    # tarjeta por clase (emoji + count)
│           ├── image_viewer.py  # QLabel escalado smooth
│           └── prob_bar.py      # barras top-1/2/3
└── docs/
    ├── decisiones.md            # que se eligio y por que
    ├── explicaciones.md         # prosa para defender el codigo
    └── guia_estudio.md          # este archivo
```

## 2. Flujo del pipeline

```
dataset/<clase>/*.jpg
        │
        ▼
src/data/manual_datasets.build_manual_splits()  ← os.walk, resize, /255
        │
        ▼
src/models/baseline_cnn.build_baseline_cnn()    ← Conv2D → MaxPool → Dense
        │
        ▼
src/training/trainer.compile_model() + .fit()   ← SGD/Adam + early stopping
        │
        ▼
config/models/best_model.keras + classes.json + metrics.json
        │
        ▼
predict.py (CLI)        o    src/ui/launcher.py (PyQt6)
        │                          │
        └────► predict_image() ◄───┘
                      │
                      ▼
              top-3 (clase, prob)
```

`predict_image()` (`src/inference/predict.py:18`) hace, en orden:

1. `tf.keras.utils.load_img(path, target_size=image_size)` → PIL.Image.
2. `tf.keras.utils.img_to_array(image)` → array HxWxC.
3. Divide por 255 (normalización a [0,1]).
4. `tf.expand_dims` para agregar la dimensión de batch.
5. `model.predict(batch, verbose=0)[0]` → vector de probabilidades.
6. `np.argsort(probs)[::-1][:top_k]` → índices del top-k.
7. Devuelve `List[Tuple[class_name, prob]]`.

## 3. Responsabilidad por archivo (1 linea)

| Archivo | Que hace |
|---|---|
| `app.py` | entry point CLI que delega a `src.ui.launcher.main` |
| `train.py` | orquesta el pipeline de entrenamiento |
| `predict.py` | inferencia por linea de comandos |
| `config/default.yaml` | hiperparametros centralizados |
| `src/config.py` | dataclasses + `load_config()` |
| `src/utils.py` | logging, seed, json, ensure_dir |
| `src/data/manual_datasets.py` | carga imágenes con `os.walk` y hace split |
| `src/data/datasets.py` | variante con `tf.data` (mas eficiente en RAM) |
| `src/data/build_processed_dataset.py` | resize + JPG desde carpeta cruda |
| `src/data/sample_to_test.py` | mueve N imagenes por clase a `test/` |
| `src/models/baseline_cnn.py` | arquitectura (Conv → Pool → Dense) |
| `src/training/trainer.py` | compile + fit + callbacks |
| `src/inference/predict.py` | `predict_image()` top-k |
| `src/ui/launcher.py` | arranca Qt, carga modelo y abre la ventana |
| `src/ui/main_window.py` | ventana 3 paneles + worker thread |
| `src/ui/styles.py` | QSS dark theme |
| `src/ui/widgets/class_card.py` | tarjeta visual por clase |
| `src/ui/widgets/image_viewer.py` | QLabel con pixmap escalado |
| `src/ui/widgets/prob_bar.py` | barra horizontal de probabilidad |

## 4. Como se almacenan los datos

- `dataset/<clase>/<clase>__<origen>__<6-digitos>.jpg` (formato de los JPG
  ya procesados). Total: 40,808 imágenes, ~8,100 por clase (balanceado).
- `test/<clase>__<resto>.jpg` (muestras pequenas movidas con
  `sample_to_test.py`, ~30 imágenes en total).
- `config/artifacts/classes.json` → `{"classes": ["aranas", "ballenas", ...]}`,
  en orden alfabético. Es la fuente de verdad para mapear índice → nombre.
- `config/artifacts/metrics.json` → pares `key: value` con `loss` y
  `compile_metrics` (accuracy) del último `model.evaluate(test_X, test_Y)`.
- `config/models/best_model.keras` → el modelo Keras entero (arquitectura +
  pesos). Se carga con `tf.keras.models.load_model(path)`.

**Punto clave para el profesor:** el orden de `class_names` que sale de
`tf.keras.utils.image_dataset_from_directory` (o de `os.listdir` ordenado)
**define** el mapeo `argmax → clase`. Si en el futuro se cambia el orden
de carpetas, hay que reentrenar. `classes.json` se usa SOLO en inferencia
para mapear probabilidades a nombres legibles.

## 5. Arquitectura del modelo

`build_baseline_cnn()` (`src/models/baseline_cnn.py:8`) implementa una
CNN minimal, fiel al notebook del profesor:

```
Input (32, 32, 3)
  → Conv2D(32, 3x3, padding='same', activation='linear')
  → LeakyReLU(0.1)
  → MaxPooling2D(2x2, padding='same')     # 16x16x32
  → Dropout(0.5)
  → Flatten                               # 8192
  → Dense(32, activation='linear')
  → LeakyReLU(0.1)
  → Dropout(0.5)
  → Dense(num_classes, activation='softmax')
```

- **Por que 1 sola capa conv**: el notebook del profesor lo hace asi y la
  pauta pide seguirlo (ver `docs/decisiones.md:6`).
- **Por que `linear` + `LeakyReLU`**: replica exactamente la arquitectura
  del notebook.
- **Por que SGD**: igual al notebook original (`learning_rate=0.001`,
  `decay=lr/100`, `epochs=20`). La "Mejora A" sube a `lr=0.005` y 40
  epochs, pero mantiene SGD.
- **Por que softmax en la salida**: clasificación multiclase mutuamente
  excluyente, las probabilidades suman 1.
- **Por que categorical crossentropy**: etiqueta es one-hot
  (`tf.keras.utils.to_categorical`).

## 6. Inferencia — `predict_image`

Codigo (`src/inference/predict.py:10-30`) resumido:

```python
def predict_image(model, class_names, image_path, image_size, top_k=3, logger=None):
    image = tf.keras.utils.load_img(image_path, target_size=image_size)
    array = tf.keras.utils.img_to_array(image)
    array = array / 255.0
    batch = tf.expand_dims(array, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_indices]
```

Decisiones:

- **Cargar y redimensionar a `image_size`**: el modelo espera ese tamaño
  fijo; `load_img` lo hace con bilinear por defecto.
- **Dividir por 255**: el modelo fue entrenado con imágenes normalizadas
  (mismo preprocesamiento en train y en inferencia).
- **`expand_dims`**: convierte `(32,32,3)` → `(1,32,32,3)` para que el
  modelo lo vea como batch de 1.
- **`np.argsort(probs)[::-1]`**: ordena descendente, toma los `top_k`
  primeros.
- **Devuelve tuplas `(str, float)`**: el `float` está en [0,1] y suma ≈1
  con los demás.

## 7. UI PyQt6 (`src/ui/`)

La ventana se lanza con `./scripts/cnn_ui.fish` o con `python app.py`.
Internamente: `cd 2-CNN && python -m src.ui.launcher`.

Layout de 3 paneles:

- **Izquierda** — `ClassListPanel` con una `ClassCard` por clase. Cada
  tarjeta muestra emoji + nombre + conteo de imágenes en `dataset/`. El
  total aparece abajo.
- **Centro** — `ImageViewer` arriba (QPixmap con `SmoothTransformation`,
  se reescala al redimensionar la ventana). Abajo: el panel de resultado
  con `Real:` (clase extraída del nombre del archivo), `Pred:` (top-1 del
  modelo) y un `✅ Correcto` o `❌ Incorrecto` + 3 `ProbBar` con el top-3.
- **Derecha** — radio buttons `Test`/`Dataset` para alternar origen, un
  `QListWidget` con las imágenes, y los botones `◀ Ant`, `🎲 Aleatoria`,
  `Sig ▶` para navegar. `Esc` cierra, `Ctrl+O` abre un `QFileDialog`.

### Prediccion sin congelar la UI

`MainWindow._predict()` crea un `PredictWorker(QThread)` y lo arranca. El
worker llama a `predict_image(...)` y emite la señal `finished(path,
predictions, dt_ms)`. La ventana principal recibe la señal en el hilo de
Qt y actualiza los widgets. Si el usuario cambia de imagen mientras hay
una predicción en vuelo, el `_on_predict_finished` descarta el resultado
si el `path` no coincide con el item actualmente seleccionado.

### Como se extrae la clase real

`_extract_true_class(path, class_names)` (`main_window.py`):

1. Toma `os.path.basename(path)`.
2. Si empieza con `<clase>__` (formato del dataset y de `sample_to_test`),
   esa es la clase real.
3. Si no, intenta con el nombre de la carpeta padre
   (`dataset/aranas/...` → `aranas`).
4. Si nada coincide, devuelve `None` y la UI muestra `(desconocida)` sin
   marcar ✅/❌.

Esto es lo que permite que la UI "compare directamente la imagen
seleccionada" (predicción vs clase real) sin pedirle nada al usuario.

## 8. Atajos de la UI

| Atajo | Accion |
|---|---|
| Click en la lista | selecciona imagen y predice |
| Doble click | (idem, además muestra detalles) |
| Radio Test / Dataset | cambia origen de la lista |
| `◀ Ant` / `Sig ▶` | navega una imagen |
| `🎲 Aleatoria` | salta a una imagen al azar |
| `Ctrl+O` | abre un `QFileDialog` para probar cualquier imagen |
| `Esc` | cierra la ventana |

## 9. Cargar carpeta externa a test

Flujo para probar el modelo con imagenes nuevas que NO estan en
`dataset/` (por ejemplo, chimp, ruiseñor, dardo, fotos propias, etc.):

### Manual (script)

1. Poner las imagenes en `2-CNN/prerender/<clase>/...` con la estructura
   de subcarpetas que se quiera (las subcarpetas son opcionales; el
   script tambien funciona con todas las imagenes sueltas en la raiz).
2. Ejecutar:

   ```
   python -m src.data.load_external_test_folder --source prerender --test-dir test
   ```

   Flags utiles:
   - `--source PATH` (default `prerender/`)
   - `--test-dir PATH` (default `test/`)
   - `--image-size HxW` (default `32x32`)
   - `--jpeg-quality N` (1-100, default 90)
   - `--overwrite` (default: skip si ya existe)
   - `--dry-run` (cuenta sin mover)
3. El script redimensiona cada imagen a 32x32, la mueve a `test/`
   con su nombre original preservado, e imprime un resumen:
   `nuevas / ya_existian / rechazadas / por_subcarpeta`.

Codigo: `src/data/load_external_test_folder.py`. Funcion reutilizable
para la UI: `convert_external_folder_to_test()`.

### UI (boton "Cargar carpeta test")

1. Abrir la UI con `./scripts/cnn_ui.fish`.
2. En el panel derecho, click en `📁 Cargar carpeta test` (debajo de
   los radios `Test`/`Dataset`).
3. Se abre un `QFileDialog` que por defecto sugiere `2-CNN/prerender/`.
   Elegir la carpeta origen.
4. La UI dispara un `LoadFolderWorker(QThread)` que llama a la misma
   funcion del script. Mientras corre, el boton queda deshabilitado y
   la status bar muestra el progreso.
5. Al terminar, sale un `QMessageBox` con el resumen
   (`Nuevas / Ya existian / Con error / Por subcarpeta`).
6. La lista de imagenes se refresca automaticamente, se auto-selecciona
   la primera imagen nueva y se dispara la prediccion.

### Importante

- Las imagenes movidas a `test/` muestran `Real: (desconocida)` en la
  UI porque las subcarpetas de `prerender/` no estan en las 5 clases
  entrenadas y los nombres se preservan sin prefijo.
- Esto es **correcto** para imagenes fuera de distribucion (OOD):
  el modelo solo conoce 5 clases (aranas, ballenas, monos, pajaros,
  ranas). Para chimp/ruiseñor/dardo, lo util es mirar la prediccion
  top-3: el modelo dira cual de las 5 clases conocidas se parece mas.
- Las imagenes de `prerender/` se **mueven** (no copian): despues de
  la operacion, `prerender/` queda vacia. Si se quieren conservar,
  sacar copia de seguridad antes.
- `prerender/` esta ignorada en `.gitignore` para no versionar las
  imagenes por accidente.

## 10. Preguntas tipo examen (respuesta corta)

**Dataset y preprocesamiento**

1. *Cuantas clases tiene el proyecto y cuales son?*
   5: aranas, ballenas, monos, pajaros, ranas.
2. *Cuantas imagenes tiene el dataset y como estan distribuidas?*
   40,808 imagenes en total, ~8,100 por clase (balanceado).
3. *Que tamaño tienen las imagenes de entrada?*
   32×32×3 (alto×ancho×canales RGB).
4. *Como se normalizan los pixeles?*
   Dividiendo entre 255 → rango [0,1].
5. *Que split se usa para train/val/test?*
   80/20 entre train y resto, luego 50/50 entre val y test
   (`validation_split=0.2`, `test_split=0.2`).
6. *Por que stratify en el split?*
   Para mantener la proporcion de clases en cada subconjunto.
7. *Que extension tienen los archivos?*
   `.jpg` despues de `build_processed_dataset.py`, nombrados
   `<clase>__<origen>__<6-digitos>.jpg`.

**Modelo**

8. *Que tipo de red es?*
   Una CNN (Convolutional Neural Network) secuencial de Keras.
9. *Cuantas capas convolucionales tiene?*
   1 sola (el baseline replica el notebook del profesor).
10. *Que tamaño de filtro usa?*
    3×3 con padding `same` (mantiene la resolución espacial).
11. *Que hace MaxPooling2D(2,2)?*
    Reduce el feature map a la mitad en cada dimension (32×32 → 16×16).
12. *Que funcion de activacion tiene la capa de salida y por que?*
    Softmax, porque es clasificación multiclase y queremos una
    distribución de probabilidad sobre las 5 clases (suman 1).
13. *Que funcion de perdida se usa?*
    `CategoricalCrossentropy` (etiquetas en one-hot).
14. *Que optimizador se usa por default y por que?*
    SGD con `learning_rate=0.001` y `decay=lr/100`, igual al notebook
    del profesor. Mejora A: `lr=0.005` y 40 epochs.
15. *Cuantos parametros entrenables tiene?*
    ~262k. Conv: 32×3×3×3 + 32 = 896. Dense 1: 32×8192 + 32 = 262,176.
    Dense 2: 5×32 + 5 = 165. Total ≈ 263,237.
16. *Por que Dropout(0.5)?*
    Para regularizar y reducir overfitting (apaga 50% de las neuronas
    en cada paso de entrenamiento).

**Inferencia**

17. *Que hace `tf.keras.utils.load_img`?*
    Carga una imagen desde disco y la redimensiona a `target_size`,
    devolviendo un objeto PIL.Image.
18. *Por que se divide entre 255?*
    Para normalizar a [0,1], igual que en el entrenamiento.
19. *Por que se usa `tf.expand_dims`?*
    Para agregar la dimension de batch (None, 32, 32, 3) que el modelo
    espera.
20. *Que devuelve `model.predict`?*
    Un array de shape `(1, num_classes)` con las probabilidades para
    cada clase; se indexa con `[0]` para sacar el batch de 1.
21. *Que hace `np.argsort(probs)[::-1][:top_k]`?*
    Ordena los indices de menor a mayor probabilidad, los invierte
    (descendente) y toma los `top_k` mejores.
22. *Que contiene `classes.json`?*
    La lista de nombres de clase en el mismo orden que los indices de
    la capa de salida (orden alfabetico). Es el mapping
    `indice → nombre_clase`.
23. *Como se decide la clase predicha?*
    `argmax` del vector de probabilidades (o equivalentemente, el
    primer elemento del top-1 ordenado).

**UI y menu**

24. *Que framework usa la UI y por que?*
    PyQt6. Ya está en `requirements.txt`, encaja con el stack del
    curso y permite control fino del estilo.
25. *Como se compara la prediccion con la clase real?*
    `_extract_true_class` lee el prefijo `<clase>__` del nombre del
    archivo o la carpeta padre; la UI muestra ✅ si coincide con el
    top-1, ❌ si no.
26. *Que hace el boton 🎲 Aleatoria?*
    Elige una imagen al azar del origen actual (test/ o dataset/).
27. *Donde corre la prediccion y por que no congela la UI?*
    En un `PredictWorker(QThread)` separado. Las señales Qt vuelven
    al hilo principal para actualizar los widgets.
28. *Que pasa si seleccionas una imagen mientras hay una prediccion
    en vuelo?*
    El `_on_predict_finished` descarta el resultado si el `path` no
    coincide con el item actualmente seleccionado.
29. *Como se lanza la UI desde el menu interactivo?*
    `./scripts/menu.fish` → opcion `2` (CNN) → opcion `6`
    (`INTERFAZ grafica PyQt6`) → `run_script cnn_ui`.
30. *Que pasa si el modelo no existe al abrir la UI?*
    `launcher.main()` muestra un `QMessageBox` con el comando
    `./scripts/trains/train_cnn.fish` y termina con codigo 1.

**Decisiones y comparaciones con el notebook**

31. *Por que el modelo baseline tiene una sola capa conv?*
    Para mantenerse 100% fiel al notebook del profesor y poder comparar
    resultados directo (ver `docs/decisiones.md:6`).
32. *Por que no hay data augmentation?*
    El notebook tampoco lo usa; se mantiene la comparacion
    (`docs/decisiones.md:9`).
33. *Que mejora Aumento de epochs?*
    Subir de 20 a 40 epochs y `lr` de 0.001 a 0.005. Sin cambiar
    arquitectura (`docs/explicaciones.md:31-35`).
34. *Que haria subir a una arquitectura mas profunda?*
    Pasar de 1 a 3 bloques conv (32→64→128 filtros) con BatchNorm
    entremedio. Estimado: 70-78% accuracy (ver
    `docs/PLAN_MEJORA.md`).

**Cargar carpeta externa (flujo OOD)**

35. *¿Para que sirve el boton "Cargar carpeta test"?*
    Para tomar imagenes de cualquier carpeta (incluso con clases que
    el modelo no conoce, como chimp/ruiseñor/dardo) y moverlas a
    `test/` ya redimensionadas a 32x32.
36. *¿Que pasa con clases que el modelo no entreno (ej: chimp)?*
    El modelo igual predice una de las 5 clases conocidas (la mas
    parecida semanticamente). La UI muestra `Real: (desconocida)` y
    la prediccion top-1/top-3.
37. *¿Por que las imagenes se mueven y no se copian?*
    Para no duplicar espacio en disco. Si se quieren conservar en
    `prerender/`, sacar una copia de seguridad antes de ejecutar el
    script o el boton.
38. *¿Donde queda guardada la "clase real" de una imagen nueva?*
    En ningun lado automatico. La UI la infiere por el prefijo del
    nombre o la carpeta padre; si no matchea ninguna clase
    entrenada, queda como `(desconocida)`.
39. *¿Que hace la funcion `convert_external_folder_to_test`?*
    Recorre la carpeta origen con `os.walk`, redimensiona cada
    imagen a 32x32 con `tf.image.resize` (bilinear), la guarda
    como JPG en `test/` con su nombre original y devuelve un
    resumen con contadores `written` / `skipped` / `rejected`.
    Es la misma funcion que usan el script CLI y la UI (una sola
    fuente de verdad).
40. *¿Que pasa si dos imagenes tienen el mismo nombre en
    subcarpetas distintas?*
    La segunda hace colision en `test/`. Por defecto el script
    la salta con un warning (no sobrescribe). Con `--overwrite`
    se puede forzar la sobreescritura.
