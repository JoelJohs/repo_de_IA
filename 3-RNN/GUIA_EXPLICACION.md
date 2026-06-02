# Guia de explicacion - Proyecto 3 (RNN Vanilla)

Material de respaldo para defender el proyecto frente al profesor.
Esta guia asume que el profesor puede preguntar **cualquier cosa**: la
arquitectura, una linea concreta, una decision de diseno, o un numero
del entrenamiento. Cada seccion responde una clase de preguntas.

Si solo tenes 5 minutos: lee el resumen ejecutivo + el Q&A del final.

---

## 1. Resumen ejecutivo (30 segundos para abrir la presentacion)

Proyecto 3 de la materia: **Asistente de Codigo Personalizado con Redes
Recurrentes**. Entrenamos una **RNN Vanilla** (`SimpleRNN`) en Keras/
TensorFlow sobre un corpus de **150 funciones en C** con estilo
consistente. El modelo aprende a nivel de **caracter** (no de token) y
se expone a un editor real (VS Code) mediante dos canales: una API
HTTP (FastAPI) y un servidor JSON-line por `stdin`/`stdout`. El editor
lo invoca con `Ctrl+Shift+Space` e inserta la continuacion generada
en el cursor.

Cumple los dos entregables de la actividad:
1. **Codigo fuente del entrenamiento**: `notebooks/entrenamiento.ipynb`
   (ejecutado, con perdida y ejemplos) + scripts en `src/`.
2. **Dataset**: `dataset/sample/funciones.c` (150 funciones).

Cumple el requisito de modelo: **RNN Vanilla**, no LSTM ni GRU.

---

## 2. Pipeline de datos (lo que ve el profesor si pregunta "como se conecta todo")

```text
                  ┌──────────────────────────────────────────┐
                  │ dataset/funciones.c                      │
                  │ (corpus crudo, 466 MB, 196 480 funciones)│
                  └────────────────┬─────────────────────────┘
                                   │ src/sample_clean.py
                                   │  (regex + heuristicas ASCII/llaves)
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │ dataset/sample/funciones.c               │
                  │ (150 funciones "limpias", 70 KB)         │
                  └────────────────┬─────────────────────────┘
                                   │ src/preprocess.py
                                   │  (vocab chars + ventanas 32)
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │ dataset/processed/                       │
                  │   X.npy (20 000, 32)  Y.npy (20 000, 32)│
                  │   meta.json (chars, block_size, vocab)   │
                  └────────────────┬─────────────────────────┘
                                   │ src/train.py
                                   │  (Embedding + SimpleRNN + TimeDist)
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │ models/rnn_v1.keras (244 KB, 17 854 par) │
                  │ models/rnn_v1.meta.json                  │
                  │ models/rnn_v1.history.json               │
                  └────────────────┬─────────────────────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
       src/predict.py        src/api.py          src/server_stdio.py
       (CLI)                 (FastAPI HTTP)      (JSON-line para editor)
              │                    │                     │
              ▼                    ▼                     ▼
       terminal              curl / Postman        VS Code extension
                                                  (vscode-extension/)
```

Tres consumidores independientes, un solo modelo.

---

## 3. Arquitectura del modelo (la pregunta mas probable)

Arquitectura en `src/train.py:42-50` y `notebooks/entrenamiento.ipynb`
celda 9:

```python
Sequential([
    Input(shape=(BLOCK_SIZE,)),                          # 32 chars
    Embedding(VOCAB_SIZE, 48),                           # 4 512 params
    SimpleRNN(64, activation="tanh", return_sequences=True),  # 7 232 params
    TimeDistributed(Dense(VOCAB_SIZE)),                  # 6 110 params
])
# Total: 17 854 parametros (69.74 KB)
```

### 3.1 Embedding

- **Tarea**: convertir cada `id` de caracter (0..93) en un vector denso de 48D.
- **Matriz aprendible** `E ∈ R^{94 x 48}`.
- **Ecuacion**: `e_t = E[x_t]` donde `x_t` es el id del caracter en la
  posicion `t` y `e_t ∈ R^{48}`.
- **Por que se aprende**: al inicio la matriz es aleatoria; durante el
  entrenamiento, caracteres que aparecen en contextos similares
  (ej. `{` seguido de `\n` seguido de espacios) terminan con embeddings
  parecidos.

### 3.2 SimpleRNN (la "RNN Vanilla" del titulo)

- **Ecuacion explicita** (la unica diferencia con LSTM/GRU es que no
  tiene compuertas):

  ```
  h_t = tanh( W_xh * e_t  +  W_hh * h_{t-1}  +  b_h )
  ```

  donde:
  - `W_xh ∈ R^{64 x 48}` (entrada -> estado)
  - `W_hh ∈ R^{64 x 64}` (estado anterior -> estado actual)
  - `b_h ∈ R^{64}` (sesgo)
  - `h_{t-1}` es el estado oculto del paso anterior
  - `h_{-1}` se inicializa en ceros

- **17 854 - 4 512 - 6 110 = 7 232** parametros (3 matrices de 64x64, 64x48 y 64).
- **"Vanilla" significa esto**: nada de forget/input/output gates
  (LSTM), nada de reset/update gates (GRU). Es exactamente la formula
  de arriba.
- **Por que `return_sequences=True`**: queremos un estado oculto en
  **cada** posicion `t` de la ventana, no solo el ultimo. Esto es
  necesario porque nuestra loss compara `Y[t]` con `Y_pred[t]` para
  todo `t` (siguiente caracter en cada posicion). Si solo
  devolviéramos el ultimo, el modelo solo aprenderia a predecir el
  caracter que viene **despues** de la ventana, no el caracter que
  viene despues de **cada** posicion interna.

### 3.3 TimeDistributed(Dense)

- **Tarea**: en cada posicion `t` de la ventana, producir un vector de
  logits de tamano `VOCAB_SIZE = 94`. Esto es "cual es la probabilidad
  de cada posible siguiente caracter en este punto".
- **Ecuacion**: `y_t = W_hy * h_t + b_y` con `W_hy ∈ R^{94 x 64}` y
  `b_y ∈ R^{94}`.
- **Por que `TimeDistributed`**: aplica la **misma** capa densa a
  cada paso temporal. Es el equivalente Keras de compartir pesos
  `W_hy` y `b_y` entre todos los `t`. Si usaramos `Dense(VOCAB_SIZE)`
  sin `TimeDistributed`, Keras colapsaria los 32 pasos a uno solo.
- **`from_logits=True` en la loss**: como el modelo devuelve logits
  crudos (no softmax), usamos `SparseCategoricalCrossentropy` con
  `from_logits=True` para que Keras aplique el softmax internamente
  (mas estable numericamente que pasarlo a mano).

### 3.4 Forma de los tensores (muy util para que el profesor vea que
entendes las dimensiones)

```
X  : (batch, 32)         int64
Y  : (batch, 32)         int64

Embedding(X) : (batch, 32, 48)         float32
SimpleRNN(.) : (batch, 32, 64)         float32
TimeDist(Dense) : (batch, 32, 94)      float32 (logits)

loss = mean( CrossEntropy( logits[t], Y[t] ) ) sobre batch y t
```

---

## 4. Walkthrough por archivo (con numeros de linea exactos)

### 4.1 `src/sample_clean.py` — limpieza del corpus

| Linea | Que hace | Por que |
|---|---|---|
| `24-31` | Regex `SIG_RE` que matchea firmas de funcion en C | Detecta `<tipo> <nombre> (<args>) {` con un solo pass O(n) |
| `34-58` | Funciones `is_ascii`, `balanced_braces`, `has_main`, `has_chinese_comment`, `has_exotic_preproc` | Heuristicas para descartar funciones con problemas |
| `71-89` | `is_clean_function(code)` retorna `(bool, reason)` | Pipeline de filtros: longitud 30-1500, sin `main`, ASCII, llaves balanceadas, firma reconocible |
| `92-111` | `extract_functions(text)` generator | Dado un texto, yield (signature, block) con depth tracking |
| `142-150` | Loop principal: encuentra todas las firmas, despues cuenta llaves | O(n) en lugar de O(n*m) |
| `61-68` | `count_style(text)` scoring | Premia funciones con `if`/`for`/`return` para que el corpus tenga logica |

**Pregunta probable del profesor**: "Como decis que la funcion esta
'limpia'?" - Respuesta: aplicamos 6 filtros en cascada. Si cualquiera
falla, la descartamos con una razon (`length=X`, `has_main`, `non_ascii`,
`unbalanced_braces`, `no_signature`, `control_keyword_name`). El reporte
en `dataset/sample/REPORTE.md` muestra cuantos candidatos cayo en cada
filtro.

### 4.2 `src/preprocess.py` — ventanas para entrenar

| Linea | Que hace |
|---|---|
| `36` | Lee el corpus como un unico string |
| `37-39` | Construye vocabulario: `chars = sorted(set(text))`, `stoi`, `itos` |
| `41` | Convierte el corpus a secuencia de ids: `seq[i] = stoi[text[i]]` |
| `49-54` | Genera ventanas shifted-by-1: `X[i] = seq[i:i+32]`, `Y[i] = seq[i+1:i+33]` |
| `56-57` | Guarda `X.npy`, `Y.npy` (int64, ~17 MB cada uno) |
| `59-66` | Guarda `meta.json` con vocab, block_size y conteos |

**Por que shifted-by-1?** En cada ventana, queremos predecir el
siguiente caracter para **cada** posicion. Si la ventana es
`[a, b, c, d, e]`, queremos que el modelo aprenda que:
- despues de `a` viene `b`
- despues de `b` viene `c`
- ...
- despues de `e` viene el primer caracter de la siguiente ventana

Esto es lo que hace que el modelo pueda generar autoregresivamente.

### 4.3 `src/train.py` — el modelo y el entrenamiento

| Linea | Que hace |
|---|---|
| `36-39` | `set_seed(42)` — fija random/numpy/tensorflow. Sin esto, cada corrida da numeros distintos. |
| `42-50` | `build_model()` — la arquitectura de 3 capas |
| `53-65` | `PrintEvery` callback — loggea la loss cada 10 epocas |
| `93-97` | Sub-muestrea a 20 000 ventanas (de las 69 842 disponibles) |
| `105-110` | `compile()` con Adam(1e-3) y `SparseCategoricalCrossentropy(from_logits=True)` |
| `112-119` | `fit()` con `verbose=0` (silencioso) y el callback custom |
| `121-123` | Guarda `rnn_v1.keras` |
| `146-164` | Copia `meta.json` al lado del modelo (lo necesita el server para reconstruir el vocabulario) |

**Por que submuestrear a 20 000?** Las 69 842 ventanas son altamente
redundantes (overlap de 31 chars entre ventanas consecutivas). En CPU,
20 000 ventanas × 80 epocas × ~70 ms/step = 11 min. Con todas las
ventanas serian 39 min para el mismo resultado. Aceptamos el tradeoff
de perder un poco de senal por 4x de velocidad.

### 4.4 `src/predict.py` — el corazon de la generacion

| Linea | Que hace |
|---|---|
| `20-45` | `@dataclass RNNModel` envuelve el modelo + `stoi` + `itos` + `block_size` + `vocab_size` |
| `48-57` | `_encode(prefix, stoi, block_size)`: convierte texto a ids, recorta o paddea a `block_size` |
| `60-68` | `_sample(logits, temperature)`: aplica softmax con temperatura. `T<=0` = argmax (greedy) |
| `71-91` | `complete()`: el **bucle autoregresivo** — 30 lineas, lo mas importante del proyecto |
| `94-96` | `complete_full()`: convenience wrapper que devuelve `prefix + text` |

**El bucle autoregresivo, linea por linea** (`src/predict.py:71-91`):

```python
for _ in range(max_new):                              # generar N chars
    logits = rnn.model.predict(state, verbose=0)[0, -1]  # (94,) logits del sig. char
    idx = _sample(logits, temperature)                 # muestrear un id
    ch = rnn.itos[idx]                                 # traducir id -> char
    generated.append(ch)                               # guardar
    state = np.concatenate([state[:, 1:], [[idx]]], axis=1)  # slide window
```

**Que hace `state[:, 1:]`?** La ventana es de tamano 32. Cada vez que
generamos un caracter, la ventana se desplaza: descartamos el primer
caracter (ya no nos sirve, queda fuera del contexto) y agregamos el
nuevo al final. Asi el modelo siempre ve los ultimos 32 caracteres
del texto (prompt + lo que llevo generado).

**Que hace `temperature`?** Divide los logits por `T` antes del
softmax. Si `T=1` es la distribucion natural del modelo. Si `T=0.5`
las probabilidades se vuelven mas picudas (mas determinista, repite
patrones del corpus). Si `T=2` se aplanan (mas creatividad, mas
ruido). Probamos `T=0.4` por default porque el corpus es chico y
temperatura alta degrada rapido.

### 4.5 `src/api.py` — FastAPI

| Linea | Que hace |
|---|---|
| `30-34` | `get_rnn()` singleton lazy: el modelo se carga **una sola vez** al primer request, no por request |
| `37-45` | `PredictRequest`/`SuggestRequest` con Pydantic — validacion automatica de tipos y rangos |
| `48-50` | `GET /health` — para healthchecks |
| `53-57` | `POST /predict` — devuelve `completion` (solo lo generado) y `full` (prompt + generado) |
| `60-70` | `POST /suggest` — top-N chars siguientes, sin muestreo (greedy) |

**Por que `get_rnn` singleton?** Cargar el modelo `.keras` toma ~3 s
(lee 244 KB + reconstruye el grafo). Si lo hicieramos por request,
cada llamada tardaria 3+ s. Con singleton, solo la primera vez.

### 4.6 `src/server_stdio.py` — JSON-line para VS Code

| Linea | Que hace |
|---|---|
| `33-37` | Hack de imports: funciona tanto con `python -m src.server_stdio` como con `python src/server_stdio.py` |
| `47-61` | `handle_complete()` — replica del bucle de `predict.py:71-91` |
| `64-70` | `handle_suggest()` — replica del `/suggest` de la API |
| `73-101` | `main()` — el bucle infinito que lee una linea JSON de stdin y escribe una linea JSON a stdout |
| `93-99` | `try/except` global — si un request falla, devolvemos `{"ok": false, "error": "..."}` en vez de matar el proceso |

**Por que JSON-line y no HTTP?** Las extensiones de VS Code se
comunican con sus servidores como subprocesos (mismo patron que
`gopls`, `rust-analyzer`, `pylsp`, `typescript-language-server`).
HTTP agregaria latencia y un puerto a coordinar. JSON-line es el
estandar de facto para Language Server Protocol.

**Por que `id` en cada request?** El servidor procesa requests en
orden FIFO (es un loop secuencial), pero igual necesitamos
correlacionar request con response. El `id` es unico por llamada.
La extension (`extension.js:100`) genera IDs monotonos y guarda las
callbacks en un `Map` (`pending`).

### 4.7 `vscode-extension/extension.js` — la extension en si

| Linea | Que hace |
|---|---|
| `16-19` | Estado global: `serverProc`, `nextId`, `pending` (Map id -> callback), `buffer` (parseo parcial) |
| `21-27` | `workspaceRootPath()` — primer folder del workspace, base de los paths relativos |
| `29-45` | Lee configuracion (`rnnC.pythonPath`, `rnnC.serverScript`, `rnnC.modelPath`) |
| `47-91` | `startServer()` — `spawn` del subproceso + manejo de stdout/stderr/exit |
| `55-72` | **Parser de JSON-line con buffer parcial**: los chunks pueden cortar lineas, asi que acumulamos en `buffer` y parseamos linea por linea |
| `93-113` | `request()` — Promise-based wrapper alrededor de stdin/stdout |
| `115-120` | `linePrefix()` — toma el texto desde el inicio de la linea actual hasta el cursor |
| `122-137` | `cmdComplete()` — el handler de `Ctrl+Shift+Space`: lee prefix, llama `request("complete", ...)`, inserta el texto |
| `172-179` | `activate()` — VS Code llama esto cuando se abre un archivo `.c`; registra comandos y arranca el server |

**Por que un servidor por sesion y no por request?** Cargar el
modelo toma ~5-10 s (TF warmup). Si lo hicieramos por cada
`Ctrl+Shift+Space`, el usuario esperaria 10 s por sugerencia. Con
un servidor persistente, solo la primera vez.

---

## 5. Resultados (los numeros concretos)

| Metrica | Valor | Como se midio |
|---|---|---|
| Vocabulario | 94 chars unicos | `len(set(corpus))` en `preprocess.py:37` |
| Corpus total | 69 874 chars | `preprocess.py` output |
| Ventanas de entrenamiento | 20 000 (de 69 842) | `train.py --max-windows 20000` |
| Epochs entrenados | 80 | `train.py --epochs 80` |
| Loss inicial | 3.07 | primera epoca del history |
| Loss final | 1.06 | ultima epoca del history |
| Tiempo de entrenamiento | ~5 min | medido en CPU, 4 cores, 17 GB RAM |
| Parametros | 17 854 | `model.summary()` |
| Tamano del modelo | 244 KB | `models/rnn_v1.keras` |
| Loss "al azar" | ln(94) ≈ 4.54 | entropia uniforme sobre 94 chars |

**Que significa loss=1.06?** Cross-entropy promedio por caracter.
Mejor que azar (4.54) por 4.3x, pero todavia ~37% de error por
caracter. No es codigo ejecutable; es un modelo didactico.

**Ejemplos de generacion** (con `temperature=0.4`, `seed=42`):

| Prompt | Generado | Observacion |
|---|---|---|
| `int sum` | `int sum[mid] < nums[0] == 0)` | El modelo "ve" indexing de arrays |
| `void print` | `void printf("FAILED: %s\\n"); }` | Aprende el patron printf + cierre de bloque |
| `for (int` | `for (int source <= '9') { return binarySearch(arr,` | Combina for + return + llamada a funcion |
| `if (a` | `if (array[i]; } if (array[mid] < nums[0])` | Patrones de array indexing |
| `return` | `return right = middle + 1; if ((source <= '9') || (n` | Asignacion + return + condicional |

**Por que esta salida "imperfecta"**: el corpus tiene 70 KB de
texto (vs. los GB de un modelo real). 80 epocas es muy poco para
que la red memorice patrones mas alla de los 32 chars de contexto.
Lo que **si** se ve: sintaxis basica de C, vocabulario de tipos,
uso de funciones conocidas (`strcmp`, `printf`, `binarySearch`,
`array`, `mid`). Lo que **no** se ve: coherencia semantica de
mas de 5-10 chars.

---

## 6. Por que cada decision (las 4 preguntas mas espinosas)

### 6.1 Char-level vs token-level?

| | Char | Token (palabra) |
|---|---|---|
| Vocab | 94 | 500-2000 |
| Tamano embedding | 48 | 64-128 |
| Tiempo CPU | bajo | medio |
| Estructura | aprende llaves, identacion | aprende nombres de funciones |
| Coherencia semantica | baja (char-by-char) | media (palabra-por-palabra) |
| Ejemplo del profesor | **si** | no |

**Ganador: char-level**. El ejemplo del profesor lo usa. Entrena
3x mas rapido en CPU. Y para una RNN Vanilla de 17 K parametros, la
diferencia en "coherencia" no compensa el costo.

### 6.2 SimpleRNN vs LSTM/GRU?

- **LSTM**: agrega compuertas `forget`, `input`, `output` → mejor
  memoria de largo plazo, ~4x mas parametros.
- **GRU**: version simplificada de LSTM con 2 compuertas → ~3x mas
  parametros que SimpleRNN.
- **SimpleRNN**: la formula `h_t = tanh(W_xh·e_t + W_hh·h_{t-1} + b_h)`.
  Memoria efectiva: ~5-10 pasos antes de que el gradiente se evapore.

**Ganador: SimpleRNN**. La actividad 3 lo exige explicitamente
("Utilizar el apartado de Autocompletado con RNN Vanilla en Keras").
Ademas, con `BLOCK_SIZE=32` la limitacion de memoria no es un
problema real: el modelo nunca necesita mirar mas alla de 32 chars
atras.

### 6.3 150 funciones vs 200 000?

- El corpus crudo tiene ~200 000 funciones heterogeneas (mezcladas
  con `int main()`, comentarios en chino, llaves inconsistentes).
- "Estilo consistente" + "60 funciones" es el requisito literal.
- 150 funciones = buen balance entre variedad (cubre sorting, math,
  strings, arrays) y consistencia.

**Ganador: 150**. Mas datos sin filtrar harian *peor* modelo:
aprenderia estilos contradictorios.

### 6.4 FastAPI + stdio (no solo uno)?

- **FastAPI**: util para `curl`/Postman y para que el profesor lo
  pruebe sin instalar nada extra. Tambien es la "sugerencia" literal
  de la actividad ("Flask o FastAPI").
- **stdio**: patron estandar para extensiones de editor. Mas
  eficiente (sin overhead de HTTP) y es exactamente como funcionan
  las extensiones serias (Copilot, Tabnine, Continue).

**Ganador: ambos**. Cumplen roles distintos. Comparten
`src/predict.py:complete()`.

---

## 7. Q&A anticipado (12 preguntas probables)

### "Por que `return_sequences=True`?"

`src/train.py:47`. Sin esto, `SimpleRNN` solo devolveria el estado
del **ultimo** paso. Pero nosotros queremos un logit en **cada**
paso, porque la loss compara `Y[t]` con la prediccion en la
posicion `t` para todo `t` de la ventana. Si solo tuvieramos el
ultimo, el modelo no aprenderia a predecir el caracter siguiente
para las posiciones internas — solo para la posicion 32.

### "Que significa la loss de 1.06?"

Es cross-entropy promedio por caracter. La entropia de una
distribucion uniforme sobre 94 chars es `ln(94) = 4.54`. Una loss
de 1.06 es ~4.3x mejor que azar. En terminos practicos: cuando el
modelo ve un caracter, concentra ~e^(-1.06) = 35% de la masa de
probabilidad en los chars correctos vs. ~1% si fuera uniforme.

### "Por que entrenar 80 epocas y no 200?"

A partir de la epoca 50, la loss ya apenas baja (plateau). Las
primeras 20 epocas son las que mas aprenden. Con 80 nos aseguramos
de haber llegado al plateau. Con 200 seria浪费时间 (mismo modelo
final).

### "Por que `from_logits=True` en la loss?"

`src/train.py:108`. El modelo devuelve logits crudos (la salida de
`Dense` sin softmax). Pasar `from_logits=True` le dice a Keras que
aplique softmax internamente — es numericamente mas estable que
hacer softmax a mano y luego `log`. Es lo que recomienda la doc de
Keras.

### "Que pasa si pongo `temperature=0`?"

`src/predict.py:60-62`. Con `T=0`, `_sample` hace `argmax` en vez
de muestreo. La salida es completamente determinista: siempre el
mismo char dado el mismo contexto. Util para reproducibilidad
pero produce texto repetitivo.

### "Por que 80 epocas y no 1000?"

`src/train.py:73`. La loss en CPU por epoca es ~3.5 s. 1000
epocas = ~1 hora. 80 epocas = ~5 min y llegamos al plateau. Mas
epocas no ayudarian con un corpus de 70 KB.

### "Por que el modelo a veces genera codigo que no compila?"

`src/predict.py:71-91`. La RNN Vanilla no tiene nocion de
gramatica. Solo aprendio que despues de `{` viene un `\n` y despues
un `if` o `for`. No valida llaves balanceadas ni parentesis.
Esto es esperado y documentado como limitacion.

### "Por que `BLOCK_SIZE=32` y no 128?"

`src/preprocess.py:28`. Es el ejemplo literal del profesor. Con
32 chars vemos el contexto suficiente para predecir el siguiente
char razonablemente. Con 128 la red tendria que mantener
informacion de 4x mas pasos, lo cual la RNN Vanilla hace mal
(gradientes que se evaporan).

### "Como se conecta con VS Code?"

`vscode-extension/extension.js:47-53`. La extension hace `spawn`
de `python src/server_stdio.py models/rnn_v1.keras`. El subproceso
vive mientras VS Code este abierto. Cada `Ctrl+Shift+Space` manda
una linea JSON a stdin y recibe una linea JSON de stdout. El
modelo se carga **una sola vez** al activarse.

### "Por que no usar LSTM o GRU?"

La actividad 3 dice textual: "RNN Vanilla". Ademas, con
`BLOCK_SIZE=32`, la limitacion de memoria de SimpleRNN no es un
problema. Si tubieramos bloques de 256 chars, ahi si
necesitariamos LSTM.

### "Por que el primer completion tarda 10 segundos y los siguientes son instantaneos?"

`vscode-extension/extension.js:47-53`. El primer `request()`
dispara `startServer()` que hace `spawn` del subproceso. El
subproceso carga `rnn_v1.keras` + compila el grafo de TF. Esto
toma 5-10 s. Los siguientes requests ya tienen el modelo en
memoria, asi que cada `predict()` toma ~50 ms.

### "Por que no usaste LM Studio u Ollama?"

Esas herramientas son para modelos grandes de lenguaje (LLMs) y
pertenecen al **Proyecto 4** (Fine-Tuning / RAG). La actividad 3
pide explicitamente entrenar una RNN desde cero con Keras, no
consumir un LLM pre-entrenado.

---

## 8. Limitaciones honestas (si el profesor pregunta)

1. **Memoria corta**: 32 chars. Una linea de 100 chars pierde el
   inicio. Para mejorar: LSTM/GRU, o aumentar `BLOCK_SIZE` y
   combinar con positional encoding estilo Transformer.

2. **Sin gramatica**: el modelo no sabe que `(((` no es valido C.
   Para mejorar: integrar un parser (LibClang) o entrenar con
   sintaxis validada.

3. **Coherencia semantica baja**: pasado 10-15 chars, el texto
   generado deja de tener sentido. Es un modelo de **lenguaje**,
   no de **comprension**.

4. **Sin GPU**: ~5 min de entrenamiento en CPU. Con una GPU
   moderna seria <30 s. No afecta la calidad, solo la velocidad.

5. **Dataset pequeno**: 70 KB vs. los GB de un modelo de
   produccion. Para mejorar: usar CodeSearchNet, GitHub scrape,
   o el corpus propio del usuario (la consigna lo permite).

6. **Solo completado por caracter**: no tenemos "suggest" estilo
   Copilot (multi-token, multi-linea). Para mejorar: beam search
   o sustituir por Transformer.

---

## 9. Archivos para mostrar al profesor (en orden)

Si el profesor quiere ver codigo, abrir en este orden (de simple a
complejo):

1. `src/predict.py:71-91` — el bucle autoregresivo (30 lineas, el
   corazon del proyecto).
2. `src/train.py:42-50` — la arquitectura (10 lineas).
3. `src/server_stdio.py:73-101` — el JSON-line server (30 lineas).
4. `vscode-extension/extension.js:122-137` — el handler de VS Code
   (15 lineas).
5. `src/api.py` — la API REST (70 lineas).
6. `notebooks/entrenamiento.ipynb` — todo el flujo, ejecutado.

---

## 10. Glosario rapido (por si el profesor usa terminos que no manejas)

| Termino | Significado |
|---|---|
| **Embedding** | Mapa de ids a vectores densos. Aprendido durante el entrenamiento. |
| **Hidden state** | El vector `h_t` de la RNN. Resume todo el contexto hasta el paso `t`. |
| **return_sequences** | Si la capa devuelve un output por paso o solo el del ultimo paso. |
| **TimeDistributed** | Aplicar la misma capa densa a cada paso de una secuencia, con pesos compartidos. |
| **Logits** | Salida cruda de la capa densa, antes de softmax. Pueden ser negativos. |
| **from_logits** | Flag de la loss: aplica softmax internamente en vez de esperar probs. |
| **Temperature** | Dividir logits por T antes de softmax. T<1 = mas determinista, T>1 = mas aleatorio. |
| **Autoregresivo** | Generar un token a la vez, alimentar el generado como input del siguiente paso. |
| **JSON-line** | Protocolo: una linea = un mensaje JSON. Usado por LSP, Copilot, etc. |
| **Spawn** | Crear un subproceso. En Node: `child_process.spawn(cmd, args)`. |
| **Singleton** | Patron que garantiza una sola instancia. En nuestro caso, el modelo cargado una vez. |
| **Lazy load** | Inicializar solo cuando se necesita por primera vez. |
| **Heuristica** | Regla simple y rapida (no optima) para filtrar/seleccionar. |
| **Vocabulary (vocab)** | Conjunto de tokens unicos que el modelo conoce. Para nosotros: 94 chars. |
| **BPTT** | Backpropagation Through Time. Como se calcula el gradiente en una RNN. |
