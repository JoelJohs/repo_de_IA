# Guia de estudio — 3-RNN

Documento corto para defender el proyecto en clase. Cubre arquitectura,
pipeline de datos (F1→F7), el modelo, los tres consumidores (CLI, HTTP,
extension VSCode) y las preguntas tipo examen.

## 1. Arbol de archivos

```text
3-RNN/
├── .python-version              # pyenv 3.11.9 (vive en la raiz del repo)
├── dataset/
│   ├── funciones.c              # corpus crudo (466 MB, opcional, ignorado por git)
│   ├── clean/                   # 8 archivos .c categorizados (arrays, strings, …)
│   ├── curated/                 # 640 funciones curadas (8 cats × 80)
│   │   ├── funciones.c          # concatenado, input real a preprocess.py
│   │   ├── REPORTE.md
│   │   ├── arrays.c, io_system.c, lists.c, math.c, …
│   ├── sample/                  # 150 funciones del sample_clean.py original
│   │   ├── funciones.c
│   │   └── REPORTE.md
│   └── processed/               # ventanas char-level (F2)
│       ├── X.npy                # (278 185, 32) int64
│       ├── Y.npy                # (278 185, 32) int64
│       └── meta.json
├── models/
│   ├── rnn_v1.keras             # ~244 KB, 17 854 parametros
│   ├── rnn_v1.meta.json         # vocab + block_size + params
│   └── rnn_v1.history.json      # curva de perdida (80 epochs)
├── notebooks/
│   └── entrenamiento.ipynb      # entrega formal ejecutado
├── src/
│   ├── sample_clean.py          # F1: muestreo del corpus crudo
│   ├── curate_per_category.py   # F1': curado por categoria (usado por train.fish)
│   ├── preprocess.py            # F2: ventanas char-level
│   ├── train.py                 # F3: arquitectura + entrenamiento
│   ├── predict.py               # F4: complete() + complete_full(), CLI
│   ├── api.py                   # F5: FastAPI
│   ├── server_stdio.py          # F6: JSON-line para VS Code
│   ├── dataset_tools.py         # utilidad original (no usada en el flujo principal)
│   └── __init__.py
├── vscode-extension/            # F7: extension VS Code
│   ├── package.json
│   ├── extension.js
│   ├── test_e2e.js              # smoke test del protocolo JSON-line
│   ├── prueba.c                 # archivo de prueba para F5
│   ├── README.md
│   └── .vscode/                 # launch.json + settings.json
├── scripts/
│   ├── train.fish               # orquestador curate → preprocess → train
│   ├── predict.fish             # wrapper de predict.py
│   ├── curate.fish              # corre curate_per_category.py
│   └── rnn-complete.fish        # atajo historico
├── docs/
│   ├── INICIO.md
│   ├── ENTORNO.md               # por que Python 3.11.9
│   ├── ESTRUCTURA.md
│   ├── DATASET.md
│   ├── DATASET_PROCESO.md
│   └── guia_estudio.md          # este archivo
├── requirements.txt
├── GUIA_DEMO.md                 # guion de demo al profesor (4 modalidades)
├── GUIA_EXPLICACION.md          # respaldo para preguntas abiertas
└── README.md
```

> El venv vive en `../../IA/.venv` (compartido con 1-game y 2-CNN).
> Ver `docs/ENTORNO.md` para el setup completo y la razon de Python 3.11.9.

## 2. Flujo del pipeline (F1→F7)

```
dataset/clean/<categoria>.c        (8 archivos: arrays, strings, trees, …)
        │
        ▼
src/curate_per_category.py         F1': 640 funciones (8 × 80)
        │   (filtros: ASCII, length 200-600, no main, control flow ≥ 1)
        ▼
dataset/curated/funciones.c        ~278 KB, concatenado
        │
        ▼
src/preprocess.py                  F2: vocab chars + ventanas (32 chars)
        │   X[i] = seq[i:i+32],  Y[i] = seq[i+1:i+33]
        ▼
dataset/processed/{X.npy, Y.npy, meta.json}
        │   shape = (278 185, 32) int64
        ▼
src/train.py                       F3: Embedding → SimpleRNN → TimeDistributed(Dense)
        │   Adam(1e-3), SparseCategoricalCrossentropy(from_logits=True)
        │   80 epochs, batch=128, 17 854 params
        ▼
models/rnn_v1.keras + meta.json + history.json
        │
        ├─────────────────┬────────────────────┐
        ▼                 ▼                    ▼
src/predict.py        src/api.py         src/server_stdio.py
   (F4: CLI)            (F5: HTTP)         (F6: JSON-line)
        │                 │                    │
        ▼                 ▼                    ▼
  terminal          curl / Postman       VS Code extension
                                        (vscode-extension/ → F7)
```

**`complete()`** (`src/predict.py:71`) hace, en orden:

1. `_encode(prefix, stoi, block_size)` → `(1, 32)` int64 (left-pad con `pad_id=0`).
2. Bucle de `max_new` iteraciones:
   - `rnn.model.predict(state, verbose=0)[0, -1]` → `(vocab,)` logits del
     último paso.
   - `_sample(logits, temperature)` → índice del siguiente char.
   - Append a `generated`, `state = concat(state[:, 1:], [[idx]])` (slide
     window).
3. Devuelve `"".join(generated)`.

## 3. Responsabilidad por archivo (1 linea)

| Archivo | Que hace |
|---|---|
| `src/sample_clean.py` | F1: sampleo heuristico del corpus crudo de 466 MB |
| `src/curate_per_category.py` | F1': curado estricto por categoria (8 archivos → 640 fns) |
| `src/preprocess.py` | F2: vocab + ventanas char-level (X.npy, Y.npy) |
| `src/train.py` | F3: arquitectura RNN vanilla + `model.fit()` + save |
| `src/predict.py` | F4: `RNNModel` + `complete()` + CLI |
| `src/api.py` | F5: FastAPI con `/predict` y `/suggest` |
| `src/server_stdio.py` | F6: JSON-line por stdin/stdout para la extension |
| `src/dataset_tools.py` | utilidad historica (no usada en el flujo actual) |
| `notebooks/entrenamiento.ipynb` | entrega formal: arquitectura, perdida, ejemplos |
| `vscode-extension/extension.js` | F7: cliente Node que spawn-ea F6 |
| `vscode-extension/package.json` | F7: comandos, keybinding, config `rnnC.*` |
| `vscode-extension/test_e2e.js` | smoke test del protocolo JSON-line (sin VS Code) |
| `vscode-extension/prueba.c` | archivo .c de prueba para F5 |
| `scripts/train.fish` | orquestador curate → preprocess → train |
| `scripts/predict.fish` | wrapper de `predict.py` |
| `scripts/curate.fish` | corre `curate_per_category.py` |
| `models/rnn_v1.keras` | modelo entrenado (17 854 params, 244 KB) |
| `models/rnn_v1.meta.json` | vocab, block_size, params, paths |
| `models/rnn_v1.history.json` | curva de perdida por epoca |

## 4. Como se almacenan los datos

- `dataset/clean/*.c` (8 archivos, 1 por categoria: `arrays.c`,
  `io_system.c`, `lists.c`, `math.c`, `misc.c`, `sorting.c`, `strings.c`,
  `trees.c`). Entrada del curador.
- `dataset/curated/funciones.c` — concatenado de las 640 funciones
  curadas (8 categorias × 80), separadas por linea en blanco. **Es el
  input real a `preprocess.py`**. Total: **278 217 chars**.
- `dataset/curated/<categoria>.c` — los 80 archivos por categoria
  individualmente (util para inspeccion).
- `dataset/curated/REPORTE.md` — metricas del curado: funciones
  escaneadas (704 212), seleccionadas (640), razones de rechazo
  agregadas.
- `dataset/sample/funciones.c` — salida alternativa de `sample_clean.py`
  (150 funciones del corpus crudo de 466 MB). **No se usa en el flujo
  real**; se conserva por valor historico.
- `dataset/processed/X.npy`, `Y.npy` — `int64` de shape
  `(278 185, 32)`. `X[i] = seq[i:i+32]`, `Y[i] = seq[i+1:i+33]`.
- `dataset/processed/meta.json` — `{chars, block_size, vocab_size,
  corpus_chars, num_windows, dataset_path}`.
- `models/rnn_v1.keras` — el modelo Keras entero (arquitectura + pesos).
  Se carga con `tf.keras.models.load_model(path)`.
- `models/rnn_v1.meta.json` — copia del `processed/meta.json` + `params`
  del entrenamiento + paths absolutos al `.keras` y al `history.json`.
- `models/rnn_v1.history.json` — array `loss` por epoca + `initial_loss`,
  `final_loss`, `epochs`, `batch_size`, `embed_dim`, `hidden`, `lr`.

**Punto clave para el profesor:** el `chars` de `meta.json` **define** el
mapeo `char ↔ id` y se aprende implicitamente desde el corpus. La
extension y los servers usan el mismo `meta.json` que el training — si
cambia el corpus, hay que reentrenar. La "fuente de verdad" para
`stoi`/`itos` es el orden alfabetico de `sorted(set(text))` en
`preprocess.py:37`.

## 5. Arquitectura del modelo

`build_model()` (`src/train.py:42`) implementa la **RNN Vanilla**
exigida por la actividad:

```python
Sequential([
    Input(shape=(None,)),                                  # block_size = 32
    Embedding(vocab_size=95, embed_dim=48),                # 4 560 params
    SimpleRNN(hidden=64, activation="tanh",
              return_sequences=True),                      # 7 232 params
    TimeDistributed(Dense(vocab_size=95)),                 # 6 155 params
])
# Total: 17 947 params entrenables
```

- **Embedding**: `95 × 48 = 4 560`. Convierte cada id de caracter en un
  vector denso de 48D. La matriz se aprende durante el entrenamiento.
- **SimpleRNN**: `64` unidades, `tanh`, `return_sequences=True`. **No es
  LSTM ni GRU** — la actividad exige RNN Vanilla literal. La recurrencia
  es `h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)`. Params:
  `(48 + 64 + 1) × 64 = 7 232`.
- **TimeDistributed(Dense(95))**: produce un logit por **cada paso
  temporal**, no solo el último. Necesario porque queremos predecir el
  siguiente char en cada posición de la ventana.
- **Perdida**: `SparseCategoricalCrossentropy(from_logits=True)` — las
  etiquetas son enteros (no one-hot).
- **Optimizador**: `Adam(learning_rate=1e-3)`.

> **Por que `return_sequences=True`?** Sin esto, la RNN solo emite el
> estado final (`h_T`); con esto, emite un estado por cada paso. Como
> nuestra ventana tiene 32 posiciones y en cada una queremos predecir el
> caracter siguiente, necesitamos los 32 logits — uno por paso.
>
> **Por que char-level?** Vocab de 95 (< 100) cabe en una `Embedding`
> pequena, entrena rapido en CPU y no requiere tokenizer externo
> (BPE/SentencePiece). Ademas reproduce la mecanica del ejemplo del
> profesor ("Autocompletado con RNN vanilla en Keras").

### Numeros del entrenamiento (seed=42)

| Metrica | Valor |
|---|---|
| Vocabulario | 95 chars unicos |
| Corpus | 278 217 chars (~272 KB) |
| Ventanas | 278 185 (sin sub-muestrear) |
| Epochs | 80 |
| Perdida inicial | 3.02 |
| Perdida final | 1.35 |
| Tiempo (CPU, 4 cores) | ~5 min |
| Parametros entrenables | 17 947 |

> La entropia inicial `ln(95) ≈ 4.55` representa el peor caso (prediccion
> uniforme). Una perdida de 1.35 significa que el modelo es ~`e^(4.55 -
> 1.35) ≈ 24` veces mas preciso que azar.

## 6. Inferencia — `RNNModel` y `complete()`

Codigo clave (`src/predict.py:48-91`):

```python
def _encode(prefix, stoi, block_size):
    ids = [stoi[c] for c in prefix if c in stoi]
    if len(ids) >= block_size:
        ids = ids[-block_size:]               # truncate por la derecha
    else:
        ids = [0] * (block_size - len(ids)) + ids  # left-pad con pad_id=0
    return np.array(ids, dtype=np.int64).reshape(1, block_size)


def _sample(logits, temperature):
    if temperature <= 0.0:
        return int(np.argmax(logits))         # greedy
    logits = logits / temperature
    logits = logits - logits.max()            # softmax numericamente estable
    probs = np.exp(logits) / np.exp(logits).sum()
    return int(np.random.choice(len(probs), p=probs))


def complete(rnn, prefix, max_new=60, temperature=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    state = _encode(prefix, rnn.stoi, rnn.block_size)
    out = []
    for _ in range(max_new):
        logits = rnn.model.predict(state, verbose=0)[0, -1]
        idx = _sample(logits, temperature)
        out.append(rnn.itos[idx])
        state = np.concatenate([state[:, 1:], [[idx]]], axis=1)
    return "".join(out)
```

Decisiones:

- **Filtrar `prefix` por `stoi`**: si el usuario escribe un caracter que
  no esta en el vocab (no deberia pasar — los `.c` son ASCII), se ignora
  silenciosamente.
- **Left-pad con `pad_id=0`**: el modelo no fue entrenado con padding,
  asi que el prefijo corto igual se procesa sin reentrenar. La RNN
  emite su primer logit despues de los 32 pasos; el prefijo corto se
  "rellena" hacia la izquierda.
- **Truncar por la derecha** si el prefijo es mas largo que `block_size`
  (=32): solo nos importan los ultimos 32 chars (la RNN no ve mas alla).
- **`temperature=0` → greedy**: `argmax` puro, sin azar. Util para
  `suggest` (top-N deterministico) y para debugging.
- **`np.random.seed(seed)`** arriba de todo: fija la secuencia exacta de
  muestreo, asi dos ejecuciones con el mismo seed dan el mismo output.
- **`state = concat(state[:, 1:], [[idx]])`**: slide window manual.
  Despues de cada char generado, se corre la ventana un paso a la
  izquierda y se append-ea el nuevo char al final.

## 7. Servidor stdio (F6) — protocolo JSON-line

`src/server_stdio.py` (105 lineas) expone el modelo al editor mediante
**JSON-line sobre stdin/stdout**. Es el patron que usan Copilot, Tabnine,
gopls, etc.

### Protocolo

```text
Request  : {"id": <int|str>, "method": "complete"|"suggest"|"ping", ...}
Response : {"id": ..., "ok": true|false, ...}
```

### Metodos

- `complete` — `{"prefix": str, "max_new"?: int=60, "temperature"?: float=0.5, "seed"?: int}`
  → `{"text": "<continuacion>"}`. Llama a `complete()` y devuelve
  solo la continuacion (sin el prefijo).
- `suggest` — `{"prefix": str, "n"?: int=5}` → `{"items": ["<char1>",
  "<char2>", ...]}`. Devuelve los **top-N chars siguientes mas
  probables** (greedy, no muestrea). Util para un dropdown de UI.
- `ping` — `{}` → `{"pong": true}`. Util para health-check desde la
  extension.

### Loop del server (lineas 73-101)

```python
def main():
    rnn = _load_rnn()
    for raw in sys.stdin:
        line = raw.strip()
        if not line: continue
        try:
            req = json.loads(line)
            method = req.get("method")
            if method == "complete":
                payload = handle_complete(rnn, req)
            elif method == "suggest":
                payload = handle_suggest(rnn, req)
            elif method == "ping":
                payload = {"pong": True}
            else:
                raise ValueError(f"unknown method: {method!r}")
            resp = {"id": req.get("id"), "ok": True, **payload}
        except Exception as exc:
            resp = {"id": ..., "ok": False, "error": str(exc),
                    "trace": traceback.format_exc(limit=2)}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
```

- **El modelo se carga UNA vez** al inicio (`rnn = _load_rnn()`) y se
  reusa por cada request — sin re-cargar pesos en cada llamada.
- **Errores no rompen el server**: el `try/except` envuelve cada linea,
  responde `ok: false` con trace, y el loop sigue escuchando.
- **Cada respuesta termina en `\n` + `flush()`** — el cliente sabe
  cuando termina cada mensaje.

## 8. Extension VSCode (F7)

`vscode-extension/extension.js` (193 lineas) es un cliente Node que
hace `spawn` de `python src/server_stdio.py` y le manda requests
cuando el usuario aprieta **Ctrl+Shift+Space** en un archivo `.c`.

### Comandos y keybinding (`package.json`)

```json
"commands": [
  { "command": "rnnC.complete", "title": "RNN: completar linea" },
  { "command": "rnnC.suggest",  "title": "RNN: elegir siguiente caracter" },
  { "command": "rnnC.status",   "title": "RNN: estado del servidor" }
],
"keybindings": [
  { "command": "rnnC.complete", "key": "ctrl+shift+space",
    "when": "editorTextFocus && editorLangId == c" }
]
```

### Configuracion por workspace (`settings.json`)

```json
{
  "rnnC.pythonPath":   "/home/jojo/develop/academic/IA/.venv/bin/python",
  "rnnC.serverScript": "src/server_stdio.py",
  "rnnC.modelPath":    "models/rnn_v1.keras",
  "rnnC.maxNew":       60,
  "rnnC.temperature":  0.4
}
```

> Los paths son **relativos al primer workspace folder** abierto. Si
> abris la carpeta `3-RNN/`, las resuelve como
> `3-RNN/src/server_stdio.py` y `3-RNN/models/rnn_v1.keras`.

### Flujo de un `complete` (extension.js:123-138)

```javascript
async function cmdComplete() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;
  const cfg = vscode.workspace.getConfiguration("rnnC");
  const prefix = linePrefix(editor);   // texto desde inicio de linea hasta cursor
  try {
    const r = await request("complete", {
      prefix,
      max_new: cfg.get("maxNew", 60),
      temperature: cfg.get("temperature", 0.4),
    });
    await editor.edit((b) => b.insert(editor.selection.active, r.text));
  } catch (err) {
    vscode.window.showErrorMessage(`RNN complete failed: ${err.message}`);
  }
}
```

- `linePrefix(editor)`: toma el texto desde el **inicio de la linea**
  hasta el cursor (no todo el archivo). Esto le da al modelo un
  contexto coherente: una linea de C.
- `request()` (lineas 94-114): correlaciona `id` ↔ `Promise` en un
  `Map`, manda una linea JSON por stdin, y resuelve cuando llega la
  respuesta con ese `id`. Timeout default 30 s.
- `editor.edit((b) => b.insert(editor.selection.active, r.text))`:
  inserta la continuacion **en el cursor** (no reemplaza nada).

### F5 → Extension Development Host

`vscode-extension/.vscode/launch.json`:

```json
{
  "type": "extensionHost",
  "request": "launch",
  "args": ["--extensionDevelopmentPath=${workspaceFolder}",
           "${workspaceFolder}"]
}
```

Pasos para probar en vivo:

1. `code vscode-extension/` — abre VSCode en la carpeta de la extension.
2. **F5** — VSCode arranca una **Extension Development Host** (ventana
   nueva con la extension cargada).
3. En esa ventana, abrir `vscode-extension/prueba.c` (o cualquier `.c`).
4. Posicionar el cursor en una linea y **Ctrl+Shift+Space** → la RNN
   inserta la continuacion directamente en el editor.

### Smoke test sin VS Code

`test_e2e.js` re-implementa el cliente Node contra el mismo
`server_stdio.py` (no usa `vscode.workspace`). Hace `ping`, `suggest`,
y dos `complete`. Si pasa, la extension va a funcionar. Se corre con:

```bash
node 3-RNN/vscode-extension/test_e2e.js
```

## 9. Preguntas tipo examen (respuesta corta)

**Dataset y preprocesamiento**

1. *Cuantas funciones tiene el dataset final?* 640 (8 categorias × 80),
   producto de `curate_per_category.py` desde `dataset/clean/`.
2. *Cuantos chars tiene el corpus?* 278 217 chars concatenados en
   `dataset/curated/funciones.c`.
3. *Que tan grandes son las ventanas?* `block_size = 32` chars.
4. *Cuantas ventanas se generan?* `corpus_chars - block_size = 278 185`.
5. *Que split se usa?* No hay split — es modelado de lenguaje (no
   clasificacion). El modelo ve **todas** las ventanas mezcladas, sin
   train/test separados.
6. *Que tipo de etiqueta tiene cada ventana?* El siguiente caracter
   en cada posicion. `Y[i] = seq[i+1 : i+33]` (X shifted by 1).
7. *Cual es el vocabulario?* 95 chars unicos (los que aparecen en el
   corpus), ordenados con `sorted(set(text))`. Mapeo en `stoi`/`itos`.

**Arquitectura del modelo**

8. *Que tipo de red es?* Una **RNN Vanilla** (`SimpleRNN`) secuencial
   de Keras. No es LSTM, no es GRU — la actividad lo exige literal.
9. *Cuantas capas tiene la arquitectura?* 3: `Embedding(95, 48)`,
   `SimpleRNN(64, return_sequences=True)`, `TimeDistributed(Dense(95))`.
10. *Por que `return_sequences=True`?* Para emitir un logit en **cada**
    paso temporal. Sin esto solo tendriamos el estado final y no
    podriamos predecir el siguiente char en cada posicion.
11. *Que hace `TimeDistributed(Dense(vocab))`?* Aplica la misma `Dense`
    a cada paso temporal. Es la forma idiomática de Keras de decir
    "misma capa lineal en cada step".
12. *Que funcion de activacion tiene la RNN?* `tanh` (default de
    `SimpleRNN`). Es la activacion que permite recurrentes positivos
    y negativos.
13. *Que funcion de perdida se usa?* `SparseCategoricalCrossentropy(
    from_logits=True)` — etiquetas enteras, salida cruda (logits).
14. *Que optimizador se usa?* `Adam(learning_rate=1e-3)`.
15. *Cuantos parametros entrenables tiene?* 17 947. Embedding: 95×48 =
    4 560. SimpleRNN: (48+64+1)×64 = 7 232. Dense: 64×95+95 = 6 155.
16. *Por que no LSTM/GRU?* La actividad 3 exige **RNN Vanilla**
    explicitamente. La diferencia: SimpleRNN no tiene compuertas, asi
    que `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`. Limita la
    "memoria" a ~`block_size` (32) chars.
17. *Por que char-level y no token-level?* Vocab de 95 (< 100) cabe en
    una `Embedding` pequena, no requiere tokenizer externo (BPE), y
    entrena rapido en CPU. Reproduce la mecanica del ejemplo del
    profesor.

**Entrenamiento**

18. *Cuantas epocas se entrenan?* 80 (default en `train.fish`).
19. *Cual es la perdida inicial y final?* 3.02 → 1.35. La inicial ya
    esta debajo de `ln(95) ≈ 4.55` (azar) gracias a la inicializacion
    de Keras.
20. *Cuanto tarda el entrenamiento?* ~5 min en CPU (4 cores, sin GPU).
21. *Que se guarda despues de entrenar?* `rnn_v1.keras` (modelo),
    `rnn_v1.meta.json` (vocab + params + paths), `rnn_v1.history.json`
    (curva de perdida).
22. *Se usa GPU?* No — TF 2.16 detecta la ausencia y degrada a CPU sin
    fallar. Si se agrega CUDA, el codigo se mueve solo.

**Inferencia y sampling**

23. *Que hace `_encode(prefix, stoi, block_size)`?* Convierte el
    prefijo a una ventana `(1, 32)` int64. Si el prefijo es corto,
    left-pad con `pad_id=0`. Si es largo, trunca por la derecha
    (ultimos 32 chars).
24. *Que pasa si el prefijo tiene un char fuera del vocab?* Se ignora
    silenciosamente (`if c in stoi`).
25. *Que hace `_sample(logits, temperature)`?* Si `temperature <= 0`,
    devuelve `argmax` (greedy). Si no, divide logits por temperatura,
    aplica softmax numericamente estable (`- logits.max()` antes de
    `exp`) y sample-ea con `np.random.choice`.
26. *Por que se hace `state[:, 1:]` y append?* Para correr la ventana
    un paso a la izquierda despues de cada prediccion (slide window
    manual).
27. *Por que `verbose=0` en `model.predict`?* Para no spamear la
    terminal con el progress bar de Keras (estamos en un loop de
    hasta 400 iteraciones).
28. *Que hace `seed` en `complete()`?* `np.random.seed(seed)` arriba
    de todo, fija la secuencia exacta de muestreo. Dos ejecuciones
    con el mismo seed dan el mismo output.
29. *Que efecto tiene la temperatura?* `T=0` es determinista (greedy).
    `T` baja (0.1-0.5) repite patrones del corpus. `T` alta (1+) mete
    mas variedad y ruido.

**Servidor stdio (F6)**

30. *Que protocolo usa `server_stdio.py`?* **JSON-line** sobre
    `stdin`/`stdout`. Una linea = un mensaje JSON. Termina cada
    respuesta con `\n` + `flush()`.
31. *Que metodos expone?* `complete` (genera continuacion),
    `suggest` (top-N chars siguientes), `ping` (health-check).
32. *El modelo se recarga en cada request?* No. Se carga **una vez**
    al inicio del proceso (`rnn = _load_rnn()`) y se reusa.
33. *Que pasa si llega un request con un metodo desconocido?* El
    server responde `{"ok": false, "error": "unknown method: '...'"}`
    y sigue escuchando.
34. *Que pasa si el server crashea?* La extension captura el `exit`
    code, rechaza todas las promesas pendientes con
    `"server exited (code=...)"`, y deja la extension sin server
    hasta que el usuario vuelva a invocar un comando.

**Extension VSCode (F7)**

35. *Que comando/keybinding inserta una completacion?*
    `RNN: completar linea` con **Ctrl+Shift+Space** (solo en archivos
    `.c`, `when: editorTextFocus && editorLangId == c`).
36. *Que hace la extension al activarse?* Hace `spawn` del
    `python src/server_stdio.py` (configurable via `rnnC.pythonPath` y
    `rnnC.serverScript`). El subproceso vive hasta que se cierra la
    Extension Development Host.
37. *Que lee `linePrefix(editor)`?* El texto desde el **inicio de la
    linea** hasta el cursor — no todo el archivo. Asi el modelo
    recibe un contexto coherente (una linea de C).
38. *Que pasa si VSCode no encuentra el Python del venv?*
    `spawn()` falla, el `cmdComplete` rechaza la promesa, y la UI
    muestra un error message: "RNN complete failed: ENOENT …".
    Solucion: configurar `rnnC.pythonPath` con la ruta absoluta al
    venv unificado (`/home/jojo/develop/academic/IA/.venv/bin/python`).
39. *Que es la Extension Development Host?* Es la **ventana nueva** de
    VSCode que se abre al apretar F5 con una configuracion de tipo
    `extensionHost`. Carga la extension desde
    `--extensionDevelopmentPath=${workspaceFolder}` sin empaquetarla.
    Es el modo "dev" de cualquier extension.
40. *Cual es el primer paso para probar la extension?* Abrir
    `code vscode-extension/`, **F5**, esperar 5-10 s a que el server
    cargue el modelo, abrir `prueba.c` en la Extension Development
    Host, **Ctrl+Shift+Space** en una linea.
41. *Que hace `Ctrl+Shift+P → RNN: elegir siguiente caracter`?* Llama
    a `suggest` en lugar de `complete`, y muestra un **quick-pick**
    con los top-5 chars siguientes. El usuario elige uno y se inserta
    en el cursor.
42. *Que hace `Ctrl+Shift+P → RNN: estado del servidor`?* Llama a
    `ping` y muestra un message arriba: "RNN server alive (pong=
    true). Use Ctrl+Shift+Space in a C file."
43. *Por que el primer Ctrl+Shift+Space tarda 10 s?* Es el **warmup
    de TensorFlow**: la primera invocacion compila el grafo y mueve
    tensores a memoria. Las siguientes reutilizan el modelo ya
    cargado y son <1 s.
44. *Que relacion tiene la extension con `test_e2e.js`?* `test_e2e.js`
    re-implementa el cliente Node de `extension.js` (mismo
    `spawn` + JSON-line) pero **sin** `vscode.workspace`. Es un smoke
    test que corre `ping`, `suggest`, y dos `complete` desde la
    terminal. Si pasa, la extension va a funcionar.
45. *Cual es la diferencia entre `dev mode` y `prod mode`?* Dev: abrir
    `vscode-extension/` y F5 (lee el codigo directo, hot-reload). Prod:
    `npm install -g @vscode/vsce && vsce package` produce un `.vsix`,
    y `code --install-extension` lo registra globalmente.

**Modelo → editor (integracion F1-F7)**

46. *Cual es el flujo completo de un Ctrl+Shift+Space?*
    1) Editor captura keystroke.
    2) `cmdComplete` calcula `prefix = linePrefix(editor)`.
    3) `request("complete", {...})` manda JSON al subproceso.
    4) `server_stdio.handle_complete` corre `complete()` (hasta 60
       chars autoregresivos).
    5) Server responde JSON con `{"ok": true, "text": "..."}`.
    6) Extension hace `editor.edit((b) => b.insert(..., r.text))`.
    7) VSCode pinta la nueva continuacion.
47. *Por que JSON-line y no gRPC o HTTP?* Es el patron mas simple
    para procesos locales: 0 dependencias, 0 puertos, debuggeable con
    `echo + pipe`. La latencia es < 1 ms por mensaje.
48. *Por que se identifica cada request con un `id`?* Para correlacionar
    request ↔ response. El server puede atender varios comandos
    concurrentes (la extension manda `suggest` mientras espera
    `complete`) y necesita saber cual respuesta va a cual Promise.
