# Proyecto 3 - RNN Vanilla para Autocompletado de Codigo C

Entrenar un modelo de lenguaje a nivel de **caracter** con una **RNN
Vanilla** (`SimpleRNN`) en Keras/TensorFlow, capaz de autocompletar codigo
C, y exponerlo a un editor (VS Code) mediante un servidor local.

> Actividad: *Asistente de Codigo Personalizado con Redes Recurrentes*
> (Prof. Ealcaraz85, Proyectos 2026 Enero Junio - Actividad 3).
> Requisito explicito: **RNN Vanilla** en Keras, **60+ funciones en C**
> con estilo consistente, **integracion con editor** via API local.

## Stack

| Componente | Version |
|---|---|
| Python | 3.11.9 (pyenv) |
| TensorFlow / Keras | 2.16.1 |
| numpy | 1.26.x |
| FastAPI | 0.136.x |
| uvicorn | 0.48.x |
| Editor | VS Code 1.85+ |

> TensorFlow 2.16.1 todavia no soporta Python 3.14. Por eso el proyecto
> fija `3.11.9` con `pyenv local`. Ver `docs/ENTORNO.md` para el detalle.

## Estructura

```text
3-RNN/
├── .python-version              # pyenv 3.11.9
├── .venv/                       # entorno virtual
├── dataset/
│   ├── funciones.c              # corpus crudo de entrada (466 MB)
│   ├── sample/                  # 150 funciones "limpias" (F1)
│   │   ├── funciones.c
│   │   └── REPORTE.md
│   └── processed/               # ventanas char-level (F2)
│       ├── X.npy, Y.npy
│       └── meta.json
├── models/
│   ├── rnn_v1.keras             # ~244 KB, ~17 K parametros
│   ├── rnn_v1.meta.json         # vocab + block_size
│   └── rnn_v1.history.json      # curva de perdida
├── notebooks/
│   └── entrenamiento.ipynb      # entrega formal, ejecutado y con outputs
├── src/
│   ├── sample_clean.py          # F1: muestreo del corpus
│   ├── preprocess.py            # F2: ventanas char-level
│   ├── train.py                 # F3: arquitectura + entrenamiento
│   ├── predict.py               # F4: complete(prefix, max_new, temperature)
│   ├── api.py                   # F5: FastAPI
│   ├── server_stdio.py          # F6: JSON-line para VS Code
│   ├── dataset_tools.py         # utilidad original (no usada en este flujo)
│   └── __init__.py
├── vscode-extension/            # F7: extension VS Code
│   ├── package.json
│   ├── extension.js
│   └── README.md
├── docs/
│   ├── INICIO.md
│   ├── ENTORNO.md
│   ├── ESTRUCTURA.md
│   ├── DATASET.md
│   └── DATASET_PROCESO.md
├── requirements.txt
└── README.md
```

## Flujo end-to-end

```bash
# 0) entorno (solo la primera vez)
cd 3-RNN
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) muestreo del corpus (de 466 MB a 70 KB de funciones limpias)
python src/sample_clean.py

# 2) preprocesamiento char-level
python src/preprocess.py

# 3) entrenamiento (~3 min en CPU, 4 cores)
python src/train.py --epochs 80 --max-windows 20000

# 4) probar en CLI
python src/predict.py --prompt "int sum" --max-new 60

# 5a) servidor FastAPI (para pruebas HTTP)
uvicorn src.api:app --port 8000
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt":"int sum","max_new":40,"temperature":0.4}'

# 5b) servidor stdio (para la extension VS Code)
echo '{"id":1,"method":"complete","prefix":"int sum","max_new":40}' | \
     python -m src.server_stdio
```

## Entregables de la actividad

1. **Notebook `notebooks/entrenamiento.ipynb`** - ejecutado, con la
   arquitectura, perdida, graficas y ejemplos de generacion.
2. **Dataset `dataset/sample/funciones.c`** - 150 funciones en C
   seleccionadas de un corpus mas grande con `src/sample_clean.py`.
3. **Modelo en `models/`** - `rnn_v1.keras` + `meta.json`.
4. **Servidor en `src/api.py` y `src/server_stdio.py`** - dos formas
   de consumir el modelo (HTTP y JSON-line).
5. **Extension VS Code en `vscode-extension/`** - instala con F5 y
   usa Ctrl+Shift+Space en cualquier archivo `.c`.

## Como usar en VS Code

### Modo desarrollo (F5)

1. Abrir la carpeta de la extension en VS Code:
   ```bash
   code vscode-extension/
   ```
2. Presionar **F5** → se abre una ventana "Extension Development Host"
   con la extension cargada.
3. En esa ventana, abrir un archivo `.c` y presionar
   **Ctrl+Shift+Space** → inserta la continuacion en el cursor.

### Instalacion permanente (en tu VS Code real)

```bash
cd vscode-extension
npm install -g @vscode/vsce
vsce package
code --install-extension rnn-c-autocomplete-0.1.0.vsix
```

### Probar el servidor sin VS Code

```bash
# Smoke test del JSON-line protocol (lo que usa la extension)
node vscode-extension/test_e2e.js

# O mas simple, con echo + python directo
echo '{"id":1,"method":"complete","prefix":"int sum","max_new":20}' | \
     python src/server_stdio.py models/rnn_v1.keras
```

### Configuracion por proyecto (settings.json del workspace)

```json
{
  "rnnC.pythonPath": "python",
  "rnnC.serverScript": "src/server_stdio.py",
  "rnnC.modelPath": "models/rnn_v1.keras",
  "rnnC.maxNew": 60,
  "rnnC.temperature": 0.4
}
```

Los defaults ya apuntan a `models/rnn_v1.keras` y `src/server_stdio.py`
**relativos a la raiz del proyecto 3-RNN** (la extension usa el primer
workspace folder como raiz). Si tu estructura cambia, ajustar aca.

## Arquitectura del modelo

```python
Sequential([
    Input(shape=(BLOCK_SIZE,)),                           # 32 chars
    Embedding(VOCAB_SIZE, 48),                            # 48-d embedding
    SimpleRNN(64, activation="tanh", return_sequences=True),
    TimeDistributed(Dense(VOCAB_SIZE)),                   # logits del sig. char
])
```

- **RNN Vanilla**: `SimpleRNN` (no LSTM, no GRU). Cumple el requisito
  literal de la actividad.
- **`return_sequences=True` + `TimeDistributed(Dense)`**: produce un
  logit por cada paso temporal, igual que en el ejemplo del profesor.
- **Perdida**: `SparseCategoricalCrossentropy(from_logits=True)`.
- **Optimizador**: Adam(1e-3).

## Numeros del entrenamiento (reproducibles con seed=42)

| Metrica | Valor |
|---|---|
| Vocabulario | 94 chars unicos |
| Corpus | 69 874 chars (~70 KB) |
| Ventanas | 69 842 (sub-muestreadas a 20 000 para velocidad) |
| Epochs | 80 |
| Perdida inicial | 3.07 |
| Perdida final | 1.06 |
| Tiempo (CPU, 4 cores) | ~5 min |
| Parametros | 17 854 |

> La entropia inicial `ln(94) ~ 4.54` representa el peor caso (prediccion
> uniforme sobre todo el vocabulario). Una perdida de 1.06 significa
> que el modelo es ~4x mas preciso que azar a la hora de elegir el
> siguiente caracter.

## Como responder las preguntas del profesor

| Pregunta | Resuesta en |
|---|---|
| *Por que SimpleRNN y no LSTM/GRU?* | La actividad 3 exige RNN Vanilla explicitamente. La diferencia es que SimpleRNN no tiene compuertas: `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`. Esto limita la memoria a ~32 chars (nuestra `BLOCK_SIZE`). |
| *Por que char-level?* | Vocab de 94 (< 100) cabe en una `Embedding` pequena; entrena rapido en CPU y no requiere tokenizer externo. |
| *Por que `return_sequences=True`?* | Necesitamos un logit en **cada** paso para comparar con el caracter **siguiente** en cada posicion. Sin esto solo tendriamos el ultimo estado. |
| *Como se conecta con el editor?* | VS Code hace `spawn` de `python -m src.server_stdio` y envia lineas JSON. El servidor mantiene el modelo en memoria y responde. |
| *Por que no LM Studio / Ollama?* | Esas herramientas son del Proyecto 4 (Fine-Tuning / RAG con LLMs). Proyecto 3 es RNN Vanilla. |
| *Por que solo 150 funciones?* | La actividad pide "por lo menos 60". 150 da variedad sin saturar la RNN; mas datos + modelo mas pequeno = mismas epocas necesarias. |
| *Como mide la calidad?* | Loss por caracter (cross-entropy). La calidad "humana" se ve generando ejemplos - el modelo aprende sintaxis basica de C pero no es ejecutable. |

## Limitaciones y extensiones

- **Memoria corta**: `BLOCK_SIZE=32` significa que el modelo no ve
  contexto mas alla de 32 chars. Una linea larga pierde el inicio.
- **Sin gramatica**: el modelo no valida parentesis ni llaves.
  Sube a 1.0 de loss = todavia ~37% de error por caracter.
- **Sin GPU**: ~5 min de entrenamiento en CPU. Con GPU seria <30 s.
- **Extensiones naturales**:
  - Sustituir `SimpleRNN` por `GRU` o `LSTM` (mas memoria, mejor
    coherencia).
  - Migrar a token-level con BPE (vocab mas rico).
  - Anadir beam search en vez de muestreo por temperatura.

## Ver tambien

- `docs/INICIO.md` - guia rapida de uso.
- `docs/ENTORNO.md` - por que Python 3.11.9 y como recrear el venv.
- `docs/DATASET.md` - como se construye el corpus de 150 funciones.
- `docs/ESTRUCTURA.md` - descripcion detallada de carpetas.
- `docs/DATASET_PROCESO.md` - flujo historico del dataset_tools.py
  original (no usado en este flujo; conservado por referencia).
