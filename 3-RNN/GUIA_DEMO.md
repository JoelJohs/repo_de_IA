# Guia de demo - Proyecto 3 (RNN Vanilla)

Como mostrar el proyecto funcionando al profesor, en 4 modalidades
distintas. Cada una toma entre 1 y 5 minutos. Tiempos medidos en CPU
(4 cores, 17 GB RAM, sin GPU).

> Antes de la demo: corre los pasos de **Setup** (1 sola vez) y
> **Generar todo** (1 sola vez por maquina). Despues podes repetir
> cualquier demo las veces que quieras.

---

## Pre-requisitos

- Linux (Arch en este caso), 4 cores, 17 GB RAM, ~5 GB libres en disco.
- `pyenv` instalado con Python 3.11.9 disponible
  (`pyenv versions` debe listar `3.11.9`).
- `code` (VS Code) instalado.
- `node` instalado (para `test_e2e.js`).

> Si no cumplis algo, mira la seccion "Troubleshooting" al final.

---

## Setup (solo la primera vez, ~3 min)

El venv esta **unificado en la raiz del repo** (`IA/.venv`) y ya tiene
todas las dependencias de los 3 proyectos instaladas. Solo hay que
activarlo.

```bash
cd /home/jojo/develop/academic/IA/3-RNN

# 1) El venv unificado vive en la raiz del repo
cd ../..
source .venv/bin/activate.fish

# 2) Verificar
python --version
# -> Python 3.11.9
python -c "import tensorflow as tf; print('TF', tf.__version__)"
# -> TF 2.16.1
```

**Que decir mientras se hace esto:**

- "El sistema tiene Python 3.14 pero TensorFlow 2.16 no lo soporta
  todavia, asi que fijamos 3.11.9 con pyenv local al repo."
- "El venv unificado aísla las dependencias — no toca el Python del
  sistema. Los 3 proyectos comparten el mismo venv."

---

## Generar todo desde cero (~5 min, 3 comandos)

Estos 3 comandos reproducen todo el pipeline. Si los modelos
(`models/rnn_v1.keras`) ya existen, podés saltarlos — los pasos
siguientes los reusan.

```bash
# Paso 1: muestreo (de 466 MB a 70 KB de funciones limpias)
python src/sample_clean.py
```

**Salida esperada** (ultimas lineas):

```
Reading first 20,000,000 bytes of dataset/funciones.c ...
Loaded 19,996,114 chars.
Found 26,439 candidate signatures.
Scanned, kept 19,565 candidates. Rejected: {'has_main': 4939, 'length=...': ...}
Wrote 150 functions to dataset/sample/funciones.c (69,874 bytes)
Reporte en dataset/sample/REPORTE.md
```

**Que decir:** "De 26 439 funciones candidatas, ~19 500 pasan los
filtros ASCII/llaves/main. Nos quedamos con las 150 que mejor
'puntúan' en control flow (`if`/`for`/`return`)."

```bash
# Paso 2: preprocesamiento (vocabulario + ventanas)
python src/preprocess.py
```

**Salida esperada:**

```
vocab_size=94 | corpus_chars=69,874 | num_windows=69,842
X.shape=(69842, 32) Y.shape=(69842, 32) dtype=int64
Saved to dataset/processed
```

**Que decir:** "Vocabulario = 94 chars unicos (sumar digitos,
letras, simbolos C). Generamos 69 842 ventanas de 32 chars cada
una, donde cada ventana X predice la ventana Y shifted-by-1 (el
siguiente caracter en cada posicion)."

```bash
# Paso 3: entrenamiento (80 epocas, ~5 min)
python src/train.py --epochs 80 --max-windows 20000
```

**Salida esperada** (resumen):

```
X=(20000, 32) Y=(20000, 32) | vocab=94 | block_size=32
Model: "sequential"
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)    ┃ Output Shape       ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding       │ (None, None, 48)   │         4,512 │
│ simple_rnn      │ (None, None, 64)   │         7,232 │
│ time_distributed│ (None, None, 94)   │         6,110 │
└─────────────────┴────────────────────┴───────────────┘
 Total params: 17,854 (69.74 KB)
  epoch    1 | loss=3.0657
  epoch   10 | loss=1.3719
  ...
  epoch   80 | loss=1.0606
loss: 3.0657 -> 1.0606
Saved model to models/rnn_v1.keras
```

**Que decir:**

- "Embedding 94x48 + SimpleRNN 64 + TimeDistributed Dense 94 = 17 854
  parámetros, 70 KB total."
- "Loss inicial ~3.07 (mejor que azar=4.54), final ~1.06. 5 minutos
  de CPU."

---

## Demo 1 — CLI directa (~1 min)

La forma más rápida de mostrar el modelo funcionando.

```bash
python src/predict.py --prompt "int sum" --max-new 25 --temperature 0.4 --seed 42
```

**Salida esperada:**

```
int sum[mid] < nums[0] == 0)
```

**Probar con otros prompts** (cada uno tarda ~1 s):

```bash
python src/predict.py --prompt "void print" --max-new 25 --temperature 0.4
python src/predict.py --prompt "for (int"   --max-new 25 --temperature 0.4
python src/predict.py --prompt "if (a"      --max-new 20 --temperature 0.4
```

**Que decir:**

- "El prompt es lo que el usuario escribio hasta el cursor. El
  modelo genera los siguientes N chars autoregresivamente."
- "Temperatura 0.4 es relativamente determinista — mas baja
  repite patrones, mas alta mete mas variedad."

**Si querés mostrar el efecto de la temperatura:**

```bash
for t in 0.0 0.4 0.8 1.2; do
  echo "--- T=$t ---"
  python src/predict.py --prompt "int fibonacci" --max-new 20 --temperature $t --seed 7
done
```

---

## Demo 2 — FastAPI + curl (~2 min)

Muestra el modelo expuesto como servicio HTTP. Util para impresionar
con `curl` o Postman.

**Terminal A** (arrancar el server):

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

**Salida esperada (en la terminal A):**

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Terminal B** (probar el server):

```bash
# Health check
curl http://127.0.0.1:8000/health
# -> {"status":"ok"}

# Prediccion
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt":"int sum","max_new":30,"temperature":0.4}'
# -> {"completion":"[mid] < nums[0] == 0)\n  return '", "full":"int sum[mid] < nums[0] == 0)\n  return '"}

# Sugerencias (top-5 chars siguientes, deterministic)
curl -X POST http://127.0.0.1:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"prefix":"int ","n":5}'
# -> {"items":["b","a","s","i","c"]}
```

**Que decir:**

- "El server arranca en ~5 s (carga el modelo en memoria)."
- "El modelo es un singleton: se carga una vez y se reusa por
  request."
- "El JSON que devuelve `suggest` son los 5 chars siguientes mas
  probables segun el modelo — útil para un dropdown."

**Cleanup**: en la Terminal A, `Ctrl+C` para matar uvicorn.

---

## Demo 3 — VS Code con la extension (~5 min)

La demo "estrella". La extension real se conecta al modelo.

### 3.1 Arrancar la extension

```bash
code vscode-extension/
```

VS Code abre la carpeta. En la barra inferior aparece
**"Run Extension"** como configuracion de launch. Presionar **F5**.

**Resultado**: se abre una **ventana nueva** de VS Code titulada
"[Extension Development Host]". En la barra de status dice
"rnnC server alive" cuando el subproceso Python termino de cargar
(5-10 s la primera vez).

### 3.2 Probar la extension

1. En la ventana de Extension Development Host, abrir cualquier
   archivo `.c` (puede ser `dataset/sample/funciones.c`).
2. Posicionar el cursor en una linea vacia al final.
3. Escribir un prompt, por ejemplo:
   ```c
   int sum
   ```
4. Presionar **Ctrl+Shift+Space**.

**Resultado esperado:** el modelo inserta la continuacion
directamente en el editor, por ejemplo:

```c
int sum[mid] < nums[0] == 0)
```

**Probar con mas prompts** (cualquier linea nueva, Ctrl+Shift+Space):

```c
void print
for (int
if (a
return
```

### 3.3 Probar el comando "elegir siguiente caracter"

1. En un archivo `.c`, escribir `int ` (con espacio).
2. `Ctrl+Shift+P` → "RNN: elegir siguiente caracter".
3. Aparece un quick-pick con `["b", "a", "s", "i", "c"]` (los 5
   chars mas probables segun el modelo).
4. Elegir uno → se inserta en el cursor.

### 3.4 Probar el comando "estado del servidor"

1. `Ctrl+Shift+P` → "RNN: estado del servidor".
2. Aparece un message arriba: "RNN server alive (pong=true). Use
   Ctrl+Shift+Space in a C file."

**Que decir:**

- "La extension hace `spawn` del subproceso Python al activarse. El
  modelo se carga una vez y se queda en memoria."
- "El primer Ctrl+Shift+Space tarda ~10 s por el warmup de TF; los
  siguientes son instantaneos."
- "La comunicacion es JSON-line por stdin/stdout — el mismo patron
  que usan Copilot, Tabnine, gopls, etc."

**Cleanup**: cerrar la Extension Development Host. El subproceso
se mata automaticamente.

### 3.5 (Opcional) Instalar la extension en tu VS Code real

Si querés usar la extension sin tener que abrir `vscode-extension/`
cada vez:

```bash
cd /home/jojo/develop/academic/IA/3-RNN/vscode-extension
npm install -g @vscode/vsce
vsce package
# produce rnn-c-autocomplete-0.1.0.vsix
code --install-extension rnn-c-autocomplete-0.1.0.vsix
```

A partir de aca, abrir cualquier `.c` en tu VS Code normal y
`Ctrl+Shift+Space` funciona. La extension busca el modelo en
`models/rnn_v1.keras` relativo a la raiz del workspace.

---

## Demo 4 — Test end-to-end sin VS Code (~1 min)

Sirve para mostrar que el flujo de la extension funciona **sin
abrir VS Code**. Util si no querés depender del editor.

```bash
node vscode-extension/test_e2e.js
```

**Salida esperada:**

```
Spawning: python /home/jojo/develop/academic/IA/3-RNN/src/server_stdio.py
/.../models/rnn_v1.keras

--- ping ---
> { id: 1, ok: true, pong: true }

--- suggest 'int ' ---
> { id: 2, ok: true, items: [ 'b', 'a', 's', 'i', 'c' ] }

--- complete 'int sum' (max_new=20) ---
> { id: 3, ok: true, text: '[mid] < nums[0] == 0' }
>>> prompt+text: int sum[mid] < nums[0] == 0

--- complete 'void print' (max_new=20) ---
> { id: 4, ok: true, text: 'f("FAILED: %s\\n"); }' }
>>> prompt+text: void printf("FAILED: %s\n"); }
```

**Que decir:**

- "Este test replica exactamente lo que hace `extension.js`
  internamente: spawn, stdin/stdout, JSON-line, request correlativo
  por id."
- "Si este test pasa, la extension va a funcionar. Si falla, ni
  abran VS Code."

---

## Resumen rapido: que demo mostrar cuando

| Situacion                              | Demo                    | Tiempo       |
| -------------------------------------- | ----------------------- | ------------ |
| "Mostrame algo rapido"                 | Demo 1 (CLI)            | 1 min        |
| "Quiero ver una API"                   | Demo 2 (FastAPI + curl) | 2 min        |
| "Quiero ver la integracion con editor" | Demo 3 (VS Code F5)     | 5 min        |
| "No tengo VS Code a mano"              | Demo 4 (test_e2e.js)    | 1 min        |
| "Mostrame todo"                        | Demo 1 → 2 → 3 → 4      | 10 min total |

---

## Troubleshooting (las 5 cosas que pueden salir mal)

### 1. `python: command not found` o version incorrecta

**Causa**: el venv unificado no se activo o `pyenv` no esta en el PATH.
**Solucion**:

```bash
cd /home/jojo/develop/academic/IA
source .venv/bin/activate.fish
python --version    # debe decir 3.11.9
```

### 2. `ModuleNotFoundError: No module named 'tensorflow'`

**Causa**: el venv esta activo pero las dependencias no se instalaron.
**Solucion**:

```bash
cd /home/jojo/develop/academic/IA
pip install -r 1-game/requirements.txt
pip install -r 2-CNN/requirements.txt
pip install -r 3-RNN/requirements.txt
```

### 3. `OSError: [Errno 98] Address already in use` (FastAPI)

**Causa**: el puerto 8000 esta ocupado por otra app.
**Solucion**: usar otro puerto:

```bash
uvicorn src.api:app --port 8001
```

### 4. La extension VS Code dice "server not running"

**Causa**: VS Code no encuentra `python` o el path al script.
**Solucion**: en la Extension Development Host, abrir
`Ctrl+Shift+P` → "Preferences: Open User Settings (JSON)" y agregar:

```json
{
  "rnnC.pythonPath": "/home/jojo/develop/academic/IA/.venv/bin/python",
  "rnnC.serverScript": "/home/jojo/develop/academic/IA/3-RNN/src/server_stdio.py",
  "rnnC.modelPath": "/home/jojo/develop/academic/IA/3-RNN/models/rnn_v1.keras"
}
```

(Rutas absolutas, no relativas.)

### 5. El primer Ctrl+Shift+Space tarda 10 segundos y los siguientes son instantaneos

**Esto NO es un bug.** Es el warmup de TensorFlow. La primera
invocacion compila el grafo y mueve tensores. Las siguientes
reutilizan el modelo ya cargado. Si querés evitarlo, podés
pre-calentar el server manualmente:

```bash
python -c "from src.predict import RNNModel; from pathlib import Path; RNNModel.load(Path('models/rnn_v1.keras'))"
```

---

## Cleanup (despues de la demo)

```bash
# Matar uvicorn si quedo corriendo
pkill -9 -f "uvicorn src.api"

# Matar el subproceso de la extension (si quedo)
pkill -9 -f "server_stdio"

# (Opcional) borrar artefactos para liberar espacio
rm -rf dataset/processed models/rnn_v1.keras models/rnn_v1.meta.json

# El venv unificado vive en IA/.venv — para borrarlo hay que ir a la raiz:
# cd ../.. && rm -rf .venv
```

El dataset `dataset/sample/funciones.c` (~70 KB) puede quedar —
se regenera en 2 s con `python src/sample_clean.py`.

---

## Cheatsheet: los 3 comandos que resuelven todo

Si solo te acordás de 3 cosas:

```bash
# Recreate everything from scratch
python src/sample_clean.py && python src/preprocess.py && python src/train.py

# Run any demo
python src/predict.py --prompt "int sum"   # demo 1
uvicorn src.api:app --port 8000            # demo 2
code vscode-extension/                     # demo 3
node vscode-extension/test_e2e.js          # demo 4
```

Eso es todo. La demo no toma mas de 10 min en total.
