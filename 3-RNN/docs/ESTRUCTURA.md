# Estructura del proyecto

```text
3-RNN/
├── dataset/
│   ├── funciones.c              # corpus crudo (input, no se entrena)
│   ├── sample/                  # 150 funciones seleccionadas (input del modelo)
│   │   ├── funciones.c
│   │   └── REPORTE.md
│   └── processed/               # ventanas char-level listas para entrenar
│       ├── X.npy, Y.npy
│       └── meta.json
├── models/
│   ├── rnn_v1.keras             # pesos del modelo
│   ├── rnn_v1.meta.json         # vocab + block_size + parametros
│   └── rnn_v1.history.json      # curva de perdida
├── notebooks/
│   └── entrenamiento.ipynb      # entrega formal con outputs ejecutados
├── src/
│   ├── __init__.py
│   ├── sample_clean.py          # F1: muestreo
│   ├── preprocess.py            # F2: ventanas
│   ├── train.py                 # F3: modelo + entrenamiento
│   ├── predict.py               # F4: complete(prefix, ...)
│   ├── api.py                   # F5: FastAPI
│   ├── server_stdio.py          # F6: JSON-line para VS Code
│   └── dataset_tools.py         # utilidad historica (no usada en este flujo)
├── vscode-extension/            # F7: extension de VS Code
│   ├── package.json
│   ├── extension.js
│   └── README.md
├── docs/
│   ├── INICIO.md
│   ├── ENTORNO.md
│   ├── ESTRUCTURA.md            # este archivo
│   ├── DATASET.md
│   └── DATASET_PROCESO.md
├── requirements.txt
└── README.md
```

## Convenciones

- **Modelos versionados**: `rnn_v1`, `rnn_v2`, etc. Cada cambio
  importante se refleja en un tag nuevo, no se sobreescriben pesos.
- **Dataset no versionado**: `dataset/funciones.c` y `dataset/sample/`
  estan en `.gitignore` (el primero pesa 466 MB; el segundo es
  regenerable).
- **Notebook ejecutado**: `notebooks/entrenamiento.ipynb` siempre
  contiene outputs - quien lo abre ve la corrida real.
- **API dual**: FastAPI para HTTP/debug, `server_stdio.py` para la
  extension. Comparten `src/predict.py:complete()`.
