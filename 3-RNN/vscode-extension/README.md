# RNN Autocomplete (C) - VS Code extension

Char-level vanilla RNN autocompletion for C files. Spawns
`src/server_stdio.py` as a subprocess and exchanges JSON-line messages.

## Install (dev mode)

1. Open this folder in VS Code:
   ```bash
   code vscode-extension/
   ```
2. Press `F5` → "Run Extension" launches a new VS Code window with the
   extension loaded.
3. Open a `.c` file in that window and press **Ctrl+Shift+Space**.

## Install (production)

```bash
cd vscode-extension
npm install -g @vscode/vsce
vsce package
code --install-extension rnn-c-autocomplete-0.1.0.vsix
```

## Configuration (settings.json del workspace)

```json
{
  "rnnC.pythonPath": "python",
  "rnnC.serverScript": "src/server_stdio.py",
  "rnnC.modelPath": "models/rnn_v1.keras",
  "rnnC.maxNew": 60,
  "rnnC.temperature": 0.4
}
```

Por defecto la extension corre desde la raiz del primer workspace
folder abierto. Si tu proyecto no es 3-RNN, ajusta `rnnC.serverScript`
a una ruta absoluta.

## Probar el servidor sin abrir VS Code

```bash
cd /home/jojo/develop/academic/IA/3-RNN
source .venv/bin/activate
node vscode-extension/test_e2e.js
```

`test_e2e.js` implementa **exactamente** el flujo de extension.js:
spawn del server, lectura/escritura JSON por linea, manejo de stderr
(tail-only). Si esto pasa, la extension va a funcionar.

## Commands

| Command | Keybind | Funcion |
|---|---|---|
| `RNN: completar linea` | `Ctrl+Shift+Space` (en archivos `.c`) | Inserta la continuacion en el cursor |
| `RNN: elegir siguiente caracter` | (palette) | Muestra quick-pick con top-5 chars |
| `RNN: estado del servidor` | (palette) | Ping para confirmar que el server vive |

## Notas de timing

- **Primer spawn**: ~5-10 s. Carga el modelo `.keras` en memoria.
- **Calls subsiguientes**: <1 s por completion de 20 chars.
- **Memoria**: ~500 MB RAM retenidos mientras la ventana este abierta
  (modelo + tensores).
