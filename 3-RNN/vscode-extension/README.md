# RNN Autocomplete (C) - VS Code extension

Char-level vanilla RNN autocompletion for C files. Spawns
`src/server_stdio.py` as a subprocess and exchanges JSON-line messages.

El venv unificado vive en `../../IA/.venv` (compartido con los proyectos
1-game y 2-CNN). La extension ya viene configurada para usarlo via
`package.json` (default `rnnC.pythonPath`) y `.vscode/settings.json`.

## Quick start (modo dev, 1 minuto)

```bash
# 1) Abrir la extension en VS Code
code vscode-extension/

# 2) Dentro de VS Code: F5 (la config de debug ya viene en .vscode/launch.json)
#    Se abre una Extension Development Host con la extension cargada.

# 3) En la nueva ventana, abrir vscode-extension/prueba.c
#    Posicionarse en una linea y apretar Ctrl+Shift+Space
```

## Probar el server sin VS Code

```bash
# Activa el venv unificado y corre el e2e test que habla el mismo
# protocolo que la extension
cd /home/jojo/develop/academic/IA
source .venv/bin/activate.fish
cd 3-RNN/vscode-extension
node test_e2e.js
```

El test hace `ping`, `suggest`, y dos `complete` — todo via JSON-line
por stdin/stdout, igual que la extension. Si el test pasa, la extension
va a funcionar en VS Code.

Para overridear el Python del server:

```bash
PYTHON_BIN=/usr/bin/python3.11 node test_e2e.js
```

## Configuracion (settings.json del workspace)

La extension ya trae `.vscode/settings.json` apuntando al venv unificado:

```json
{
  "rnnC.pythonPath": "/home/jojo/develop/academic/IA/.venv/bin/python",
  "rnnC.serverScript": "src/server_stdio.py",
  "rnnC.modelPath": "models/rnn_v1.keras",
  "rnnC.maxNew": 60,
  "rnnC.temperature": 0.4
}
```

`serverScript` y `modelPath` son relativos al primer workspace folder
abierto. Si abris la carpeta `3-RNN/`, las resuelve como
`3-RNN/src/server_stdio.py` y `3-RNN/models/rnn_v1.keras`.

## Install (produccion)

```bash
cd vscode-extension
npm install -g @vscode/vsce
vsce package
code --install-extension rnn-c-autocomplete-0.1.0.vsix
```

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
