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

## Install (production)

```bash
cd vscode-extension
vsce package          # produces a .vsix (needs `npm i -g @vscode/vsce`)
code --install-extension rnn-c-autocomplete-0.1.0.vsix
```

## Configuration (settings.json)

```json
{
  "rnnC.pythonPath": "python",
  "rnnC.serverScript": "src/server_stdio.py",
  "rnnC.modelPath": "models/rnn_v1.keras",
  "rnnC.maxNew": 60,
  "rnnC.temperature": 0.4
}
```

## Usage

Open a `.c` file and press **Ctrl+Shift+Space** → inserts the model's
continuation at the cursor. Use the command palette → "RNN: estado del
servidor" to ping the server.

The server is a long-lived subprocess. The first keystroke triggers
~5-10 s of TF warmup (one-time model load); subsequent calls are fast.
