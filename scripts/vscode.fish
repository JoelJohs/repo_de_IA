#!/usr/bin/env fish
# Abre VSCode con la extension RNN de 3-RNN cargada en modo dev.
# Uso:
#   ./scripts/vscode.fish         # abre VSCode con el workspace en 3-RNN
#   ./scripts/vscode.fish prod    # instala la extension globalmente
#
# Despues de que se abra VSCode:
#   1. F5 (Run > Start Debugging) para lanzar la Extension Development Host
#   2. En la nueva ventana: abrir 3-RNN/vscode-extension/prueba.c
#   3. Apretar Ctrl+Shift+Space en cualquier linea -> el modelo completa

set script_dir (dirname (realpath (status filename)))
set repo_root (dirname $script_dir)
set ext_dir "$repo_root/3-RNN/vscode-extension"

if not test -d "$ext_dir"
    echo "[vscode] No se encontro la extension en $ext_dir" >&2
    exit 1
end

if not command -v code >/dev/null
    echo "[vscode] El comando 'code' no esta en el PATH" >&2
    echo "[vscode] Instalar VSCode y correr 'code --version' primero" >&2
    exit 1
end

# Sanity check: venv unificado existe
set python_bin "$repo_root/.venv/bin/python"
if not test -x $python_bin
    echo "[vscode] No se encontro el venv unificado en $python_bin" >&2
    exit 1
end

# Sanity check: modelo entrenado
set model "$repo_root/3-RNN/models/rnn_v1.keras"
if not test -f $model
    echo "[vscode] No se encontro el modelo en $model" >&2
    echo "[vscode] Entrenarlo primero: cd 3-RNN && python src/train.py --epochs 80" >&2
    exit 1
end

# Sanity check: node disponible
if not command -v node >/dev/null
    echo "[vscode] 'node' no esta en el PATH. Instalar nodejs." >&2
    exit 1
end

echo "[vscode] Venv:  $python_bin"
echo "[vscode] Model: $model"
echo "[vscode] Ext:   $ext_dir"
echo
echo "[vscode] Smoke test del server (lo que la extension va a spawn-ear)..."
node "$ext_dir/test_e2e.js" 2>&1 | tail -8
echo
echo "[vscode] Abriendo VSCode con la extension..."

if test (count $argv) -ge 1; and test "$argv[1]" = "prod"
    echo "[vscode] Modo produccion: empaquetando e instalando la extension..."
    cd $ext_dir
    npm install -g @vscode/vsce
    vsce package
    code --install-extension rnn-c-autocomplete-0.1.0.vsix
else
    code $ext_dir
end
