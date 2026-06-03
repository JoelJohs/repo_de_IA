#!/usr/bin/env fish
# Ejecuta el juego 1-game: recoleccion manual + MLP.
#
# Uso:
#   ./scripts/game.fish              # modo normal

set script_dir   (dirname (realpath (status filename)))
set repo_root    (dirname $script_dir)
set python_bin   "$repo_root/.venv/bin/python"
set project_root "$repo_root/1-game"

if not test -x $python_bin
    echo "[game] No se encontro el venv en $python_bin" >&2
    echo "[game] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

if not test -d "$project_root/src"
    echo "[game] No se encontro src/ en $project_root" >&2
    exit 1
end

cd $project_root
exec env PYTHONPATH="src" $python_bin -m src.main
