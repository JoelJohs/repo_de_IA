#!/usr/bin/env fish

set script_dir (dirname (realpath (status filename)))
set project_root (dirname $script_dir)

set python_bin "$project_root/.venv/bin/python"
set config_path "$project_root/config/default.yaml"

if not test -x $python_bin
    echo "No se encontro el venv en $python_bin"
    echo "Crea el entorno y dependencias primero."
    exit 1
end

if test (count $argv) -ge 1
    set config_path $argv[1]
end

if test (count $argv) -ge 2
    echo "Uso: (basename (status filename)) [config]"
    exit 1
end

exec $python_bin "$project_root/train.py" --config $config_path
