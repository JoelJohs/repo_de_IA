#!/usr/bin/env fish

set script_dir (dirname (realpath (status filename)))
set project_root (dirname $script_dir)

set repo_root (dirname $project_root)
set python_bin "$repo_root/.venv/bin/python"
set config_path "$project_root/config/default.yaml"
set default_model "$project_root/config/models/best_model.keras"
set default_images_dir "$project_root/test"

if not test -x $python_bin
    echo "No se encontro el venv raiz en $python_bin"
    echo "El venv unificado esta en la raiz del repo (IA/.venv)."
    exit 1
end

set model_path $default_model
set image_path $default_images_dir

if test (count $argv) -ge 1
    set model_path $argv[1]
end

if test (count $argv) -ge 2
    set image_path $argv[2]
end

if test (count $argv) -ge 3
    set config_path $argv[3]
end

if test (count $argv) -ge 4
    echo "Uso: (basename (status filename)) [modelo] [imagen|directorio] [config]"
    exit 1
end

exec $python_bin "$project_root/predict.py" --config $config_path --model $model_path --image $image_path
