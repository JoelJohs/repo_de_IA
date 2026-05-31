#!/usr/bin/env fish

set script_dir (dirname (realpath (status filename)))
set project_root (dirname $script_dir)

set python_bin "$project_root/.venv/bin/python"

if not test -x $python_bin
    echo "No se encontro el venv en $python_bin"
    echo "Crea el entorno y dependencias primero."
    exit 1
end

set dataset_dir "$project_root/dataset"
set test_dir "$project_root/test"
set per_class 5
set seed 42

if test (count $argv) -ge 1
    set per_class $argv[1]
end

if test (count $argv) -ge 2
    set seed $argv[2]
end

if test (count $argv) -ge 3
    echo "Uso: (basename (status filename)) [por_clase] [seed]"
    exit 1
end

exec $python_bin "$project_root/src/data/sample_to_test.py" --dataset $dataset_dir --test-dir $test_dir --per-class $per_class --seed $seed
