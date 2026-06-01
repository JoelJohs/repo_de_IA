# Proceso historico del dataset (legacy)

> **Nota:** este documento describe el flujo original basado en
> `src/dataset_tools.py`. **Ya no se usa** en el flujo principal del
> Proyecto 3. El flujo vigente esta en `docs/DATASET.md` y se reduce a
> `src/sample_clean.py`.

## Por que ya no se usa

El flujo original tomaba el corpus crudo de 466 MB, lo partia por tema
(arrays, math, sorting, etc.) y deduplicaba. Resultado: **883 854
funciones** en archivos `dataset/clean/*.c`, ~466 MB adicionales. Esto:

1. Saturaba el `.gitignore` con archivos enormes.
2. Entrenar sobre 880 K funciones en CPU es prohibitivo (>24 h).
3. La actividad pide **"60 funciones unicas con estilo consistente"**,
   no 880 K con estilo heterogeneo.

## Lo que se conserva

`src/dataset_tools.py` sigue en el repo como referencia historica.
No se ejecuta en el flujo nuevo. Si en algun momento se quiere
re-crear el `dataset/clean/`, se puede correr:

```bash
python src/dataset_tools.py \
    --c-input dataset/funciones.c \
    --jsonl-dir dataset \
    --out-dir dataset/clean
```

Esto regenera la particion por tema (~30-60 s) pero no se usa para
entrenar.

## Lo que se reemplazo

- `dataset/clean/` (~466 MB, 8 archivos `.c` por tema) → eliminado del
  flujo principal.
- `dataset_tools.py --c-input` → reemplazado por `sample_clean.py`.
- `src/preprocess.py` se reescribio para leer `dataset/sample/`
  (single file) en vez de iterar sobre `clean/*.c`.
