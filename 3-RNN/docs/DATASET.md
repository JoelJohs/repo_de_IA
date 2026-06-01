# Dataset

El proyecto usa **dos** copias del corpus de funciones en C, con
propositos diferentes.

## `dataset/funciones.c` (input crudo)

Volcado heterogeneo (~466 MB, ~196 K funciones) usado **solo** como
fuente de materia prima. El archivo original mezcla:

- multiples estilos de llaves (K&R y Allman),
- sangrias de 2 y 4 espacios,
- comentarios en varios idiomas (incluyendo chino),
- `int main()` con programas completos,
- macros, headers, includes.

**No se entrena directamente** con este archivo. Se conserva porque
fue la fuente historica del proyecto y `.gitignore` lo excluye del
repo, asi que no infla el codigo.

## `dataset/sample/funciones.c` (corpus de entrenamiento)

150 funciones "limpias" (~70 KB) que **si se usan** para entrenar.
Las genera `src/sample_clean.py` aplicando heuristicas:

| Filtro                 | Razon                                            |
| ---------------------- | ------------------------------------------------ |
| Longitud 30-1500 chars | Evita funciones triviales o gigantes             |
| ASCII puro             | Sin comentarios en otros idiomas                 |
| Sin `int main()`       | Solo queremos librerias, no entry points         |
| Llaves balanceadas     | Garantiza que se extrajo el bloque completo      |
| Firma reconocible      | Detecta `int/void/char/...` + nombre + `(` + `{` |
| Nombre no es keyword   | Excluye `if(`, `for(`, `while(`, etc.            |

Ademas, el script ordena los candidatos por un score que prioriza
funciones con control flow (`if`, `for`, `return`) y descarta las
extremadamente largas o cortas.

El reporte de cada corrida queda en
`dataset/sample/REPORTE.md` (conteo, razones de rechazo, 5 funciones
al azar como muestra).

## `dataset/processed/` (artefactos de preprocesamiento)

Salida de `src/preprocess.py`:

- `meta.json` - vocabulario, `block_size`, conteos.
- `X.npy`, `Y.npy` - ventanas deslizantes (int64, shape
  `(num_windows, 32)`).

Estos archivos son reproducibles a partir de `dataset/sample/funciones.c`.

## Como regenerar todo

```bash
python src/sample_clean.py         # genera dataset/sample/
python src/preprocess.py           # genera dataset/processed/
```

## Tamaños tipicos

| Archivo                       | Tamano |
| ----------------------------- | ------ |
| `dataset/funciones.c` (crudo) | 466 MB |
| `dataset/sample/funciones.c`  | 70 KB  |
| `dataset/processed/X.npy`     | ~17 MB |
| `dataset/processed/Y.npy`     | ~17 MB |
| `models/rnn_v1.keras`         | 244 KB |
