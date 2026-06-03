# nvim-rnn (cargado como `rnn`)

Plugin Neovim para autocompletado de C/C++ usando el modelo RNN del proyecto
`3-RNN/` (carpeta hermana en este repo).

## Que hace

Captura el contenido de la linea actual del editor (todo lo que esta antes
del cursor), lo envia al modelo RNN entrenado (`models/rnn_v1.keras`) via el
servidor stdio `src/server_stdio.py` (o via el wrapper `rnn-complete.fish`)
e inserta la continuacion en el cursor.

Por defecto se activa solo en archivos `.c` y `.cpp`.

## Mapeos (modo insert)

| Tecla | Accion |
|---|---|
| `Ctrl+Space` | Completa la linea actual e inserta la sugerencia |
| `Ctrl+Shift+P` | Muestra la sugerencia en una ventana flotante (sin insertar) |

Tambien podes usar los comandos `:RnnComplete` y `:RnnPreview`.

## Instalacion (1 minuto)

En tu `~/.config/nvim/init.lua`:

```lua
-- Cargar el plugin (ajusta la ruta si tu repo esta en otra ubicacion)
-- Truco: hay que agregar el PADRE al package.path, no la carpeta del plugin,
-- para que Lua pueda encontrar <parent>/rnn/init.lua como modulo "rnn".
local plugin_parent = vim.fn.expand("~/IA/scripts")
vim.opt.runtimepath:prepend(plugin_parent .. "/rnn")
package.path = plugin_parent .. "/?.lua;" .. plugin_parent .. "/?/init.lua;" .. package.path

-- Activar con la config por defecto
require("rnn").setup()

-- O con overrides, por ejemplo para apuntar a otra ruta del repo:
-- require("rnn").setup({
--   rnn_complete_script = "/otro/path/3-RNN/scripts/rnn-complete.fish",
--   trigger_key = "<C-Space>",
--   filetypes = { "c", "cpp", "h", "hpp" },
-- })
```

Listo. Abrí cualquier `.c` y probá `Ctrl+Space`.

## Requisitos

- Neovim 0.5+ (probado con 0.9+)
- El venv unificado en `~/IA/.venv` con `tensorflow==2.16.1` instalado
  (es el venv compartido de los 3 proyectos)
- `3-RNN/models/rnn_v1.keras` entrenado (`python src/train.py --epochs 80`)

El plugin usa por defecto el wrapper `3-RNN/scripts/rnn-complete.fish`, que
a su vez invoca al python del venv unificado (`~/IA/.venv/bin/python`) y
maneja opciones (`-n`, `-t`, `-s`).

## Opciones de `setup(opts)`

| Campo | Default | Descripcion |
|---|---|---|
| `rnn_complete_script` | `~/IA/3-RNN/scripts/rnn-complete.fish` | Path al wrapper .fish |
| `python_bin` | `~/IA/.venv/bin/python` | Python del venv raiz (sin TF) |
| `server_script` | `~/IA/3-RNN/src/server_stdio.py` | Script del server stdio |
| `model_path` | `~/IA/3-RNN/models/rnn_v1.keras` | Modelo entrenado |
| `max_new` | `60` | Caracteres a generar por completada |
| `temperature` | `0.4` | Sampling temperature (0 = greedy) |
| `seed` | `42` | Semilla aleatoria (reproducibilidad) |
| `trigger_key` | `<C-Space>` | Tecla en modo insert para completar |
| `filetypes` | `{ "c", "cpp" }` | En que tipos de archivo se mapea |
| `use_fish_wrapper` | `true` | Si `false`, llama directo a `server_stdio.py` |

## Por que un wrapper y no llamar directo

El plugin envia el prefijo al wrapper `.fish` que a su vez invoca
`src/predict.py`. Esto desacopla el plugin del path del modelo y centraliza
toda la logica de invocacion en un solo lugar. Si queres cambiar el script
que arma la peticion JSON, solo tocas el wrapper o `src/predict.py`, no el
plugin.

Si el wrapper .fish no existe o no es ejecutable, el plugin cae
automaticamente a la llamada directa via `server_stdio.py` (JSON-line por
stdin).
