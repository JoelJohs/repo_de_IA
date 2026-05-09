# Interfaz y menu

Archivo principal: `src/ui/menu.py`

## Menu principal

Funcion: `dibujar_menu(...)`

Responsabilidades:

- Pinta el fondo.
- Renderiza el titulo y opciones.
- Muestra estado actual de datos y modelo.
- Muestra mensajes contextuales (errores o confirmaciones).

## Controles que se muestran

- `M`, `A`, `T`, `C`, `F`, `Q` (modos y acciones generales).
- `ESPACIO` para saltar.
- `ABAJO` o `S` para agacharse.

## Textos y estados

El menu usa las constantes de color en `src/core/constantes.py`.

Lineas de estado mostradas:

- Memoria: cantidad de muestras en `datos_modelo`.
- Modelo: si hay o no un MLP entrenado.
- Resolucion actual y escala.

## Notas de usabilidad

- Los mensajes se muestran con `fuente_chica`.
- Si el modelo no esta entrenado y se presiona `A`, se muestra advertencia.
