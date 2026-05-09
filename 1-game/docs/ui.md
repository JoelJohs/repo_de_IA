# Interfaz y menu

Archivo principal: `src/ui/menu.py`

## Menu principal

Funcion: `dibujar_menu(...)`

Responsabilidades:

- Pinta el fondo.
- Renderiza el titulo y opciones.
- Muestra estado actual de datos y modelo.
- Muestra mensajes contextuales (errores o confirmaciones).

## Cambios requeridos por la entrega

Si se agrega agacharse y nuevas balas, se recomienda actualizar los textos del menu para incluir los controles nuevos:

- `ABAJO` o `S`: agacharse.
- `X` o `K`: ataque frontal (si se implementa).

## Textos y estados

El menu usa las constantes de color en `src/core/constantes.py`.

Lineas de estado mostradas:

- Memoria: cantidad de muestras en `datos_modelo`.
- Modelo: si hay o no un MLP entrenado.
- Resolucion actual y escala.

## Notas de usabilidad

- Los mensajes se muestran con `fuente_chica`.
- Si el modelo no esta entrenado y se presiona `A`, se muestra advertencia.
