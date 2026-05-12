# Interfaz y menu

Archivo principal: `src/ui/menu.py`

## Menu principal

Funcion: `dibujar_menu(...)`

Responsabilidades:

- Pinta el fondo.
- Renderiza el titulo desde `assets/game/title.png` y las opciones.
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

## Estilo visual (retro + horror religioso)

- Fondo con la imagen `assets/game/title.png` escalada a pantalla completa.
- Viñeta ligera para mejorar contraste del texto.
- Panel semitransparente para el bloque de opciones.
- Acentos dorados en teclas y texto blanco para legibilidad.

## Layout del menu

- La imagen `assets/game/title.png` es TODO el fondo y ya incluye el titulo.
- El menu se dibuja en la parte inferior de la imagen, dentro de un panel oscuro.
- El bloque de estado queda al final del panel.
