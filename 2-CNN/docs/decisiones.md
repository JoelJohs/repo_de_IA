# Decisiones del modelo (resumen rapido)

## Cambios principales

- **Imagenes a 32x32**: el dataset original es 32x32, asi que no conviene reducir a 21x28 porque se pierde detalle.
- **Modelo simple (1 conv)**: se alinea 100% al notebook para no desviarse de la pauta del curso.
- **SGD + 20 epochs**: igual que el notebook (learning rate 0.001 y decay).
- **Mejora A (hyperparams)**: se sube a 40 epochs y learning rate a 0.005 sin cambiar arquitectura.
- **Sin augmentation**: el notebook no lo usa, asi que se quita para mantener la comparacion.
- **Muestreo a test**: se agrega un script para mover imagenes del dataset a `test/` y probar predicciones rapidas.
- **Carga manual**: se cambia el pipeline para leer imagenes con `os.walk` como en el notebook.

## Lo que espero mejorar

- Probabilidades mas separadas (menos "todo cerca de 0.2").
- Mejor accuracy en test.
- Predicciones mas estables en `test/`.
