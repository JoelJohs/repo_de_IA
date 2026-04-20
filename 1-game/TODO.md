# TODO del proyecto (1-game)

Este archivo resume lo pendiente segun la guia y el estado actual del juego.

## Gameplay y controles

- [ ] Implementar agacharse del personaje para esquivar balas altas.
- [ ] Registrar en el dataset la accion de agacharse (nueva etiqueta o nueva clase).
- [ ] Ajustar colisiones para diferenciar salto vs agacharse.
- [ ] Balancear dificultad con balas altas y bajas.
- [ ] Agregar HUD minimo para mostrar estado (manual/auto, score o racha).

## Dataset y features

- [ ] Agregar nuevas features (altura_jugador, velocidad_vertical, tiempo_desde_salto, distancia_normalizada, velocidad_normalizada, tiempo_impacto_estimado, bala_anterior_esquivada).
- [ ] Actualizar estructura Sample con las nuevas features.
- [ ] Actualizar registrar_decision_manual con features nuevas.
- [ ] Exportar CSV mejorado con columnas completas.

## Modelos y entrenamiento

- [ ] Implementar multiples modelos (MLP profundo, Random Forest, SVM, KNN) y compararlos.
- [ ] Medir accuracy, precision, recall, F1 y matriz de confusion.
- [ ] Validacion cruzada (k-fold) para robustez.
- [ ] Seleccionar automaticamente el mejor modelo.

## Persistencia y reproducibilidad

- [ ] Guardar/cargar modelo y scaler (joblib).
- [ ] Guardar metadata del entrenamiento (fecha, features, muestras).
- [ ] Listar modelos guardados desde el menu.

## Visualizacion y analisis

- [ ] Graficas: matriz de confusion, comparacion de modelos, ROC.
- [ ] Importancia de features (si aplica) y distribuciones.
- [ ] Curva de aprendizaje para evaluar overfitting/underfitting.

## Extensiones opcionales

- [ ] Sistema de puntuacion y rachas.
- [ ] Dificultades (facil/medio/dificil/experto).
- [ ] Sistema de vidas y pantalla de game over.
- [ ] Power-ups basicos.
- [ ] Comparacion humano vs IA con estadisticas.
