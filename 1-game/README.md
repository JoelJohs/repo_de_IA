# Proyecto 1 - Juego con IA (Pygame + MLP)

Juego 2D en Pygame que recolecta datos en modo manual y entrena un MLP para decidir acciones en modo automatico. El proyecto esta organizado por componentes para explicar el flujo con claridad.

## Ejecucion

Desde la carpeta `1-game/`:

```bash
python -m src.main
```

## Estructura principal

```text
1-game/
├── src/
│   ├── main.py
│   ├── core/
│   ├── juego/
│   ├── ml/
│   ├── render/
│   └── ui/
├── assets/
├── docs/
└── requirements.txt
```

## Documentacion

- `docs/arquitectura.md`: arquitectura por capas y flujo general.
- `docs/juego.md`: logica del juego y componentes.
- `docs/ml.md`: dataset, entrenamiento y decision automatica.
- `docs/render.md`: assets y dibujo.
- `docs/ui.md`: menu y controles.
- `docs/examen.md`: preguntas tipo examen y respuestas cortas.

## Controles

- `M`: modo manual (reinicia dataset y modelo).
- `A`: modo automatico (usa MLP entrenado).
- `T`: entrena MLP.
- `C`: exporta CSV.
- `F`: fullscreen.
- `Q`: salir.
- `ESPACIO`: saltar.
- `ABAJO` o `S`: agacharse.
- `ESC` o `P`: volver al menu.
