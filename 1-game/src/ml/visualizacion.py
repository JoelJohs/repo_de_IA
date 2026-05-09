from typing import List

import matplotlib
import matplotlib.pyplot as plt

from core.tipos import Sample


def _configurar_matplotlib() -> None:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            pass
    plt.ion()


def graficar_datos_2d(datos_modelo: List[Sample]) -> str:
    if not datos_modelo:
        return "No hay datos para graficar."

    _configurar_matplotlib()

    xs = [s.distancia for s in datos_modelo]
    ys = [s.velocidad_bala for s in datos_modelo]
    cs = ["red" if s.salto == 1 else "blue" for s in datos_modelo]

    fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
    plt.figure(fig_num)
    plt.clf()

    ax = plt.gca()
    ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
    ax.set_xlabel("Distancia jugador-bala")
    ax.set_ylabel("Velocidad bala")
    ax.set_title("Datos entrenamiento MLP (rojo=salto, azul=no salto)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.draw()

    return "Mostrando gráfica 2D interactiva (puedes rotar/zoom)."


def graficar_datos_3d(datos_modelo: List[Sample]) -> str:
    if not datos_modelo:
        return "No hay datos para graficar."

    _configurar_matplotlib()

    xs = [s.distancia for s in datos_modelo]
    ys = [s.velocidad_bala for s in datos_modelo]
    zs = list(range(len(datos_modelo)))
    cs = ["red" if s.salto == 1 else "blue" for s in datos_modelo]

    fig = plt.figure("Datos MLP - 3D", figsize=(8, 6))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
    ax.set_xlabel("Distancia")
    ax.set_ylabel("Velocidad bala")
    ax.set_zlabel("Índice (tiempo aproximado)")
    ax.set_title("Datos entrenamiento MLP 3D (rojo=salto, azul=no salto)")
    plt.tight_layout()
    plt.show(block=False)
    plt.draw()

    return "Mostrando gráfica 3D interactiva (puedes rotar/zoom)."
