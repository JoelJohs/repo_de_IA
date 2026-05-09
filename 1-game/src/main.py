import os
import sys


def _configurar_ruta() -> None:
    if __package__:
        return
    raiz = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if raiz not in sys.path:
        sys.path.insert(0, raiz)


def main() -> None:
    _configurar_ruta()
    if __package__:
        from .juego.juego import Juego
    else:
        from src.juego.juego import Juego

    Juego().loop()


if __name__ == "__main__":
    main()
