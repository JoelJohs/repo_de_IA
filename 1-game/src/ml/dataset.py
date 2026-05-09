import csv
import os
from typing import List

from core.tipos import Sample


def registrar_decision_manual(
    datos_modelo: List[Sample],
    bala_disparada: bool,
    jugador_x: int,
    bala_x: int,
    en_suelo: bool,
    velocidad_bala: int,
) -> None:
    if not bala_disparada:
        return
    distancia = abs(jugador_x - bala_x)
    salto_label = 0 if en_suelo else 1
    datos_modelo.append(
        Sample(
            velocidad_bala=float(velocidad_bala),
            distancia=float(distancia),
            salto=salto_label,
        )
    )


def exportar_datos_csv(datos_modelo: List[Sample], base_dir: str) -> str:
    if not datos_modelo:
        return "No hay datos para exportar."

    ruta = os.path.join(base_dir, "datos_mlp.csv")

    try:
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["velocidad_bala", "distancia", "salto"])
            for s in datos_modelo:
                writer.writerow([s.velocidad_bala, s.distancia, s.salto])
    except Exception as e:
        return f"Error al guardar CSV: {e}"

    return f"CSV guardado en datos_mlp.csv ({len(datos_modelo)} filas)."
