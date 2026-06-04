import csv
import os
from typing import List

from core.tipos import Sample


def registrar_decision_manual(
    datos_modelo: List[Sample],
    jugador_x: int,
    bala_x: int,
    bala_y: int,
    en_suelo: bool,
    agachado: bool,
    accion: int,
    velocidad_bala: int,
    bala_arriba: bool,
    ataque_color: int,
) -> None:
    distancia = abs(jugador_x - bala_x)
    salto_label = 0 if en_suelo else 1
    datos_modelo.append(
        Sample(
            velocidad_bala=float(velocidad_bala),
            distancia=float(distancia),
            salto=salto_label,
            bala_y=float(bala_y),
            bala_arriba=1 if bala_arriba else 0,
            agachado=1 if agachado else 0,
            accion=int(accion),
            ataque_color=int(ataque_color),
        )
    )


def exportar_datos_csv(datos_modelo: List[Sample], base_dir: str) -> str:
    if not datos_modelo:
        return "No hay datos para exportar."

    ruta = os.path.join(base_dir, "datos_mlp.csv")

    try:
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "velocidad_bala",
                    "distancia",
                    "salto",
                    "bala_y",
                    "bala_arriba",
                    "agachado",
                    "accion",
                    "ataque_color",
                ]
            )
            for s in datos_modelo:
                writer.writerow(
                    [
                        s.velocidad_bala,
                        s.distancia,
                        s.salto,
                        s.bala_y,
                        s.bala_arriba,
                        s.agachado,
                        s.accion,
                        s.ataque_color,
                    ]
                )
    except Exception as e:
        return f"Error al guardar CSV: {e}"

    return f"CSV guardado en datos_mlp.csv ({len(datos_modelo)} filas)."
