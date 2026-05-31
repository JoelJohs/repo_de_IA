from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from core.tipos import Sample


def entrenar_arbol(
    datos_modelo: List[Sample],
) -> Tuple[Optional[DecisionTreeClassifier], Optional[int], str]:
    samples = list(datos_modelo)
    if len(samples) < 80:
        clase_unica = 0
        return (
            None,
            clase_unica,
            "Modelo trivial entrenado: SIEMPRE NADA (0)."
            " Junta datos (>= 80) para un modelo mas fino.",
        )
    X = [
        [
            s.velocidad_bala,
            s.distancia,
            s.bala_y,
            float(s.bala_arriba),
            float(s.puntaje),
            float(s.ataque_color),
        ]
        for s in samples
    ]
    y = [s.accion for s in samples]
    clases = sorted(set(y))
    if len(clases) < 2:
        clase_unica = int(clases[0])
        tipo = (
            "SIEMPRE NADA (0)"
            if clase_unica == 0
            else "SIEMPRE SALTA (1)"
            if clase_unica == 1
            else "SIEMPRE AGACHA (2)"
        )
        return (
            None,
            clase_unica,
            f"Modelo trivial entrenado: {tipo}. Junta datos de ambas clases para un modelo mas fino.",
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=4)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, None, f"Arbol entrenado. Accuracy test ≈ {acc:.3f}"


def decision_auto_arbol(
    modelo: Optional[DecisionTreeClassifier],
    clase_unica: Optional[int],
    bala_disparada: bool,
    en_suelo: bool,
    jugador_x: int,
    bala_x: int,
    bala_y: int,
    velocidad_bala: int,
    puntaje: int,
    bala_arriba: bool,
) -> Tuple[int, Optional[float]]:
    if (not bala_disparada) or (not en_suelo):
        return 0, None

    distancia = abs(jugador_x - bala_x)

    if clase_unica is not None and modelo is None:
        proba_salto = 1.0 if clase_unica == 1 else 0.0
        return clase_unica, proba_salto

    if modelo is None:
        return 0, None

    X = [
        [
            float(velocidad_bala),
            float(distancia),
            float(bala_y),
            1.0 if bala_arriba else 0.0,
            float(puntaje),
            1.0 if bala_arriba else 0.0,
        ]
    ]
    proba_salto = None
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X)[0]
        clases = list(modelo.classes_)
        if 1 in clases:
            proba_salto = float(proba[clases.index(1)])
    pred = int(modelo.predict(X)[0])
    return pred, proba_salto
