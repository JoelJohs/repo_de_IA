from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from core.tipos import Sample


def entrenar_modelo(
    datos_modelo: List[Sample],
) -> Tuple[Optional[MLPClassifier], Optional[StandardScaler], Optional[int], str]:
    samples = list(datos_modelo)
    if len(samples) < 80:
        return None, None, None, "Necesitas más datos (>= 80). Juega en MANUAL."
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
            None,
            clase_unica,
            f"Modelo trivial entrenado: {tipo}. Junta datos de ambas clases para un modelo más fino.",
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = MLPClassifier(
        hidden_layer_sizes=(3, 3),
        activation="relu",
        solver="adam",
        max_iter=300000,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, scaler, None, f"MLP entrenado. Accuracy test ≈ {acc:.3f}"


def decision_auto_saltar(
    modelo: Optional[MLPClassifier],
    scaler: Optional[StandardScaler],
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

    if modelo is None or scaler is None:
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
    Xs = scaler.transform(X)
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(Xs)[0]
        clases = list(modelo.classes_)
        proba_salto = None
        if 1 in clases:
            proba_salto = float(proba[clases.index(1)])
        pred = int(modelo.predict(Xs)[0])
        return pred, proba_salto

    pred = int(modelo.predict(Xs)[0])
    proba_salto = 1.0 if pred == 1 else 0.0
    return pred, proba_salto
