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
    X = [[s.velocidad_bala, s.distancia] for s in samples]
    y = [s.salto for s in samples]
    clases = sorted(set(y))
    if len(clases) < 2:
        clase_unica = int(clases[0])
        tipo = "SIEMPRE NO-SALTA (0)" if clase_unica == 0 else "SIEMPRE SALTA (1)"
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
    velocidad_bala: int,
) -> Tuple[bool, Optional[float]]:
    if (not bala_disparada) or (not en_suelo):
        return False, None

    distancia = abs(jugador_x - bala_x)

    if clase_unica is not None and modelo is None:
        proba_salto = 1.0 if clase_unica == 1 else 0.0
        return clase_unica == 1, proba_salto

    if modelo is None or scaler is None:
        return False, None

    X = [[float(velocidad_bala), float(distancia)]]
    Xs = scaler.transform(X)
    if hasattr(modelo, "predict_proba"):
        proba_salto = float(modelo.predict_proba(Xs)[0][1])
        decision = proba_salto >= 0.5
        return decision, proba_salto

    pred = int(modelo.predict(Xs)[0])
    proba_salto = 1.0 if pred == 1 else 0.0
    return pred == 1, proba_salto
