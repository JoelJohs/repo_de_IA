"""FastAPI server exposing the char-level RNN for autocomplete.

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health      -> liveness
    POST /predict     -> {"prompt": str, "max_new"?: int, "temperature"?: float}
                       returns {"completion": str, "full": str}
    POST /suggest     -> {"prefix": str, "n"?: int}
                       returns {"items": [str, ...]}  (top-N greedy suggestions)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .predict import RNNModel, complete

DEFAULT_MODEL = Path("models/rnn_v1.keras")

app = FastAPI(title="RNN Autocomplete API", version="0.1.0")

_rnn: RNNModel | None = None


def get_rnn() -> RNNModel:
    global _rnn
    if _rnn is None:
        _rnn = RNNModel.load(DEFAULT_MODEL)
    return _rnn


class PredictRequest(BaseModel):
    prompt: str
    max_new: int = Field(default=60, ge=1, le=400)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)


class SuggestRequest(BaseModel):
    prefix: str
    n: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    rnn = get_rnn()
    cont = complete(rnn, req.prompt, req.max_new, req.temperature)
    return {"completion": cont, "full": req.prompt + cont}


@app.post("/suggest")
def suggest(req: SuggestRequest) -> dict:
    rnn = get_rnn()
    # greedy: pick the top-N next chars (no sampling) to give deterministic options
    from .predict import _encode
    import numpy as np

    state = _encode(req.prefix, rnn.stoi, rnn.block_size)
    logits = rnn.model.predict(state, verbose=0)[0, -1]
    top = np.argsort(-logits)[: req.n]
    items = [rnn.itos[int(i)] for i in top]
    return {"items": items}
