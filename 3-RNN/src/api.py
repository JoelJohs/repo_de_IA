"""Local API to serve autocomplete predictions."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RNN Autocomplete API")


class PredictRequest(BaseModel):
    prompt: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    # TODO: Load model and tokenizer, return prediction.
    return {"completion": ""}
