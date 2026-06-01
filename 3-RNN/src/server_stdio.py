"""JSON-line server over stdin/stdout for editor integration.

The editor spawns this script as a subprocess and writes one JSON request per
line to its stdin. This script writes one JSON response per line to stdout.

Protocol
--------
Request  : {"id": <int|str>, "method": "complete"|"suggest"|"ping", ...}
Response : {"id": ..., "ok": true|false, ...}

Methods
-------
complete  : {"prefix": str, "max_new"?: int=60, "temperature"?: float=0.5}
            -> {"text": "<generated continuation>"}
suggest   : {"prefix": str, "n"?: int=5}
            -> {"items": ["<top-1 next char>", ...]}

Run it from a shell for a quick smoke test:
    echo '{"id":1,"method":"complete","prefix":"int sum","max_new":30}' | python src/server_stdio.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

from .predict import RNNModel, _encode, _sample

DEFAULT_MODEL = Path("models/rnn_v1.keras")


def _load_rnn() -> RNNModel:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    return RNNModel.load(model_path)


def handle_complete(rnn: RNNModel, req: dict) -> dict:
    prefix = req.get("prefix", "")
    max_new = int(req.get("max_new", 60))
    temperature = float(req.get("temperature", 0.5))
    seed = req.get("seed")
    if seed is not None:
        np.random.seed(int(seed))
    state = _encode(prefix, rnn.stoi, rnn.block_size)
    out: list[str] = []
    for _ in range(max_new):
        logits = rnn.model.predict(state, verbose=0)[0, -1]
        idx = _sample(logits, temperature)
        out.append(rnn.itos[idx])
        state = np.concatenate([state[:, 1:], np.array([[idx]], dtype=np.int64)], axis=1)
    return {"text": "".join(out)}


def handle_suggest(rnn: RNNModel, req: dict) -> dict:
    prefix = req.get("prefix", "")
    n = int(req.get("n", 5))
    state = _encode(prefix, rnn.stoi, rnn.block_size)
    logits = rnn.model.predict(state, verbose=0)[0, -1]
    top = np.argsort(-logits)[:n]
    return {"items": [rnn.itos[int(i)] for i in top]}


def main() -> None:
    rnn = _load_rnn()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            if method == "complete":
                payload = handle_complete(rnn, req)
            elif method == "suggest":
                payload = handle_suggest(rnn, req)
            elif method == "ping":
                payload = {"pong": True}
            else:
                raise ValueError(f"unknown method: {method!r}")
            resp = {"id": req_id, "ok": True, **payload}
        except Exception as exc:  # noqa: BLE001
            resp = {
                "id": req.get("id") if isinstance(req, dict) else None,
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc(limit=2),
            }
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
