#!/usr/bin/env python3
"""
FastAPI entry point for the RAG API.
Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RAG - Seguridad Pública México",
    description="API de consulta RAG sobre documentos de seguridad pública en México",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_MODEL = "qwen2.5:7b-instruct"
K = 5
TEMPERATURE = 0.1


@app.on_event("startup")
async def startup():
    from src.api.routes import init_api
    init_api(ollama_model=OLLAMA_MODEL, k=K, temperature=TEMPERATURE)


from src.api.routes import router
app.include_router(router)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
