#!/usr/bin/env python3
"""
Enriquece el chunks.jsonl existente con author/institution/year.

Más rápido que reprocesar PDFs: hace JOIN entre el source_file del chunk
y el campo "archivo" de documentos.json.

Uso: python enrich_chunks.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
CHUNKS_FILE = BASE_DIR / "corpus" / "processed" / "chunks.jsonl"
METADATA_FILE = BASE_DIR / "corpus" / "pdf" / "metadata" / "documentos.json"


def load_doc_metadata():
    with open(METADATA_FILE) as f:
        docs = json.load(f)
    return {doc["archivo"]: doc for doc in docs}


def main():
    if not CHUNKS_FILE.exists():
        print(f"  ✗ {CHUNKS_FILE} no existe")
        return 1

    docs = load_doc_metadata()
    print(f"  Documentos cargados: {len(docs)}")

    enriched = []
    missing = set()
    for line in CHUNKS_FILE.open():
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        meta = chunk.get("metadata", {})
        source_file = meta.get("source_file", "")
        doc = docs.get(source_file)
        if not doc:
            missing.add(source_file)
            meta["author"] = "Desconocido"
            meta["institution"] = "Desconocida"
            meta["year"] = None
        else:
            meta["author"] = doc.get("autor", "Desconocido")
            meta["institution"] = doc.get("institucion", "Desconocida")
            meta["year"] = doc.get("anio")
        chunk["metadata"] = meta
        enriched.append(chunk)

    if missing:
        print(f"  ⚠ {len(missing)} source_files sin match en metadata:")
        for s in sorted(missing)[:5]:
            print(f"     - {s}")

    # Backup antes de sobrescribir
    backup = CHUNKS_FILE.with_suffix(".jsonl.bak")
    if not backup.exists():
        import shutil
        shutil.copy2(CHUNKS_FILE, backup)
        print(f"  Backup: {backup}")

    with CHUNKS_FILE.open("w") as f:
        for chunk in enriched:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"  ✓ {len(enriched)} chunks enriquecidos en {CHUNKS_FILE}")
    sample = enriched[0]["metadata"]
    print(f"  Sample metadata: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
