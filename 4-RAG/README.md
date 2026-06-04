# RAG sobre Seguridad Pública en México

Pipeline completo de Retrieval-Augmented Generation sobre un corpus de 23 PDFs institucionales (INEGI, UNODC, SCJN, World Bank, etc.) en 13 categorías, con fine-tuning LoRA sobre Qwen 2.5 0.5B.

## Sobre el corpus y sus autores

El corpus fue **curado por jojo** como parte del Proyecto 4 de IA. Contiene 23 documentos de **16 instituciones** (INEGI, México Evalúa, MUCD, ONC, UNODC, UNAM, ICG, SCJN/ACNUR/CICR, Centro Prodh, Integralia, Fundación Carolina, Cámara de Diputados, Noria Research, World Bank, ICRC, US State Department) publicadas entre 2021 y 2025.

El sistema distingue **tres niveles de atribución**:

1. **El curador** (jojo): reunió los documentos, no produce análisis.
2. **Los autores institucionales** (16): cada documento tiene su立场 editorial.
3. **El corpus en su conjunto**: es plural, no tiene postura única.

Para más detalle, ver [`corpus/processed/MANIFIESTO.md`](corpus/processed/MANIFIESTO.md) y [`corpus/processed/AUTORES.md`](corpus/processed/AUTORES.md).

## Requisitos

- Python 3.11.9
- Ollama (para generación) o transformers + GPU (recomendado)

## Instalación

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
```

## Pipeline

| Paso | Script | Descripción |
|---|---|---|
| Scraper | `src/ingest/scraper.py` | Descarga y valida PDFs |
| Chunking | `src/ingest/chunking.py` | Extrae texto y divide en chunks (con author/institution/year) |
| Embeddings | `src/embeddings/embeddings.py` | Indexa en ChromaDB con all-MiniLM-L6-v2 |
| RAG | `src/rag/rag.py` | Recuperación + generación con Ollama (con MANIFIESTO en system prompt) |
| LoRA | `src/finetuning/finetune.py` | Fine-tuning Qwen 0.5B con 330 ejemplos (300 base + 30 atribución) |
| Integrate | `src/finetuning/integrate.py` | RAG + LoRA con atribución visible |

## Uso

```bash
# Pipeline completo
python run_pipeline.py

# RAG interactivo
python run_pipeline.py --rag

# Consulta directa
python src/rag/rag.py "¿Qué estados tienen más homicidios?"

# Reindexar vectordb (después de cambiar metadata)
python enrich_chunks.py && python reindex_vectordb.py

# Generar dataset para fine-tuning (300 base + 30 atribución)
python datasets/behavioral_dataset.py

# Entrenar LoRA (recomendado: GPU/Colab T4)
python src/finetuning/finetune.py

# RAG + LoRA (modo interactivo)
python src/finetuning/integrate.py -i

# Comparativa before/after con 13 preguntas oficiales
python src/evaluation/compare.py --mode both --save
```

## Notebook

`notebooks/rag_seguridad_mexico.ipynb` — explicación paso a paso del pipeline.

## Corpus

23 PDFs, 7,595 chunks, 13 categorías: homicidios, crimen_organizado, tierra_caliente, desplazamiento_forzado, extorsion, militarizacion, prevencion_social, violencia_urbana, violencia_rural, registro_victimas, cifra_negra, subregistro, impacto_socioeconomico.

## Atribución y atribución al autor

El sistema está diseñado para responder preguntas sobre el autor del corpus (3 niveles). Ver:
- [`docs/AUTOR_RAG.md`](docs/AUTOR_RAG.md) — documentación técnica completa
- `corpus/processed/MANIFIESTO.md` — inyectado en el system prompt
- `corpus/processed/AUTORES.md` — catálogo de los 16 autores
- `datasets/atribution_examples.jsonl` — 30 ejemplos para fine-tuning
- `src/evaluation/official_questions.json` Q11-Q13 — meta-preguntas de validación

## Migración a Google Colab (recomendado)

Para entrenar el LoRA en GPU T4 (6-10 min vs ~1h en CPU), ver `Guias/GUIA_RAG_COLAB.md`. El Paquete #5 con el kit copy-paste está pendiente de generarse (después de esta reestructuración).
