# RAG sobre Seguridad Pública en México

Pipeline completo de Retrieval-Augmented Generation sobre un corpus de 23 PDFs institucionales (INEGI, UNODC, SCJN, World Bank, etc.) en 13 categorías.

## Requisitos

- Python 3.11.9
- Ollama (para generación)

## Instalación

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
```

## Pipeline

| Paso | Script | Descripción |
|---|---|---|
| Scraper | `src/ingest/scraper.py` | Descarga y valida PDFs |
| Chunking | `src/ingest/chunking.py` | Extrae texto y divide en chunks |
| Embeddings | `src/embeddings/embeddings.py` | Indexa en ChromaDB con all-MiniLM-L6-v2 |
| RAG | `src/rag/rag.py` | Recuperación + generación con Ollama |

## Uso

```bash
# Pipeline completo
python run_pipeline.py

# RAG interactivo
python run_pipeline.py --rag

# Consulta directa
python src/rag/rag.py "¿Qué estados tienen más homicidios?"

# Reindexar vectordb
python run_pipeline.py --rebuild
```

## Notebook

`notebooks/rag_seguridad_mexico.ipynb` — explicación paso a paso del pipeline.

## Corpus

23 PDFs, 7,595 chunks, 13 categorías: homicidios, crimen_organizado, tierra_caliente, desplazamiento_forzado, extorsion, militarizacion, prevencion_social, violencia_urbana, violencia_rural, registro_victimas, cifra_negra, subregistro, impacto_socioeconomico.
