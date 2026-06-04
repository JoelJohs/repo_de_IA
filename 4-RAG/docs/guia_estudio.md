# Guia de estudio — 4-RAG

Documento corto para defender el proyecto en clase. Cubre arquitectura,
pipeline de datos, recuperacion + generacion, evaluacion, fine-tuning
LoRA y preguntas tipo examen.

## 1. Arbol de archivos

```text
4-RAG/
├── run_pipeline.py                  # orquestador de todo el pipeline
├── README.md
├── requirements.txt
├── corpus/
│   ├── pdf/                         # 23 PDFs categorizados (input)
│   │   ├── metadata/documentos.json # indice: titulo, autor, institucion, anio, categoria
│   │   ├── cifra_negra/             # 1 doc
│   │   ├── crimen_organizado/       # 1 doc
│   │   ├── desplazamiento_forzado/ # 2 docs
│   │   ├── extorsion/               # 3 docs
│   │   ├── homicidios/              # 2 docs
│   │   ├── impacto_socioeconomico/  # 5 docs
│   │   ├── militarizacion/          # 2 docs
│   │   ├── prevencion_social/       # 1 doc
│   │   ├── registro_victimas/       # 1 doc
│   │   ├── subregistro/             # 1 doc
│   │   ├── tierra_caliente/         # 2 docs
│   │   ├── violencia_rural/         # 1 doc
│   │   └── violencia_urbana/        # 1 doc
│   └── processed/
│       └── chunks.jsonl             # 7 595 chunks (chunk_size=1000, overlap=200)
├── vectordb/
│   ├── chroma.sqlite3               # ChromaDB con la coleccion 'seguridad_mexico'
│   └── info.json                    # {model, collection, total_chunks, dimension}
├── datasets/
│   ├── behavioral_dataset.py        # genera 300 ejemplos comportamentales
│   └── finetuning.jsonl             # dataset para LoRA (300 ejemplos)
├── models/                          # (vacio: los LoRA adapters viven en src/)
├── src/
│   ├── ingest/
│   │   ├── scraper.py               # descarga + valida PDFs de fuentes oficiales
│   │   └── chunking.py              # RecursiveCharacterTextSplitter sobre PDFs
│   ├── embeddings/
│   │   └── embeddings.py            # all-MiniLM-L6-v2 + Chroma
│   ├── rag/
│   │   └── rag.py                   # retrieval + Ollama, CLI interactivo
│   ├── api/
│   │   ├── main.py                  # FastAPI app (uvicorn entry)
│   │   ├── routes.py                # /health, /query, /query/stream
│   │   └── schemas.py               # Pydantic
│   ├── evaluation/
│   │   ├── test_bank.json           # 20 preguntas con expected_sources/categories
│   │   ├── official_questions.json  # 10 preguntas oficiales (3 niveles)
│   │   ├── evaluate.py              # P@5, R@5, category_match, faithfulness
│   │   ├── compare.py               # before/after: Ollama 7B vs LoRA 0.5B
│   │   └── results.json             # metricas guardadas
│   └── finetuning/
│       ├── generate_dataset.py      # genera rag_dataset.jsonl (50 chunks + test bank)
│       ├── finetune.py              # LoRA sobre Qwen2.5-0.5B-Instruct
│       └── integrate.py             # RAG + LoRA integrado
├── notebooks/
│   └── rag_seguridad_mexico.ipynb   # explicacion paso a paso
└── docs/
    ├── plan_scraper.md              # plan del scraper
    ├── TODO.md                      # TODOs historicos
    └── guia_estudio.md              # este archivo
```

> El venv unificado vive en `../../IA/.venv` (compartido con los
> proyectos 1-game, 2-CNN, 3-RNN). Ver `README.md` para el setup.

## 2. Flujo del pipeline (5 etapas)

```text
                          ┌────────────────────────────────────────┐
                          │ 23 PDFs categorizados (corpus/pdf/*)   │
                          │  INEGI, UNODC, SCJN, World Bank, …     │
                          └──────────────────┬─────────────────────┘
                                             │ src/ingest/scraper.py
                                             │  (descarga + valida + registra)
                                             ▼
                          ┌────────────────────────────────────────┐
                          │ corpus/pdf/metadata/documentos.json    │
                          │  {titulo, autor, institucion, anio,    │
                          │   categoria, archivo, num_paginas}     │
                          └──────────────────┬─────────────────────┘
                                             │ src/ingest/chunking.py
                                             │  PyPDFLoader + RecursiveCharacterTextSplitter
                                             ▼
                          ┌────────────────────────────────────────┐
                          │ corpus/processed/chunks.jsonl          │
                          │   7 595 chunks, ~1 000 chars c/u       │
                          │   {chunk_id, text, metadata{...}}     │
                          └──────────────────┬─────────────────────┘
                                             │ src/embeddings/embeddings.py
                                             │  all-MiniLM-L6-v2 + Chroma (cosine)
                                             ▼
                          ┌────────────────────────────────────────┐
                          │ vectordb/chroma.sqlite3                │
                          │  collection 'seguridad_mexico'         │
                          │  7 595 embeddings de 384 dim           │
                          └──────────────────┬─────────────────────┘
                                             │
                ┌─────────────────┬──────────┴──────────┬─────────────────┐
                ▼                 ▼                     ▼                 ▼
        src/rag/rag.py      src/api/main.py     src/evaluation/    src/finetuning/
        (CLI interactivo)   (FastAPI HTTP)      evaluate.py        finetune.py
                │                 │                     │                 │
                ▼                 ▼                     ▼                 ▼
         terminal/curl      :8000/query          results.json       lora_adapter/
                                                P@5=0.26 R@5=0.53   + integrate.py
                                                cat_match=0.7      (RAG + LoRA)
```

**Una sola consulta** (`query_once` en `src/rag/rag.py:77`) hace, en orden:

1. `load_vectorstore()` — instancia `HuggingFaceEmbeddings` con
   `all-MiniLM-L6-v2` (CPU, normalizado) + `Chroma(persist_directory=...)`.
2. `retriever = vectordb.as_retriever(search_kwargs={"k": 5})` —
   configura el recuperador para devolver los 5 chunks mas similares.
3. `docs = retriever.invoke(question)` — embeddea la pregunta, busca en
   el indice HNSW, devuelve los 5 chunks.
4. `format_docs(docs)` — concatena `doc.page_content` con doble newline.
5. `PROMPT_TEMPLATE.format(context, question)` — mete el contexto + la
   pregunta en el prompt con la instruccion "Responde basandote
   unicamente en el contexto".
6. `ask_ollama_stream(prompt, model)` — `requests.post` a
   `http://localhost:11434/api/generate` con `stream=True`,
   `temperature=0.1`, `num_predict=2048`.
7. Itera `iter_lines()`, concatena tokens, los va impriendo en vivo.

## 3. Responsabilidad por archivo (1 linea)

| Archivo | Que hace |
|---|---|
| `run_pipeline.py` | Orquestador CLI: chunking, embeddings, rag, evaluate, api, finetune, compare, official |
| `src/ingest/scraper.py` | Descarga PDFs de URLs hard-coded, valida `%PDF`, cuenta paginas, registra en `documentos.json` |
| `src/ingest/chunking.py` | `PyPDFLoader` + `RecursiveCharacterTextSplitter(1000, 200)` → `chunks.jsonl` |
| `src/embeddings/embeddings.py` | `all-MiniLM-L6-v2` + `Chroma.from_documents(collection_metadata={"hnsw:space": "cosine"})` |
| `src/rag/rag.py` | `load_vectorstore()` + retriever + `ask_ollama_stream()` + CLI con `/fuentes`, `/modelo`, `/salir` |
| `src/api/main.py` | FastAPI app + CORS middleware + startup que llama `init_api` |
| `src/api/routes.py` | `GET /health`, `POST /query` (LangChain chain: retriever → prompt → Ollama), `POST /query/stream` |
| `src/api/schemas.py` | Pydantic: `QueryRequest`, `QueryResponse`, `HealthResponse`, `FilteredQueryRequest` |
| `src/evaluation/evaluate.py` | `precision@k`, `recall@k`, `category_match`, `faithfulness_score` sobre `test_bank.json` |
| `src/evaluation/compare.py` | Before (Ollama 7B) vs After (Qwen0.5B + LoRA) en las 10 preguntas oficiales |
| `src/evaluation/test_bank.json` | 20 preguntas con `expected_keywords`, `expected_categories`, `expected_sources` |
| `src/evaluation/official_questions.json` | 10 preguntas oficiales con `nivel` (extraccion/sintesis/analisis) |
| `src/finetuning/generate_dataset.py` | Ollama genera Q&A desde chunks muestreados + test bank |
| `src/finetuning/behavioral_dataset.py` | 300 ejemplos sinteticos: citas / neutralidad / socratico / incertidumbre |
| `src/finetuning/finetune.py` | LoRA (r=8, alpha=32) sobre Qwen2.5-0.5B-Instruct con `peft` + `trl` |
| `src/finetuning/integrate.py` | RAG + LoRA: `PeftModel.from_pretrained(base, adapter)` en lugar de Ollama |
| `vectordb/chroma.sqlite3` | Persistencia de ChromaDB (cosine, HNSW) |
| `vectordb/info.json` | `{model, collection, total_chunks: 7595, dimension: 384}` |

## 4. Como se almacenan los datos

- `corpus/pdf/<categoria>/<archivo>.pdf` — los 23 PDFs originales,
  organizados por categoria. **No se versionan** (estan en `.gitignore`).
  Algunos pesan 30-50 MB cada uno.
- `corpus/pdf/metadata/documentos.json` — indice central: lista de 23
  objetos con `titulo`, `autor`, `institucion`, `anio`, `url`,
  `categoria`, `numero_paginas`, `fecha_descarga`, `archivo`. Es la
  **fuente de verdad** de que PDFs hay y como se llaman.
- `corpus/processed/chunks.jsonl` — JSON-line con 7 595 entradas. Cada
  una: `{chunk_id, text, metadata: {category, document_title,
  source_file, pages, chunk_index}}`. **El chunk_id es
  `<source_stem>_<idx:04d>`** (ej. `homicidios_inegi_2025_0000`).
- `vectordb/chroma.sqlite3` — base SQLite de ChromaDB. Contiene los
  embeddings (vector index HNSW) + los documents + los metadatos.
  **Tamano: ~70 MB** (los embeddings son float32, ~700 KB por cada
  1000 chunks × 384 dim × 4 bytes).
- `vectordb/info.json` — `{model, collection, total_chunks, dimension}`.
- `datasets/finetuning.jsonl` — 300 ejemplos comportamentales con
  `behavior` (citas/neutralidad/socratico/incertidumbre), `instruction`
  y `response`. Formato listo para `Trainer` de HF.
- `src/finetuning/lora_adapter/` — output de `finetune.py`: `adapter_*
  .safetensors` + `adapter_config.json`. **Se ignora por git** (es
  output).
- `src/finetuning/dataset/rag_dataset.jsonl` — output de
  `generate_dataset.py`: 50 Q&A + 20 del test bank.

**Punto clave para el profesor:** el `chunk_id` <--> `chunk_index` en
metadata es **estable y deterministico** — se regenera identico cada vez
que se corre el chunking, siempre que el PDF no cambie. Esto permite
que un chunk especifico se pueda citar en cualquier momento y lugar
sin ambiguedad.

### Distribucion de chunks por categoria

| Categoria | Docs | Chunks |
|---|---:|---:|
| desplazamiento_forzado | 2 | 1 579 |
| impacto_socioeconomico | 5 | 1 298 |
| militarizacion | 2 | 1 069 |
| subregistro | 1 | 1 033 |
| registro_victimas | 1 | 934 |
| extorsion | 3 | 562 |
| homicidios | 2 | 290 |
| crimen_organizado | 1 | 266 |
| tierra_caliente | 2 | 255 |
| violencia_rural | 1 | 127 |
| cifra_negra | 1 | 75 |
| violencia_urbana | 1 | 60 |
| prevencion_social | 1 | 47 |
| **Total** | **23** | **7 595** |

> El desbalance (1 579 vs 47) refleja el tamano de los PDFs originales,
> no una decision editorial. Para mejorar la cobertura en categorias
> pequenas, agregar mas documentos a `DIRECT_SOURCES` en `scraper.py`
> y re-correr chunking + embeddings.

## 5. Arquitectura del pipeline (los 3 modulos)

```text
┌──────────────────────────────────────────────────────────────┐
│                    RETRIEVAL (ChromaDB)                       │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│  │ question    │ ─► │ all-MiniLM-L6-v2 │ ─► │ top-5 chunks │ │
│  │ "¿..."     │    │  embed 384d      │    │ (HNSW cos)   │ │
│  └─────────────┘    └──────────────────┘    └──────────────┘ │
└─────────────────────────┬────────────────────────────────────┘
                          │ context (concat page_content)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              GENERATION (Ollama o LoRA)                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ "Eres un asistente experto en seguridad publica en   │    │
│  │  Mexico. Usa SOLO el contexto proporcionado.         │    │
│  │  Contexto: {context}                                 │    │
│  │  Pregunta: {question}                                │    │
│  │  Respuesta:"                                         │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Ollama (qwen2.5:7b-instruct, T=0.1) o               │    │
│  │ Qwen2.5-0.5B-Instruct + LoRA adapter (T=0.1)        │    │
│  └──────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│                  respuesta (stream)                          │
└──────────────────────────────────────────────────────────────┘
```

### 5.1 Embeddings — `all-MiniLM-L6-v2`

- 22 MB de pesos, **384 dim**, max 256 tokens.
- Encoder transformer (MiniLM = "Mini Language Model"), 6 capas.
- **Por que este modelo?** Es el mejor balance tamano/calidad en la
  familia sentence-transformers para CPU. Corre a ~3 000 sentencias/s
  en CPU y mantiene 0.6+ en STS benchmarks.
- `encode_kwargs={"normalize_embeddings": True}` — los vectores salen
  normalizados (norma L2 = 1), asi la similitud coseno se reduce a
  dot product.
- `collection_metadata={"hnsw:space": "cosine"}` — ChromaDB usa
  **HNSW** (Hierarchical Navigable Small World) como indice y mide
  distancia coseno.

### 5.2 Retrieval — ChromaDB + HNSW

- `vectordb.as_retriever(search_kwargs={"k": 5})` — devuelve los
  5 chunks mas similares a la query.
- La busqueda es **aproximada** (no exhaustiva): HNSW sacrifica un
  poco de recall por velocidad (~10-50 ms en CPU para 7 595 vectores).
- El `metadata` de cada chunk viaja con el documento: cuando recuperamos
  un chunk, sabemos de que PDF viene, en que pagina, y a que categoria
  pertenece. Esto es lo que permite citar fuentes y filtrar por
  categoria.

### 5.3 Generation — Ollama o LoRA

**Default (production):** Ollama serve `qwen2.5:7b-instruct`.

```python
payload = {
    "model": "qwen2.5:7b-instruct",
    "prompt": prompt,
    "stream": True,
    "options": {"temperature": 0.1, "num_predict": 2048},
}
response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
```

- `temperature=0.1` (baja) — respuestas mas deterministas/factuales, no
  creativas.
- `num_predict=2048` — limite duro de tokens generados.
- Streaming (`stream=True`) — cada token se manda como una linea JSON
  en NDJSON; el cliente los va concatenando en `full`.

**Alternativa (LoRA):** `Qwen2.5-0.5B-Instruct` + adapter LoRA
entrenado sobre `datasets/finetuning.jsonl`. ~0.5 tok/s en CPU
(mucho mas lento que Ollama 7B con GPU, pero corre **sin** Ollama y
**sin** GPU).

## 6. Servidor FastAPI (`src/api/`)

- **`main.py`** — instancia `FastAPI(title="RAG - Seguridad Publica
  Mexico", version="1.0.0")` + `CORSMiddleware` permisivo (CORS para
  permitir frontends). El `startup` event llama `init_api(...)` que
  pre-carga el vectorstore.
- **`routes.py`** — expone:
  - `GET /health` — devuelve `{status, vectorstore, total_chunks,
    ollama_model}`. Util para health-check.
  - `POST /query` — recibe `{question, k}`, ejecuta el chain LangChain
    completo (retriever → prompt → Ollama), devuelve `{answer,
    sources: [{content, metadata}], latency}`.
  - `POST /query/stream` — `StreamingResponse` con `chain.astream(...)`,
    emite chunks en vivo.
- **LangChain chain** (lineas 61-65 de `routes.py`):
  ```python
  chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
  )
  ```
  LangChain se encarga de orquestar: tomar la question, invocar el
  retriever, formatear el contexto, meterlo en el prompt, llamar a
  Ollama, devolver la respuesta.

## 7. Evaluacion (`src/evaluation/`)

### 7.1 `evaluate.py` — metricas de retrieval

Sobre el `test_bank.json` (20 preguntas con expected_sources y
expected_categories), corre el retriever con `k=5` y mide:

- **`precision_at_k`** — proporcion del top-k que viene de una fuente
  esperada. `0.26` (promedio) significa que ~1.3 de los 5 chunks son
  de la fuente correcta.
- **`recall_at_k`** — proporcion de fuentes esperadas que aparecen
  en el top-k. `0.53` (promedio) significa que recuperamos la mitad
  de las fuentes que el test marcaba como correctas.
- **`category_match`** — proporcion de categorias esperadas que
  aparecen en el top-k. `0.7` (promedio) es **muy bueno** — significa
  que aunque el chunk especifico no sea el correcto, casi siempre
  estamos en la categoria correcta.
- **`faithfulness_score`** — proporcion de oraciones de la respuesta
  que tienen >30% de overlap de palabras (>3 letras) con el contexto.
  Mide si la respuesta esta "anclada" en el corpus.
- **Tiempo de retrieval** — `0.1s` promedio, `1.25s` max. CPU only,
  HNSW se banca 7 595 vectores en milisegundos.

### 7.2 `compare.py` — Before vs After

Corre las 10 preguntas oficiales con dos modos:

- **Before**: Ollama `qwen2.5:7b-instruct` (default, 7B params).
- **After**: Qwen2.5-0.5B-Instruct + LoRA adapter (0.5B params,
  ~17 MB con LoRA).

Mide latencia y 4 "comportamientos":
- `cite_source` — uso de "segun", "de acuerdo con", "reporta", …
- `neutral` — "diferentes perspectivas", "complementarias", …
- `uncertainty` — "no detalla", "fuera del alcance", …
- `socratic` — "antes de responder", "paso a paso", …

### 7.3 Preguntas oficiales (3 niveles)

`official_questions.json` define 10 preguntas en 3 niveles:

| Nivel | Cantidad | Tipo |
|---|---:|---|
| extraccion | 3 | factual directo (entidades con mas homicidios, grupos en Tierra Caliente) |
| sintesis | 4 | cruce de 2-3 fuentes (causas socioeconomicas, contraste estrategias) |
| analisis | 3 | juicio critico (contradicciones ONGs/gobierno, vacios de informacion) |

## 8. Fine-tuning LoRA (`src/finetuning/`)

### 8.1 Datasets

**`generate_dataset.py`** — toma 50 chunks aleatorios del vectorstore,
usa Ollama para generar (pregunta, respuesta condensada), y los junta
con los 20 del test bank. Output: `rag_dataset.jsonl` (~70 registros).

**`behavioral_dataset.py`** — 300 ejemplos **sinteticos** (no
requieren Ollama) divididos en 4 comportamientos x 75:

- `citas` (75) — el modelo cita una institucion y un titulo del corpus
  antes de responder.
- `neutralidad` (75) — el modelo presenta multiples perspectivas sin
  tomar partido.
- `socratico` (75) — el modelo hace preguntas guia antes de responder
  ("Sobre que aspecto te gustaria profundizar?").
- `incertidumbre` (75) — el modelo reconoce explicitamente cuando
  algo esta fuera del corpus.

Distribucion: 85% corpus-relevantes, 15% genericos. Esto fuerza al
LoRA a aprender **como responder en el estilo del corpus**, no
solo **que responder**.

### 8.2 Entrenamiento

`finetune.py` aplica LoRA (Low-Rank Adaptation) sobre
**Qwen2.5-0.5B-Instruct**:

```python
lora_config = LoraConfig(
    r=8,                       # rango bajo de la descomposicion
    lora_alpha=32,             # factor de escala
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

Hiperparametros default:
- `epochs=3`, `batch_size=1`, `gradient_accumulation_steps=4` (efectivo
  batch = 4).
- `lr=2e-4` (tipico de LoRA, 10x mas alto que full fine-tuning).
- `max_length=512`.
- Solo `q_proj, k_proj, v_proj, o_proj` (attention) — el resto del
  modelo queda **congelado**. ~0.5% de los parametros se entrenan
  (≪1M de ~500M).

**Advertencia importante:** el training en CPU es **extremadamente
lento** (~1 hora por epoch en CPU). El codigo detecta CUDA y usa GPU
automaticamente si esta disponible.

### 8.3 Integracion (`integrate.py`)

`LoRARAG.load_model()` carga el modelo base + el adapter:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "src/finetuning/lora_adapter")
model.eval()
```

Modo de uso:

```bash
# consulta unica
python src/finetuning/integrate.py --query "¿Cuantos homicidos hubo en 2024?"

# interactivo
python src/finetuning/integrate.py -i

# comparar contra Ollama
python src/finetuning/integrate.py -i --use-ollama
```

## 9. Preguntas tipo examen (respuesta corta)

**Dataset y corpus**

1. *Cuantos PDFs hay en el corpus y cuantas categorias?* 23 PDFs en 13
   categorias (homicidios, crimen_organizado, tierra_caliente,
   desplazamiento_forzado, extorsion, militarizacion, prevencion_social,
   violencia_urbana, violencia_rural, registro_victimas, cifra_negra,
   subregistro, impacto_socioeconomico).
2. *Que fuentes se usaron?* INEGI, UNODC, SCJN/ACNUR, World Bank,
   ICRC, IEP, U.S. State Department, México Evalúa, ONC, MUCD, ICG,
   Noria Research, Integralia, Centro Prodh, Fundación Carolina, UNAM,
   Cámara de Diputados.
3. *Cuantos chunks se generan y por que?* 7 595. Cada PDF pasa por
   `PyPDFLoader` (1 doc por pagina) y luego por
   `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
4. *Cual es la diferencia entre chunk_size y chunk_overlap?* `chunk_size`
   = 1 000 chars por chunk. `chunk_overlap` = 200 chars compartidos
   entre chunks consecutivos para que el retriever no pierda contexto
   en los bordes.
5. *Por que `RecursiveCharacterTextSplitter` y no otro?* Es el default
   de LangChain. Intenta cortar primero en `\n\n`, despues `\n`,
   despues `. `, etc. — asi no parte oraciones a la mitad cuando puede
   evitarlo.
6. *Como se valida un PDF al descargarlo?* En `scraper.py`: verifica
   que el contenido empiece con `b"%PDF"`, que tenga >= 3 paginas, y
   registra `numero_paginas` y `fecha_descarga` en `documentos.json`.

**Embeddings y vector store**

7. *Que modelo de embeddings se usa?* `sentence-transformers/
   all-MiniLM-L6-v2`. 22 MB, 384 dim, max 256 tokens.
8. *Por que este modelo?* Es el mejor trade-off tamano/calidad en la
   familia sentence-transformers para correr en CPU. 3 000
   sentencias/s, >0.6 en STS.
9. *Que base de datos vectorial se usa?* ChromaDB con indice HNSW
   (Hierarchical Navigable Small World).
10. *Que distancia usa ChromaDB?* Coseno (`hnsw:space: cosine` en
    `collection_metadata`). Por eso normalizamos embeddings
    (`normalize_embeddings: True`) — dot product ≡ coseno.
11. *Cuantos embeddings hay y que dimension tienen?* 7 595 vectores
    de 384 dim. Persistidos en `vectordb/chroma.sqlite3` (~70 MB).
12. *Por que no usar FAISS o Pinecone?* ChromaDB ya viene integrado
    con LangChain, es zero-config (SQLite en disco), y soporta
    metadata filtering nativamente. FAISS seria mas rapido pero
    requiere mas plumbing. Pinecone es cloud-only.
13. *Cuanto tarda en embeddear los 7 595 chunks?* ~5-10 min en CPU
    (cuenta como el startup de la app). Una vez indexado, una query
    tarda ~10-50 ms.

**Recuperacion y generacion**

14. *Que hace `vectordb.as_retriever(search_kwargs={"k": 5})`?*
    Configura un `Retriever` que devuelve los 5 chunks mas similares
    a la query segun la distancia coseno del HNSW.
15. *Por que k=5 y no 10?* Empiricamente k=5 da el mejor balance
    recall/latencia para este corpus. El `evaluate.py` mide esto y
    confirma que P@5=0.26, R@5=0.53.
16. *Que temperatura tiene Ollama?* `temperature=0.1` (baja). El
    objetivo es factual/determinista, no creativo.
17. *Que hace `num_predict=2048`?* Limita a 2 048 tokens la respuesta
    generada. Suficiente para respuestas de 5-6 parrafos.
18. *Por que streaming (`stream=True`)?* Para no esperar 30-60 s a que
    Ollama genere toda la respuesta. Cada token se va emitiendo y
    mostrando en vivo en el CLI.
19. *Cual es el flujo completo de una query?* (1) `load_vectorstore()`
    → (2) `retriever.invoke(question)` → top-5 chunks → (3)
    `format_docs(docs)` → context string → (4) `PROMPT_TEMPLATE.format(
    context, question)` → prompt completo → (5)
    `requests.post(OLLAMA_URL, ...)` con `stream=True` → (6) iterar
    `iter_lines()`, acumular tokens → (7) `print(token, end="")` en vivo.
20. *Cual es la diferencia entre `rag.py` y la API en `api/routes.py`?*
    `rag.py` es CLI interactivo (loop `while True` + `input()`); la
    API usa LangChain chain + `StreamingResponse`. Mismo Ollama, misma
    collection, distinta interfaz.

**Evaluacion**

21. *Que metricas mide `evaluate.py`?* `precision_at_5`, `recall_at_5`,
    `category_match`, `faithfulness_score`, retrieval time.
22. *Que significa P@5=0.26 en este sistema?* De los 5 chunks
    recuperados, ~1.3 son de la fuente correcta. Es bajo, pero
    **R@5=0.53** y **category_match=0.7** muestran que el sistema
    recupera la categoria correcta aunque no siempre el PDF exacto.
23. *Que mide `category_match` y por que es importante?* Proporcion de
    categorias esperadas que aparecen en el top-k. 0.7 es muy bueno
    — significa que aunque fallen los fuentes especificas, casi
    siempre estamos en el area correcta del corpus. Esto le da al
    LLM contexto relevante aunque el chunk exacto no este.
24. *Que hace `faithfulness_score`?* Divide la respuesta en oraciones
    y mide la proporcion que tienen >30% de overlap de palabras
    significativas (>3 letras) con el contexto. Mide si la respuesta
    esta "anclada" en el corpus vs. inventada.
25. *Cuantas preguntas tiene `test_bank.json`?* 20. Cada una con
    `expected_keywords`, `expected_categories`, `expected_sources`.
26. *Cuantas preguntas oficiales hay y en cuantos niveles?* 10
    preguntas: 3 de extraccion, 4 de sintesis, 3 de analisis.
27. *Que hace `compare.py`?* Ejecuta las 10 preguntas oficiales con
    dos modos: before (Ollama 7B) y after (Qwen 0.5B + LoRA).
    Mide latencia y 4 "comportamientos" (citas, neutral,
    incertidumbre, socratico).
28. *Por que importa el LoRA si Ollama 7B es mas grande?* El LoRA corre
    **sin Ollama y sin GPU** — util en maquinas restringidas. Ademas
    aprende el **estilo** de respuesta que queremos (citas,
    neutralidad), no solo el contenido.

**Fine-tuning**

29. *Que es LoRA?* Low-Rank Adaptation: en vez de actualizar todos
    los pesos del modelo, agrega matrices pequenas (rango r) en
    paralelo a las proyecciones de atencion. Entrena ~0.5% de los
    parametros con calidad similar a full fine-tuning.
30. *Que parametros de LoRA se usan?* `r=8`, `alpha=32`, `dropout=0.1`,
    `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`. Solo
    attention, MLP queda congelado.
31. *Cuantos parametros entrenables tiene el LoRA?* Para Qwen 0.5B con
    r=8 en q/k/v/o: ~0.5-1M parametros (vs 500M del modelo completo).
    El adapter pesa ~4-8 MB en disco.
32. *Cual es el modelo base?* `Qwen/Qwen2.5-0.5B-Instruct`. Es la
    version "Instruct" (fine-tuned para seguir instrucciones) de
    Qwen 2.5 con 0.5B parametros.
33. *Por que Qwen 0.5B y no 7B?* Para que el LoRA se pueda entrenar
    en CPU (los 7B no caben en RAM tipica para training). El
    trade-off es: 14x mas chico, ~14x menos capaz, pero suficiente
    para responder con el contexto correcto.
34. *Que es `datasets/finetuning.jsonl`?* 300 ejemplos
    comportamentales (75 de cada uno: citas, neutralidad, socratico,
    incertidumbre) generados **sin** Ollama — son plantillas con
    documentos reales del corpus.
35. *Que hace `generate_dataset.py`?* Usa Ollama para generar pares
    (pregunta, respuesta condensada) sobre 50 chunks aleatorios +
    20 del test bank. ~70 registros, output
    `src/finetuning/dataset/rag_dataset.jsonl`.
36. *Que hace `integrate.py`?* Une el RAG con el LoRA: carga el
    modelo base + adapter via `PeftModel.from_pretrained`, lo usa
    como generador en lugar de Ollama. Modo `--use-ollama` para
    comparar.
37. *Por que CPU es "extremadamente lento" para LoRA training?*
    `peft` no tiene cuantizacion automatica habilitada (la variable
    `BitsAndBytesConfig` esta importada pero no usada). Sin
    quantizacion, el modelo completo vive en RAM en float32 (~2 GB
    para 0.5B) y cada backward pass recorre 500M parametros.
38. *Que pasa si corro finetune.py sin el dataset?* Falla
    inmediatamente con `FileNotFoundError` en `load_dataset_jsonl()`.
    Hay que correr `behavioral_dataset.py` primero para generar
    `datasets/finetuning.jsonl`.

**API**

39. *Cual es la URL de la API?* `http://localhost:8000`. Endpoints:
    `GET /health`, `POST /query`, `POST /query/stream`.
40. *Que devuelve `/health`?*
    `{status, vectorstore, total_chunks, ollama_model}`. Util para
    readiness check.
41. *Cual es la diferencia entre `/query` y `/query/stream`?* `/query`
    espera a que Ollama genere toda la respuesta y devuelve JSON con
    `{answer, sources, latency}`. `/query/stream` emite la respuesta
    en chunks via `StreamingResponse` con `media_type="text/plain"`.
42. *Como se inicia la API?*
    `uvicorn 4-RAG/src/api/main:app --host 0.0.0.0 --port 8000`. O
    `./scripts/rag.fish --api`.
43. *Por que CORS abierto (`allow_origins=["*"]`)?* Para que cualquier
    frontend (incluso en otro puerto) pueda consumirla durante el
    desarrollo. En produccion se restringe a dominios especificos.

**Integracion y flujo end-to-end**

44. *Cual es el flujo end-to-end de "Hola, ¿cuantos homicidos hubo en
    2024?"?* (1) CLI: `./scripts/rag.fish` → (2) `python -m src.rag.rag
    -i` → (3) `load_vectorstore()` carga ChromaDB (5-10 s) → (4)
    usuario escribe la pregunta → (5) `retriever.invoke("...")`
    busca los 5 chunks mas relevantes → (6) los muestra con
    `show_sources(docs)` → (7) mete contexto + pregunta en
    `PROMPT_TEMPLATE` → (8) `requests.post(OLLAMA_URL, ...)` con
    stream → (9) imprime tokens en vivo → (10) usuario puede ver
    las fuentes con `/fuentes`.
45. *Que pasa si Ollama no esta corriendo?* `requests.post` falla con
    `ConnectionError` despues del timeout. El RAG no funciona hasta
    que `ollama serve` este activo. Workaround: usar
    `integrate.py` (LoRA, no requiere Ollama).
46. *Que pasa si la pregunta no esta en el corpus?* El retriever
    devuelve los 5 chunks mas cercanos semanticamente, que pueden o
    no ser relevantes. El prompt instruye a Ollama "Si el contexto
    no contiene la informacion, indicálo claramente", asi que
    idealmente la respuesta sera "el corpus no contiene…". El LoRA
    re-entrenado aprende explicitamente este comportamiento
    (categoria "incertidumbre" en `behavioral_dataset.py`).
47. *Por que se normalizan embeddings?* Para que dot product ≡
    similitud coseno (no hace falta restar normas, los embeddings
    viven en una hiperesfera unitaria). ChromaDB con `hnsw:space:
    cosine` espera vectores normalizados.
48. *Que ventaja tiene usar LangChain chain vs `requests.post`
    manual?* LangChain abstrae el flujo retriever → prompt → llm y
    permite reusar la misma logica en el CLI (`rag.py`) y en la API
    (`routes.py`). Ademas, `.astream()` da streaming out-of-the-box
    sin tener que parsear NDJSON.
49. *Que metricas finales tiene el proyecto?* P@5=0.26, R@5=0.53,
    category_match=0.7, retrieval time avg=0.1s, max=1.25s, 7 595
    chunks indexados, 20 preguntas en test bank, 10 preguntas
    oficiales en 3 niveles.
50. *Cual es la diferencia entre `evaluate.py` y `compare.py`?*
    `evaluate.py` mide el retrieval solo (sin generacion), sirve
    para tunnear k y la estrategia. `compare.py` mide el sistema
    end-to-end (retrieval + Ollama vs retrieval + LoRA) y compara
    comportamientos de la respuesta generada.
