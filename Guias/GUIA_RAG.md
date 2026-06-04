# Guia de defensa oral - Proyecto 4 (RAG + LoRA)

> Documento de estudio para la defensa oral del Proyecto 4.
> Mismo formato que `GUIA_CNN.md` y `GUIA_RNN.md`, pero adaptado al
> **RAG (Retrieval-Augmented Generation) sobre 23 PDFs de seguridad
> publica en Mexico**, con fine-tuning LoRA sobre Qwen2.5-0.5B.
>
> **TL;DR**: tenes un asistente que responde preguntas sobre violencia
> en Mexico **solo con informacion real** de 23 PDFs institucionales
> (INEGI, UNODC, World Bank, etc.). Recupera los 5 parrafos mas
> relevantes con embeddings, se los pasa a un LLM, y contesta citando
> fuentes. Ademas hiciste un fine-tuning LoRA para que el modelo chico
> (0.5B) aprenda a responder en el estilo del corpus.

---

## 0. Resumen ejecutivo (1 carilla)

Las 15 preguntas que el profe **seguro** te tira, con respuestas de
30 segundos cada una.

| # | Pregunta | Respuesta corta |
|---|---|---|
| 1 | *Que es RAG?* | Retrieval-Augmented Generation: en vez de que el LLM "alucine" la respuesta, primero **busca** los parrafos relevantes en una base vectorial y despues **genera** la respuesta con esos parrafos como contexto. |
| 2 | *Por que ChromaDB y no FAISS/Pinecone?* | ChromaDB ya viene integrado con LangChain, persiste en SQLite local, soporta metadata filtering nativo. FAISS seria mas rapido pero requiere mas plumbing. Pinecone es cloud-only. |
| 3 | *Por que `all-MiniLM-L6-v2`?* | Mejor trade-off tamano/calidad para correr en CPU. 22 MB, 384 dim, 6 capas transformer, ~3000 sent/s en CPU, >0.6 en STS benchmarks. |
| 4 | *Que hace HNSW?* | Hierarchical Navigable Small World: indice aproximado para busqueda de vecinos mas cercanos en espacios de alta dimension. Sacrifica un poco de recall por velocidad (~10-50 ms en CPU para 7595 vectores). |
| 5 | *Por que distancia coseno y no L2?* | Los embeddings estan normalizados (norma L2=1), asi coseno = dot product. Coseno mide el **angulo** entre vectores (similitud semantica), L2 mide la **distancia geometrica** (sensible a la magnitud). |
| 6 | *Que es LoRA?* | Low-Rank Adaptation: en vez de actualizar todos los 500M pesos del modelo, agrega matrices pequenas (rango r=8) en paralelo a las proyecciones de atencion. Entrena ~0.5% de los parametros con calidad similar. |
| 7 | *Por que Qwen 0.5B y no 7B para LoRA?* | Los 7B no caben en RAM tipica para training (~14 GB en fp32). 0.5B cabe en ~2 GB y corre en CPU. Trade-off: 14x mas chico, ~14x menos capaz, pero suficiente para responder con el contexto correcto. |
| 8 | *Que mide P@5=0.26?* | De los 5 chunks que recuperamos, en promedio ~1.3 vienen de la fuente esperada. Es bajo, pero R@5=0.53 y **category_match=0.7** muestran que casi siempre caemos en la categoria correcta aunque no sea el PDF exacto. |
| 9 | *Que pasa si Ollama no esta corriendo?* | `requests.post` a `localhost:11434` falla con `ConnectionError`. Soluciones: (a) arrancar Ollama, (b) usar `integrate.py` con LoRA (no requiere Ollama). |
| 10 | *Que diferencia tiene con la CNN/RNN?* | La CNN clasifica imagenes, la RNN genera texto, el RAG **busca informacion** en una base de conocimiento y la usa para responder. Es un sistema, no un modelo. |
| 11 | *Cuantos documentos y chunks hay?* | 23 PDFs en 13 categorias. Con `chunk_size=1000` y `chunk_overlap=200` se generan 7.595 chunks. |
| 12 | *Por que chunk_size=1000 y overlap=200?* | Suficiente contexto para que el LLM entienda la idea (~200 tokens). Overlap=200 evita perder contexto en los bordes entre chunks. |
| 13 | *Que es el dataset comportamental de 300 ejemplos?* | 4 estilos x 75: **citas** (responde citando institucion), **neutralidad** (presenta multiples perspectivas), **socratico** (hace preguntas guia antes de responder), **incertidumbre** (reconoce cuando algo no esta en el corpus). 85% corpus-relevantes, 15% genericos. |
| 14 | *Por que `category_match=0.7` importa mas que P@5?* | Porque aunque el chunk especifico no sea el correcto, si caemos en la categoria correcta, el LLM tiene contexto relevante para generar una respuesta util. El LLM **rellena** lo que falta. |
| 15 | *Cual es el flujo completo de una query?* | (1) `load_vectorstore()` carga ChromaDB → (2) `retriever.invoke(pregunta)` busca top-5 chunks similares → (3) `format_docs()` concatena el texto → (4) `PROMPT_TEMPLATE.format()` mete contexto+pregunta en el prompt → (5) `requests.post(OLLAMA_URL)` con stream → (6) el LLM genera token por token → (7) se imprimen en vivo. |

---

## 1. El viaje end-to-end (5 etapas)

Diagrama de las **5 etapas** del pipeline.

```
   ┌──────────────────────────────────────────────────────────┐
   │  23 PDFs en corpus/pdf/<categoria>/                      │
   │  INEGI, UNODC, SCJN, World Bank, MUCD, ONC, …           │
   │  Algunos pesan 30-50 MB cada uno                         │
   └─────────────────────┬────────────────────────────────────┘
                         │ src/ingest/scraper.py (F1)
                         │ descarga + valida %PDF + ≥3 paginas
                         │ genera documentos.json
                         ▼
   ┌──────────────────────────────────────────────────────────┐
   │  corpus/pdf/metadata/documentos.json                     │
   │  23 entradas: {titulo, autor, institucion, anio,         │
   │  categoria, archivo, numero_paginas, fecha_descarga}     │
   └─────────────────────┬────────────────────────────────────┘
                         │ src/ingest/chunking.py (F2)
                         │ PyPDFLoader + RecursiveTextSplitter
                         │ (chunk_size=1000, overlap=200)
                         ▼
   ┌──────────────────────────────────────────────────────────┐
   │  corpus/processed/chunks.jsonl                            │
   │  7.595 chunks, ~1000 chars c/u                           │
   │  {chunk_id, text, metadata: {category, document_title,  │
   │  source_file, pages, chunk_index}}                        │
   └─────────────────────┬────────────────────────────────────┘
                         │ src/embeddings/embeddings.py (F3)
                         │ all-MiniLM-L6-v2 → vectores 384-d
                         │ persistidos en ChromaDB (HNSW coseno)
                         ▼
   ┌──────────────────────────────────────────────────────────┐
   │  vectordb/chroma.sqlite3                                 │
   │  collection 'seguridad_mexico'                           │
   │  7.595 embeddings de 384 dim (~70 MB en disco)           │
   │  + info.json con metadata                                │
   └─────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────┼────────────┬──────────────┐
            ▼            ▼            ▼              ▼
       rag.py      api/main.py   evaluate.py    finetune.py
       (CLI)       (FastAPI)     (P@5, R@5)     (LoRA r=8)
            │            │            │              │
            ▼            ▼            ▼              ▼
        terminal      :8000       results.json   lora_adapter/
        + /fuentes    /query      R@5=0.53       + integrate.py
        + /modelo     /stream     cat=0.7        (RAG+LoRA)
```

### Tamanos clave (de `docs/guia_estudio.md`)

| Archivo | Tamano |
|---|---|
| `corpus/pdf/*/*.pdf` (23 PDFs) | ~300-400 MB total (no versionado) |
| `corpus/processed/chunks.jsonl` | ~17 MB (7.595 chunks × ~2 KB) |
| `vectordb/chroma.sqlite3` | ~70 MB (embeddings float32: 7595 × 384 × 4B) |
| `datasets/finetuning.jsonl` | ~80 KB (300 ejemplos) |
| `src/finetuning/lora_adapter/` | ~5-8 MB (solo el adapter) |
| `models/all-MiniLM-L6-v2/` (cache) | ~90 MB (descargado por HF la primera vez) |

### Distribucion de chunks por categoria

| Categoria | Docs | Chunks |
|---|---:|---:|
| desplazamiento_forzado | 2 | 1.579 |
| impacto_socioeconomico | 5 | 1.298 |
| militarizacion | 2 | 1.069 |
| subregistro | 1 | 1.033 |
| registro_victimas | 1 | 934 |
| extorsion | 3 | 562 |
| homicidios | 2 | 290 |
| crimen_organizado | 1 | 266 |
| tierra_caliente | 2 | 255 |
| violencia_rural | 1 | 127 |
| cifra_negra | 1 | 75 |
| violencia_urbana | 1 | 60 |
| prevencion_social | 1 | 47 |
| **Total** | **23** | **7.595** |

El desbalance (1.579 vs 47) refleja el tamano de los PDFs originales,
no una decision editorial. Categorias chicas como `prevencion_social`
podrian ampliarse agregando mas docs a `DIRECT_SOURCES` en
`scraper.py:30-165` y re-corriendo chunking + embeddings.

---

## 2. Entradas y salidas (con correccion de misconceptions)

### 2.1 El input NO es una imagen ni texto crudo

A diferencia de CNN/RNN, el input del RAG es **una pregunta en lenguaje
natural**:

| | CNN (proy 2) | RNN (proy 3) | RAG (proy 4) |
|---|---|---|---|
| Input | `numpy.ndarray` shape `(32, 32, 3)` | 32 indices de char | string libre: `"¿Cuantos homicidios hubo en 2024?"` |
| Que representa | Pixeles RGB | 32 chars consecutivos | Pregunta en espanol |
| Procesamiento | 1 sola pasada | 32 pasos recurrentes | (1) embed → (2) retrieve → (3) generate |

### 2.2 El output NO es una clase

Tambien a diferencia de la CNN, el output **no es una de 5 clases**. Es
una **respuesta en lenguaje natural + las fuentes**:

```json
{
  "answer": "Segun INEGI, en 2024 se registraron aproximadamente 30.000 "
            "homicidios dolosos en Mexico, con una tasa de 23.5 por cada "
            "100.000 habitantes.",
  "sources": [
    {
      "content": "INEGI: Defunciones por Homicidio 2024. Resultados completos...",
      "metadata": {
        "category": "homicidios",
        "document_title": "Defunciones por Homicidio 2024 - Resultados completos",
        "source_file": "homicidios/homicidios_inegi_2025.pdf",
        "pages": "5-7",
        "chunk_index": 42
      }
    },
    /* ... 4 fuentes mas */
  ],
  "latency": 12.3
}
```

### 2.3 El "modelo" son 2 (o 3) partes independientes

Esto es lo que confunde al profe. El "modelo RAG" **no es un solo
archivo**, son **componentes separados que se enchufan**:

| Componente | Tamano | Donde vive | Que hace |
|---|---|---|---|
| **Embedder** | 22 MB | `~/.cache/huggingface/` (cache de HF) | Convierte texto → vector 384-d |
| **Vectordb** | ~70 MB | `vectordb/chroma.sqlite3` | Almacena vectores + metadatos de los 7.595 chunks |
| **Generador** | ~5 GB (RAM) | Ollama corriendo o cargado en `integrate.py` | Convierte (contexto + pregunta) → respuesta |

Cuando cambias el corpus, solo re-indexas el vectordb. Cuando cambias
el generador (Ollama 7B ↔ LoRA 0.5B), el embedder y el vectordb siguen
sirviendo. **El "modelo" no se reentrena, se reensambla.**

### 2.4 Tres lugares de almacenamiento, NO se mezclan

| Carpeta | Que hay | Regenerable? | Versionado? |
|---|---|---|---|
| `corpus/pdf/*/*.pdf` | Los PDFs originales | No (es la fuente) | No (estan en `.gitignore`) |
| `corpus/processed/chunks.jsonl` | Los 7.595 chunks | **Si** (chunking es deterministico) | No (regenerable) |
| `vectordb/chroma.sqlite3` | Los embeddings + metadata | **Si** (embeddings es deterministico) | No (regenerable) |

> **Regla mnemotecnica**: el PDF es la **materia prima**, los chunks
> son el **producto intermedio**, el vectordb es el **producto
> terminado**. Si perdes cualquiera de los 3, podes regenerar todo a
> partir del PDF.

---

## 3. Los 3 modulos del RAG

### 3.1 Modulo (a): Embedding con `all-MiniLM-L6-v2`

| | Detalle |
|---|---|
| Modelo | `sentence-transformers/all-MiniLM-L6-v2` |
| Tamano | 22 MB |
| Dim de salida | 384 |
| Capas | 6 (transformer encoder) |
| Tokens max | 256 |
| Velocidad CPU | ~3.000 sentencias/s |
| Benchmark STS | > 0.6 Spearman |
| Dispositivo | CPU (no necesita GPU) |

**Como se carga** (`embeddings.py:45-49`):

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu", "local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)
```

**Por que este modelo:**
- Es el mejor trade-off tamano/calidad en la familia
  sentence-transformers para CPU.
- MiniLM = "Mini Language Model", es una version destilada de BERT
  para embeddings.
- El `encode_kwargs={"normalize_embeddings": True}` hace que los
  vectores salgan con norma L2 = 1 → coseno = dot product.

### 3.2 Modulo (b): Retrieval con ChromaDB + HNSW

| | Detalle |
|---|---|
| Vector DB | ChromaDB 0.5+ |
| Indice | HNSW (Hierarchical Navigable Small World) |
| Distancia | Coseno (`hnsw:space: cosine`) |
| Persistencia | SQLite local en `vectordb/chroma.sqlite3` |
| Busqueda | Aproximada (~10-50 ms para 7.595 vectores en CPU) |
| Metadata filtering | Si, nativo |

**Como se configura** (`embeddings.py:56-62`):

```python
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=str(VECTORDB_DIR),
    collection_name=COLLECTION_NAME,
    collection_metadata={"hnsw:space": "cosine"},
)
```

**HNSW en una oracion:** construye un grafo multi-nivel donde cada nodo
conecta a sus vecinos mas cercanos. La busqueda empieza por el nivel
alto (pocos nodos, saltos grandes) y baja refinando hasta el nivel
cero (busqueda exhaustiva local). Resultado: **logaritmico** en vez de
lineal.

### 3.3 Modulo (c): Generation con Ollama o LoRA

**Default (produccion):** Ollama serve `qwen2.5:7b-instruct`.

| | Ollama 7B | LoRA 0.5B |
|---|---|---|
| Parametros | 7B (~4 GB en RAM) | 0.5B + adapter (~1.5 GB) |
| Velocidad CPU | ~0.5 tok/s (lento) | ~0.5 tok/s (similar) |
| Velocidad GPU (Colab T4) | ~30 tok/s (rapido) | ~10 tok/s (rapido) |
| Calidad de respuesta | Alta | Media |
| Requiere Ollama | Si | No |
| Personalizable | No (modelo cerrado) | Si (adapter en `lora_adapter/`) |

**Como se invoca Ollama** (`rag.py:54-74`):

```python
payload = {
    "model": "qwen2.5:7b-instruct",
    "prompt": prompt,
    "stream": True,
    "options": {
        "temperature": 0.1,      # baja: factual, no creativo
        "num_predict": 2048,     # limite duro de tokens
    },
}
response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
```

`temperature=0.1` (baja) garantiza respuestas mas deterministas/factuales
— queremos que cite el corpus, no que invente.

### 3.4 Tabla resumen de los 3 modulos

| Modulo | Input | Output | Latencia CPU | Dependencia externa |
|---|---|---|---|---|
| **Embedding** | string (pregunta o chunk) | vector 384-d | ~1-5 ms | all-MiniLM-L6-v2 (HF cache) |
| **Retrieval** | vector 384-d | top-5 chunks (texto + metadata) | ~10-50 ms | ChromaDB local |
| **Generation** | (contexto + pregunta) | string (respuesta) | ~30-60 s | Ollama corriendo en :11434 |

---

## 4. Recorrido archivo por archivo

### 4.1 `run_pipeline.py` (orquestador CLI)

El punto de entrada unico. Flags: `--rag`, `--api`, `--evaluate`,
`--finetune`, `--compare`, `--official`, `--rebuild`, `--skip-chunking`,
`--skip-embeddings`.

| Lineas | Que hace |
|---|---|
| `27-28` | `VENV_PYTHON = BASE_DIR.parent / ".venv" / "bin" / "python"` — fuerza el venv unificado |
| `31-40` | `run_script()` wrapper de `subprocess.run()` con manejo de errores |
| `67-72` | Ejecuta chunking + embeddings (los dos pasos obligatorios) |
| `74-75` | `--rag` → llama a `rag.py -i` (modo interactivo) |
| `80-86` | `--api` → arranca uvicorn en el puerto 8000 |
| `94-95` | `--finetune` → llama a `finetune.py` (LoRA training) |

### 4.2 `src/ingest/scraper.py` (F1: descarga de PDFs)

Descarga los 23 PDFs de URLs hard-coded, valida que sean PDFs reales
y tengan >= 3 paginas.

| Lineas | Que hace |
|---|---|
| `21-27` | `CATEGORIES` - las 13 categorias tematicas |
| `30-165` | `DIRECT_SOURCES` - lista hard-coded de 23 PDFs con URL, titulo, autor, institucion, anio, categoria |
| `178-179` | `is_valid_pdf()` - verifica que el contenido empiece con `b"%PDF"` |
| `181-188` | `count_pages()` - usa PyMuPDF (`fitz`) para contar paginas |
| `202-245` | `download_source()` - descarga, valida, escribe el PDF, agrega metadata |
| `247-296` | `run()` - itera DIRECT_SOURCES, deduplica por URL, muestra barras de progreso por categoria |

**Validaciones:**
- Cabecera magica `%PDF` (linea 219)
- Minimo 3 paginas (linea 224)
- Registra `numero_paginas` y `fecha_descarga` en `documentos.json`

### 4.3 `src/ingest/chunking.py` (F2: extraccion + division)

Toma cada PDF, extrae texto con PyPDFLoader, divide con
RecursiveCharacterTextSplitter.

| Lineas | Que hace |
|---|---|
| `19-20` | `CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 200` |
| `22-29` | `SEPARATORS` - jerarquia: doble newline, newline, oracion, palabra, char |
| `37-78` | `chunk_document()` - carga el PDF, divide, agrega metadata (category, document_title, source_file, pages, chunk_index) |
| `67-68` | `chunk_id` = `<source_stem>_<idx:04d>` (ej. `homicidios_inegi_2025_0000`) — **estable y deterministico** |
| `81-114` | `run()` - itera todos los docs del metadata, acumula chunks, los escribe a `chunks.jsonl` |

**El truco de RecursiveTextSplitter:** intenta cortar primero en
`\n\n` (entre parrafos), despues en `\n` (entre lineas), despues en
`. ` (entre oraciones), etc. Asi **no parte las oraciones a la mitad**
cuando puede evitarlo.

### 4.4 `src/embeddings/embeddings.py` (F3: vectorizacion + Chroma)

| Lineas | Que hace |
|---|---|
| `19-20` | `MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"`, `COLLECTION_NAME = "seguridad_mexico"` |
| `23-32` | `load_chunks_as_documents()` - lee `chunks.jsonl`, construye objetos `Document` con metadata |
| `35-73` | `build_vectorstore()` - instancia el embedder, conecta a ChromaDB, agrega `collection_metadata={"hnsw:space": "cosine"}` |
| `52` | Si existe el vectordb, lo borra con `shutil.rmtree(VECTORDB_DIR)` antes de reindexar |
| `66-73` | Guarda `info.json` con `{model, collection, total_chunks, dimension: 384}` |
| `76-86` | `get_vectorstore()` - solo abre el vectordb existente (sin reindexar) |

**Diferencia clave:**
- `build_vectorstore(rebuild=True)` = reindexar todo desde `chunks.jsonl` (~5-10 min)
- `get_vectorstore()` = solo abrir el ChromaDB ya indexado (~5-10 s)

### 4.5 `src/rag/rag.py` (F4: CLI interactivo)

El usuario final lo usa: escribe preguntas, recibe respuestas con
fuentes.

| Lineas | Que hace |
|---|---|
| `20-28` | `PROMPT_TEMPLATE` - estructura: "Eres un asistente experto en seguridad publica en Mexico. Usa SOLO el contexto. Contexto: {context}. Pregunta: {question}. Respuesta:" |
| `31-34` | `SKIP_PREFIXES` - frases que Ollama a veces antepone ("aqui esta", "claro", "por supuesto", ...) y que limpiamos |
| `37-38` | `format_docs()` - concatena los page_content de los chunks con doble newline |
| `41-51` | `load_vectorstore()` - helper para abrir el ChromaDB |
| `54-74` | `ask_ollama_stream()` - POST a `:11434/api/generate` con `stream=True`, `temperature=0.1`, `num_predict=2048` |
| `77-97` | `query_once()` - una sola consulta: load vectorstore → retrieve top-5 → format context → call Ollama → print stream |
| `100-108` | `show_sources()` - muestra las 5 fuentes con su archivo, paginas, categoria y preview del texto |
| `111-118` | `strip_prefix()` - limpia los prefijos tipicos que Ollama agrega |
| `121-216` | `run_cli()` - loop interactivo con comandos especiales: `/fuentes`, `/modelo NOMBRE`, `/salir` |

### 4.6 `src/api/main.py` + `src/api/routes.py` (F5: FastAPI HTTP)

| Archivo | Lineas | Que hace |
|---|---|---|
| `main.py` | `14-18` | Crea la app FastAPI con titulo y version |
| `main.py` | `20-26` | Agrega `CORSMiddleware` permisivo (CORS para cualquier frontend) |
| `main.py` | `33-36` | En `startup` llama a `init_api()` que pre-carga el vectorstore |
| `routes.py` | `14-17` | `init_api()` - carga el vectorstore y guarda la config del modelo |
| `routes.py` | `20-28` | `GET /health` - liveness check con count de chunks |
| `routes.py` | `31-70` | `POST /query` - LangChain chain: retriever → prompt → llm → response |
| `routes.py` | `73-106` | `POST /query/stream` - `StreamingResponse` con `chain.astream()` |

**LangChain chain** (`routes.py:61-65`):

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

LangChain orquesta: toma la pregunta, invoca el retriever, formatea el
contexto, lo mete en el prompt, llama a Ollama, devuelve la respuesta.
La **misma logica** se usa en el CLI (`rag.py`) y en la API
(`routes.py`) — solo cambia la interfaz.

### 4.7 `src/evaluation/evaluate.py` (F6: metricas de retrieval)

Sobre `test_bank.json` (20 preguntas con expected_sources y
expected_categories), corre el retriever con `k=5` y mide:

| Metrica | Que mide | Nuestro resultado |
|---|---|---|
| `precision_at_k` | Proporcion del top-k que viene de una fuente esperada | **0.26** (~1.3 de 5 chunks correctos) |
| `recall_at_k` | Proporcion de fuentes esperadas que aparecen en el top-k | **0.53** (recuperamos la mitad) |
| `category_match` | Proporcion de categorias esperadas en el top-k | **0.70** (casi siempre en el area correcta) |
| `faithfulness_score` | Proporcion de oraciones de la respuesta con >30% de overlap con el contexto | Mide si la respuesta esta "anclada" en el corpus |
| `retrieval_time` | Tiempo del retrieve en segundos | avg=0.1s, max=1.25s |

**Por que `category_match=0.7` importa mas que P@5=0.26:** porque aunque
el chunk especifico no sea el correcto, si caemos en la categoria
correcta, el LLM tiene contexto **relevante** (no exactamente el mismo,
pero del mismo tema) y puede generar una respuesta util.

### 4.8 `src/evaluation/compare.py` (F7: before/after)

Corre las **10 preguntas oficiales** con dos modos:
- **Before**: Ollama `qwen2.5:7b-instruct` (default, 7B params).
- **After**: Qwen2.5-0.5B-Instruct + LoRA adapter (0.5B params, ~17 MB con LoRA).

Mide latencia y **4 comportamientos** definidos en `BEHAVIORS`
(`compare.py:23-28`):
- `cite_source` - "segun", "de acuerdo con", "reporta", ...
- `neutral` - "diferentes perspectivas", "complementarias", ...
- `uncertainty` - "no detalla", "fuera del alcance", ...
- `socratic` - "antes de responder", "paso a paso", ...

### 4.9 `src/finetuning/generate_dataset.py` (F8: dataset para LoRA)

Toma 50 chunks aleatorios del vectorstore, usa Ollama para generar
(par pregunta, respuesta condensada), y los junta con los 20 del test
bank. Output: `src/finetuning/dataset/rag_dataset.jsonl` (~70
registros).

### 4.10 `datasets/behavioral_dataset.py` (F9: dataset comportamental)

**300 ejemplos sinteticos** (no requieren Ollama), divididos en 4
comportamientos x 75:

| Comportamiento | Cantidad | Que aprende |
|---|---:|---|
| `citas` | 75 | Citar institucion y titulo antes de responder |
| `neutralidad` | 75 | Presentar multiples perspectivas sin tomar partido |
| `socratico` | 75 | Hacer preguntas guia antes de responder |
| `incertidumbre` | 75 | Reconocer explicitamente cuando algo no esta en el corpus |

**85% corpus-relevantes** (referencian docs reales), **15% genericos**
(preguntas abiertas). Esto fuerza al LoRA a aprender **como responder
en el estilo del corpus**, no solo **que responder**.

### 4.11 `src/finetuning/finetune.py` (F10: LoRA training)

Aplica LoRA sobre **Qwen2.5-0.5B-Instruct** con PEFT + TRL:

| Lineas | Que hace |
|---|---|
| `32` | `BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"` |
| `47-51` | `format_example()` - serializa cada ejemplo como `<|system|>...<|user|>...<|assistant|>...` |
| `61-66` | Args: `--epochs=3`, `--batch-size=1`, `--lr=2e-4`, `--lora-r=8`, `--lora-alpha=32`, `--max-length=512` |
| `100-108` | `LoraConfig(r=8, alpha=32, dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])` |
| `112-120` | Tokeniza los ejemplos con `padding="max_length"`, `truncation=True` |
| `122-136` | `TrainingArguments` con `gradient_accumulation_steps=4` (batch efectivo = 4) |
| `148-150` | `trainer.save_model()` guarda el adapter en `src/finetuning/lora_adapter/` |

**Hiperparametros default:**
- `epochs=3`, `batch_size=1`, `gradient_accumulation_steps=4` (efectivo batch = 4)
- `lr=2e-4` (tipico de LoRA, 10x mas alto que full fine-tuning)
- `max_length=512`
- Solo `q_proj, k_proj, v_proj, o_proj` (attention) — el resto del modelo queda **congelado**
- ~0.5% de los parametros se entrenan (~0.5-1M de ~500M)

**Advertencia:** el training en CPU es **extremadamente lento** (~1 hora
por epoch en CPU). El codigo detecta CUDA y usa GPU automaticamente. En
Colab con T4 tarda ~2-3 min por epoch.

### 4.12 `src/finetuning/integrate.py` (F11: RAG + LoRA)

Une el RAG con el LoRA: carga el modelo base + adapter via
`PeftModel.from_pretrained`, lo usa como generador en lugar de Ollama.

```bash
# consulta unica
python src/finetuning/integrate.py --query "¿Cuantos homicidos hubo en 2024?"

# interactivo
python src/finetuning/integrate.py -i

# comparar contra Ollama
python src/finetuning/integrate.py -i --use-ollama
```

`LoRARAG.load_model()` (`integrate.py:41-63`):

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "src/finetuning/lora_adapter")
model.eval()
```

Si no encuentra el adapter, usa el modelo base solo (con un warning).

---

## 5. Fine-tuning LoRA (seccion propia)

### 5.1 Por que LoRA y no full fine-tuning

| | Full fine-tuning | LoRA |
|---|---|---|
| Parametros entrenados | 100% (~500M para Qwen 0.5B) | ~0.5% (~0.5-1M) |
| Memoria GPU necesaria | ~16 GB (modelo + gradientes + optimizador) | ~2 GB |
| Tiempo de training (CPU) | Infeasible (>1 dia/epoch) | ~1 h/epoch |
| Tiempo de training (T4 GPU) | ~30 min/epoch | ~2-3 min/epoch |
| Calidad final | Marginalmente mejor | Comparable |
| Adapter reusable | No (modelo completo) | Si (~5-8 MB) |

**El truco de LoRA:** en vez de actualizar la matriz `W` (de dimension
`d × d`), LoRA descompone la actualizacion como
`ΔW = A · B` donde `A` es `d × r` y `B` es `r × d` con `r << d`
(aca `r=8` vs `d=896`). El numero de parametros entrenados pasa de
`d²` a `2·d·r`. Para `d=896, r=8`: 802.816 vs 14.336 = **56x menos**.

### 5.2 Hiperparametros exactos

```python
LoraConfig(
    r=8,                        # rango bajo de la descomposicion
    lora_alpha=32,              # factor de escala (alpha/r = 4x amplificacion)
    lora_dropout=0.1,           # regularizacion
    bias="none",                # no entrenar biases
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

**Por que `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`:**
- Son las **4 proyecciones de atencion** de cada capa transformer.
- Entrenar solo attention (no MLP) es lo standard y suele dar mejor
  relacion calidad/tiempo.
- Si quisieras **toda** la red, podrias agregar `target_modules=["q_proj",
  "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.

### 5.3 El dataset comportamental: la parte "inteligente"

El LoRA no aprende **contenido** (eso ya esta en el corpus), aprende
**estilo de respuesta**. Por eso el dataset no es Q&A normal, sino 4
comportamientos:

| Comportamiento | Ejemplo de training | Que se le ensena al modelo |
|---|---|---|
| **Citas** | "Segun INEGI (archivo: homicidios_inegi_2025.pdf), ..." | A **referenciar la institucion y el titulo** antes de responder |
| **Neutralidad** | "El corpus presenta diferentes perspectivas..." | A **presentar multiples visiones** sin tomar partido |
| **Socratico** | "Antes de responder, revisemos que dice el corpus..." | A **hacer preguntas guia** antes de dar una conclusion |
| **Incertidumbre** | "El corpus no contiene informacion suficiente..." | A **reconocer limites** cuando algo no esta |

**Por que 4 comportements y no 1:** un modelo chico (0.5B) tiende a
copiar literalmente los chunks. El LoRA le da **4 modos de responder**
para que elija segun el contexto.

**Por que 85% corpus-relevantes:** para que el modelo aprenda el
vocabulario y las instituciones del dominio (INEGI, MUCD, ONC, etc.).

**Por que 15% genericos:** para que no se vuelva **solo** un echo del
corpus — tambien tiene que saber responder cosas basicas.

### 5.4 Cuando usar Ollama vs LoRA

| Escenario | Usar | Por que |
|---|---|---|
| Defensa con GPU disponible | **Ollama 7B** | Mas capacidad, mejor calidad |
| Defensa sin GPU, sin internet | **LoRA 0.5B** | Corre en CPU, ~1.5 GB RAM |
| Comparativa para el profe | **Ambos** | `compare.py` los pone lado a lado |
| Produccion (servidor) | **Ollama 7B** | Latencia, calidad, mantenible |

---

## 6. Cosas puntuales del procesamiento + trampas conocidas

### 6.1 Cosas puntuales

| Tema | Detalle |
|---|---|
| Chunk size vs overlap | `chunk_size=1000` chars por chunk. `chunk_overlap=200` chars compartidos entre chunks consecutivos. El overlap evita perder contexto en los bordes. |
| Por que cosine + normalize | Con vectores normalizados (norma L2=1), dot product ≡ coseno. ChromaDB con `hnsw:space: cosine` espera vectores normalizados. |
| P@5=0.26 vs category_match=0.7 | El chunk especifico no siempre es el correcto (P@5 bajo), pero casi siempre caemos en la **categoria** correcta (category_match alto). El LLM puede generar respuestas utiles con contexto relevante aunque no sea el exacto. |
| chunk_id estable | `chunk_id = <source_stem>_<idx:04d>`. Es deterministico: si re-corre chunking sobre el mismo PDF, da los mismos IDs. Esto permite citar chunks especificos en cualquier momento. |
| Temperature 0.1 | Baja (no creativa). Queremos respuestas factuales, no "creativas". Si subis a 0.7-1.0 el modelo empieza a inventar. |
| num_predict=2048 | Limite duro de tokens de la respuesta. ~5-6 parrafos. Suficiente para 95% de las consultas. |
| Streaming | `stream=True` en Ollama. Cada token llega como una linea NDJSON. El CLI los va concatenando e imprimiendo en vivo. |
| Sin GPU el LoRA es muy lento | `peft` no tiene quantizacion automatica habilitada (la variable `BitsAndBytesConfig` esta importada pero no usada en `finetune.py:22`). Sin quantizacion, el modelo vive en RAM en fp32 (~2 GB para 0.5B) y cada backward pass recorre 500M parametros. Solucion: Colab con T4 + quantizacion 4-bit. |
| Tests | 20 en `test_bank.json`, 10 oficiales en `official_questions.json` (3 extraccion, 4 sintesis, 3 analisis) |
| Recuperacion cuando falta info | El prompt instruye: "Si el contexto no contiene la informacion, indicalo claramente". El LoRA re-entrenado aprende explicitamente este comportamiento (categoria "incertidumbre"). |

### 6.2 Trampas conocidas (objeciones del profe)

| Objecion | Respuesta |
|---|---|
| *Por que RAG y no solo un LLM mas grande?* | Un LLM de 7B alucina en datos especificos de Mexico (cifras, instituciones, anos). RAG le da **contexto verificable** y permite **citar fuentes**. |
| *Por que no un LLM de 70B?* | No entra en GPU tipica (~140 GB). 7B es el sweet spot para correr en GPU de 16 GB (T4 de Colab). |
| *Por que `all-MiniLM-L6-v2` y no `bge-large`?* | bge-large es 1.3 GB y 1024 dim. MiniLM es 22 MB y 384 dim. Para este corpus (7.595 chunks) la diferencia de calidad no justifica 60x mas recursos. |
| *Por que distancia coseno y no L2?* | Los embeddings estan normalizados, asi coseno = dot product. Coseno mide el **angulo** (similitud semantica), L2 mide **distancia geometrica** (sensible a magnitud). Para embeddings normalizados, coseno es lo correcto. |
| *Por que `chunk_size=1000` y no 500?* | Mas contexto por chunk = menos chances de que la informacion se parta. 1000 chars ~ 200 tokens, suficiente para una idea completa. Mas chico = mas chunks = retrieval menos preciso. |
| *Por que HNSW y no IVF?* | HNSW es mas rapido para <1M vectores y no requiere training. IVF seria mejor para millones de vectores. |
| *Por que LoRA y no QLoRA?* | QLoRA agrega quantizacion 4-bit al base model. Es lo que se hace en Colab para que entre en T4. En el codigo local el modelo ya entra en RAM sin quantizar. |
| *Por que Qwen 0.5B y no Llama o Mistral?* | Qwen 2.5 es excelente para espanol (entrenado en datos multilingues). 0.5B es el tamano justo para que LoRA corra en CPU. |
| *El codigo no compila, no, espera — la respuesta a veces es incoherente.* | El LoRA chico (0.5B) tiene capacidad limitada. Para preguntas fuera del corpus o muy complejas, Ollama 7B responde mejor. `compare.py` documenta esta diferencia. |
| *Por que temperatura 0.1?* | Baja = respuestas mas deterministas/factuales. Para tareas de Q&A factual no queres creatividad. Si subis a 0.7+ el modelo empieza a inventar contexto. |
| *Que pasa si Ollama no esta corriendo?* | `requests.post` a `localhost:11434` falla con `ConnectionError`. Soluciones: (a) `ollama serve`, (b) usar `integrate.py` con LoRA, (c) migrar a Colab. |
| *Por que `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`?* | Son las 4 proyecciones de atencion. Entrenar solo attention (no MLP) es el standard en LoRA — mejor relacion calidad/tiempo. |
| *Por que 4 comportements en el dataset y no 1?* | Un modelo chico tiende a copiar literalmente los chunks. Con 4 estilos, aprende a **elegir** como responder segun el contexto (citas si hay fuente, neutral si hay polemica, etc.). |
| *El chunk_id cambia entre corridas?* | No, es deterministico: `<source_stem>_<idx:04d>`. Si re-corre chunking sobre el mismo PDF, da los mismos IDs. Esto permite citar chunks especificos sin ambiguedad. |
| *Por que CORS abierto en la API?* | Para que cualquier frontend (incluso en otro puerto) pueda consumirla durante desarrollo. En produccion se restringe a dominios especificos. |

---

## 7. Glosario compacto (~35 terminos)

| Termino | Significado |
|---|---|
| **RAG** | Retrieval-Augmented Generation: busqueda + generacion. El LLM responde con contexto recuperado. |
| **Retriever** | Componente que busca los chunks mas relevantes dado un query |
| **Embedding** | Vector denso de dimension fija (aca 384) que representa un texto |
| **Cosine similarity** | Medida de similitud entre vectores = coseno del angulo. Rango [-1, 1]. |
| **HNSW** | Hierarchical Navigable Small World. Indice aproximado para nearest-neighbor en alta dim. |
| **ChromaDB** | Vector database open-source, persistencia en SQLite, metadata filtering nativo |
| **Chunk** | Trozo de texto (aca 1000 chars) que se indexa y recupera como unidad |
| **Chunk overlap** | Chars compartidos entre chunks consecutivos. Evita perder contexto en bordes. |
| **PyPDFLoader** | Loader de LangChain que extrae texto de PDFs (1 doc por pagina) |
| **RecursiveTextSplitter** | Splitter de LangChain que corta en jerarquia: `\n\n` > `\n` > `. ` > `, ` > ` ` |
| **all-MiniLM-L6-v2** | Modelo de sentence-transformers. 22 MB, 384 dim, 6 capas. |
| **Sentence-transformers** | Libreria de HuggingFace para modelos de embeddings semanticos |
| **Ollama** | Servidor local de LLMs. Corre en `localhost:11434`. API REST simple. |
| **Qwen2.5** | Familia de LLMs de Alibaba. 0.5B, 1.5B, 7B, 14B, 32B, 72B. |
| **LangChain** | Framework de orquestacion de LLMs. Maneja chains, retrievers, prompts, output parsers. |
| **Prompt template** | String con placeholders que se llenan en runtime (ej: `{context}`, `{question}`) |
| **Streaming** | Generar token por token en vez de esperar la respuesta completa |
| **NDJSON** | Newline-Delimited JSON. Un JSON por linea. Lo usa Ollama en su API. |
| **LoRA** | Low-Rank Adaptation. Metodo de fine-tuning que entrena ~0.5% de los parametros. |
| **PEFT** | Parameter-Efficient Fine-Tuning. Libreria de HuggingFace para LoRA y similares. |
| **QLoRA** | LoRA + quantizacion 4-bit del modelo base. Cabe en GPU de 16 GB. |
| **Base model** | Modelo pre-entrenado sin fine-tuning (aca: `Qwen/Qwen2.5-0.5B`) |
| **Instruct model** | Modelo con fine-tuning de instrucciones (aca: `Qwen/Qwen2.5-0.5B-Instruct`) |
| **Adapter** | Archivos de pesos LoRA (~5-8 MB). Se "pegan" al base model para inferencia. |
| **target_modules** | Que modulos del transformer entrenar con LoRA. Aca: q/k/v/o proj. |
| **LoraConfig** | Config de LoRA: `r`, `alpha`, `dropout`, `target_modules`, `task_type` |
| **Faithfulness** | Metrica: la respuesta esta "anclada" en el contexto vs inventada |
| **P@K, R@K** | Precision y Recall a K documentos recuperados |
| **Category match** | Proporcion de categorias esperadas en el top-K |
| **FastAPI** | Framework HTTP async para Python. Usado en `api/main.py`. |
| **Uvicorn** | Servidor ASGI que corre FastAPI. |
| **CORS** | Cross-Origin Resource Sharing. Middleware para permitir frontends en otros puertos. |
| **Pydantic** | Validacion de tipos para Python. Usado en schemas de FastAPI. |
| **bitsandbytes** | Libreria de quantizacion 4-bit/8-bit. Requerida para QLoRA. |
| **RAGAS** | Framework para evaluar RAG (no usado aca, pero es el standard de la industria) |
| **Hybrid search** | Combinar busqueda semantica (embeddings) con keyword (BM25). Mejora recall. |

---

## Anexo: comandos utiles para la defensa

```bash
# === Setup ===
cd 4-RAG
../.venv/bin/python run_pipeline.py           # chunking + embeddings
../.venv/bin/python run_pipeline.py --rebuild  # forzar reindexacion

# === RAG CLI ===
../.venv/bin/python run_pipeline.py --rag
# > ¿Cuantos homicidios hubo en Mexico en 2024?
# > /fuentes
# > /modelo qwen2.5:0.5b
# > /salir

# === RAG API ===
../.venv/bin/python run_pipeline.py --api --port 8000
# en otra terminal:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Que dice INEGI sobre cifra negra?", "k": 5}'

# === Evaluacion ===
../.venv/bin/python run_pipeline.py --evaluate
../.venv/bin/python run_pipeline.py --official
../.venv/bin/python run_pipeline.py --compare   # Ollama 7B vs LoRA 0.5B

# === Fine-tuning LoRA ===
../.venv/bin/python datasets/behavioral_dataset.py             # 300 ejemplos
../.venv/bin/python src/finetuning/generate_dataset.py         # 70 con Ollama
../.venv/bin/python src/finetuning/finetune.py --epochs 3     # entrena LoRA
../.venv/bin/python src/finetuning/integrate.py -i             # RAG + LoRA

# === Smoke test rapido (sin LoRA entrenado) ===
../.venv/bin/python src/rag/rag.py "¿Que estados tienen mas homicidios?" -m qwen2.5:0.5b
```

## Ver tambien

- `Guias/GUIA_CNN.md` - guia del Proyecto 2 (CNN)
- `Guias/GUIA_RNN.md` - guia del Proyecto 3 (RNN char-level)
- `Guias/GUIA_RAG_COLAB.md` - **migracion a Google Colab** (siguiente guia)
- `4-RAG/README.md` - README principal
- `4-RAG/docs/guia_estudio.md` - guia tecnica detallada ya existente
- `4-RAG/docs/INICIO.md` - quick start
