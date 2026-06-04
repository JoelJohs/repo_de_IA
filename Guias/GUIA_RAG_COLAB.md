# Guia de migracion a Google Colab — Proyecto 4 (RAG + LoRA)

> El profesor recomendo migrar el Proyecto 4 a **Google Colab** porque
> te da una **GPU T4 gratis**, lo que cambia el juego:
> - El **fine-tuning LoRA** pasa de ~1 hora/epoch en CPU a ~2-3 min/epoch
> - El **embedding** se mantiene en CPU (es rapido igual)
> - El **retrieve** se mantiene en CPU (ChromaDB anda bien)
> - La **generacion** se mueve a GPU (con `transformers` local, sin Ollama)
>
> Esta guia te lleva paso a paso: desde abrir el notebook hasta tener
> el RAG + LoRA funcionando en la nube, todo con snippets copy-paste.

---

## 0. Por que migrar (y por que NO)

### Ventajas

| Beneficio | Detalle |
|---|---|
| **GPU T4 gratis** | 16 GB VRAM. Suficiente para Qwen 0.5B + LoRA con quantizacion 4-bit. |
| **LoRA 50x mas rapido** | ~2-3 min/epoch en T4 vs ~1 hora/epoch en CPU. |
| **Cero setup local** | No necesitás Ollama, ni 16 GB de RAM, ni CUDA. |
| **Persistencia en Drive** | Podés guardar vectordb + adapter y reusarlos entre sesiones. |
| **Notebooks versionados** | La entrega formal del proyecto es un `.ipynb` ejecutado. |

### Desventajas (y como mitigarlas)

| Problema | Mitigacion |
|---|---|
| **Sesiones max 12 h** | Guardar checkpoints en Drive cada X epochs. |
| **GPU no siempre disponible** | A veces toca esperar cola o usar CPU. El LoRA igual corre, solo que mas lento. |
| **Drive mount falla a veces** | Re-montar con `drive.mount('/content/drive', force_remount=True)`. |
| **No se puede usar Ollama** | Reemplazamos por `transformers` local con T4 (sección 3). |
| **Archivos del corpus** | Hay que subirlos. Ver sección 2 (3 opciones). |

---

## 1. Setup del entorno

### 1.1 Crear el notebook

1. Andá a https://colab.research.google.com/
2. **Archivo → Nuevo notebook**
3. Renombralo a `4_RAG_LoRA_Seguridad_Mexico.ipynb`
4. **Entorno de ejecucion → Cambiar tipo de entorno de ejecucion**
   - Tipo: Python 3
   - Acelerador de hardware: **GPU T4** (gratis)

### 1.2 Celda 1: montar Drive + verificar GPU

```python
from google.colab import drive
drive.mount('/content/drive')

!nvidia-smi
# Esperado: Tesla T4, 16 GB VRAM, driver ~525+
```

### 1.3 Celda 2: instalar dependencias

```python
# IMPORTANTE: Colab viene con TF 2.15+ y Python 3.10. No usar el venv local.
!pip install -q \
  langchain>=0.3.0 \
  langchain-community>=0.3.0 \
  langchain-huggingface>=1.0.0 \
  langchain-chroma>=1.0.0 \
  langchain-core>=0.3.0 \
  langchain-text-splitters>=0.3.0 \
  chromadb>=0.5.0 \
  sentence-transformers>=3.0.0 \
  pypdf>=5.0.0 \
  pymupdf>=1.27.0 \
  fastapi>=0.115.0 \
  uvicorn>=0.32.0 \
  peft>=0.12.0 \
  trl>=0.9.0 \
  bitsandbytes>=0.43.0 \
  accelerate>=0.33.0
```

> **Nota sobre `bitsandbytes`**: requerido para QLoRA (quantizacion
> 4-bit). En Colab a veces falla la instalacion por la version de CUDA;
> si pasa, reiniciar runtime y volver a correr.

### 1.4 Celda 3: definir paths en Drive

```python
from pathlib import Path

# Todas las rutas cuelgan de aca. Drive persiste entre sesiones.
DRIVE_ROOT = Path("/content/drive/MyDrive/4-RAG-colab")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

CORPUS_DIR   = DRIVE_ROOT / "corpus"   # PDFs
VECTORDB_DIR = DRIVE_ROOT / "vectordb" # ChromaDB persistido
LORA_DIR     = DRIVE_ROOT / "lora"     # Adapter entrenado
HF_CACHE     = DRIVE_ROOT / "hf_cache" # Cache de modelos de HF

# Crear subdirs
for d in [CORPUS_DIR, VECTORDB_DIR, LORA_DIR, HF_CACHE]:
    d.mkdir(parents=True, exist_ok=True)

# Configurar cache de HuggingFace para que use Drive (no /root)
import os
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_CACHE)

print(f"Drive root: {DRIVE_ROOT}")
print(f"GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()}")
```

---

## 2. Cargar el corpus (3 opciones)

### Opcion A: descargar los 23 PDFs desde las URLs (RECOMENDADA)

Es la mas limpia porque no requiere subir archivos.

```python
import json, re, time
from datetime import datetime
from pathlib import Path
import requests
import fitz  # PyMuPDF

# Esto replica src/ingest/scraper.py:30-165 del proyecto local
DIRECT_SOURCES = [
    # === HOMICIDIOS ===
    ("homicidios", "INEGI", 2025,
     "https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/edr/DH2024_CP_Ene-dic.pdf",
     "Defunciones por Homicidio 2024 - Comunicado de prensa"),
    ("homicidios", "INEGI", 2025,
     "https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2025/edr/DH2024_RR_Ene-dic.pdf",
     "Defunciones por Homicidio 2024 - Resultados completos"),
    # ... PEGA ACA EL RESTO DE DIRECT_SOURCES DE src/ingest/scraper.py:30-165
    # (o importa el modulo desde el repo, ver Opcion C)
]

def download_pdf(url, dest):
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        if r.content[:4] != b"%PDF":
            print(f"  ✗ No es PDF: {url[:60]}")
            return False
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  ✗ Error: {url[:60]}: {e}")
        return False

# Descargar todo
ok, fail = 0, 0
for categoria, inst, anio, url, titulo in DIRECT_SOURCES:
    dest = CORPUS_DIR / categoria / f"{categoria}_{inst.lower().replace(' ','_')}_{anio}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if download_pdf(url, dest):
        ok += 1
        print(f"  ✓ {dest.name}")
    else:
        fail += 1

print(f"\nDescargados: {ok} | Fallos: {fail}")
```

> **Tip**: para no copiar los 23 URLs a mano, importa el modulo del
> repo local (ver Opcion C).

### Opcion B: subir un zip a Drive

```python
# 1) En tu maquina local: zip -r 4-rag-corpus.zip corpus/pdf/
# 2) Subilo a Drive manualmente
# 3) En Colab:
import zipfile
ZIP_PATH = Path("/content/drive/MyDrive/4-rag-corpus.zip")
EXTRACT_TO = CORPUS_DIR.parent  # /content/drive/MyDrive/4-RAG-colab

if ZIP_PATH.exists():
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_TO)
    print(f"Extraido a {EXTRACT_TO}")
else:
    print(f"Subi el zip a {ZIP_PATH} y reejecuta")
```

### Opcion C: clonar el repo y copiar el corpus (MAS FACIL)

```python
# 1) Subi el repo a GitHub (si no esta)
# 2) En Colab:
!git clone https://github.com/TU_USUARIO/IA.git /content/IA
!cp -r /content/IA/4-RAG/corpus/pdf/* $CORPUS_DIR/
!ls $CORPUS_DIR
```

**Tabla comparativa:**

| Opcion | Pros | Contras |
|---|---|---|
| **A: descargar URLs** | No requiere archivos locales, reproducible | Tarda 5-10 min, depende de internet |
| **B: zip a Drive** | Rapido una vez subido | Manual, hay que mantener el zip sincronizado |
| **C: clonar repo** | Un comando, tenes todo | Requiere repo en GitHub |

**Recomendacion**: si el repo ya esta en GitHub, usa C. Si no, A.

---

## 3. Reemplazar Ollama (transformers local + T4)

**Ollama no funciona en Colab** (es un binario que necesita correr
como servicio persistente). Lo reemplazamos por `transformers` local
cargado en la T4.

### 3.1 Opcion recomendada: Qwen 0.5B + LoRA en GPU (full local)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Modelo base (chico, entra en T4 sin quantizar)
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Adapter LoRA (si ya lo entrenaste antes, ver seccion 5)
ADAPTER_DIR = LORA_DIR  # /content/drive/MyDrive/4-RAG-colab/lora

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,   # T4 nativo: bf16 (mejor que fp16, sin loss de rango)
    device_map="cuda:0",          # forzar GPU
    trust_remote_code=True,
)

# Aplicar LoRA si existe
if (ADAPTER_DIR / "adapter_config.json").exists():
    print(f"Cargando adapter desde {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
else:
    print(f"Sin adapter. Usando modelo base solo.")
    model = base

model.eval()
model = model.to("cuda")
print(f"Modelo en {next(model.parameters()).device}")
```

### 3.2 Funcion de generacion

```python
def generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in answer:
        answer = answer.split("<|assistant|>")[-1].strip()
    return answer
```

### 3.3 Test rapido

```python
test_prompt = """<|system|>
Eres un asistente experto en seguridad publica en Mexico.</s>
<|user|>
Di "hola" en una oracion.</s>
<|assistant|>"""

print(generate(test_prompt, max_new_tokens=50))
# Esperado: "Hola, soy un asistente..."
```

---

## 4. Pipeline RAG en celdas

### 4.1 Chunking (replica `src/ingest/chunking.py`)

```python
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

# Recolectar todos los PDFs
pdfs = list(CORPUS_DIR.rglob("*.pdf"))
print(f"PDFs encontrados: {len(pdfs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
)

chunks = []
for pdf_path in pdfs:
    categoria = pdf_path.parent.name
    pages = PyPDFLoader(str(pdf_path)).load()
    if not pages:
        continue
    docs = splitter.split_documents(pages)
    source_stem = pdf_path.stem
    for idx, doc in enumerate(docs):
        text = doc.page_content.strip()
        if not text:
            continue
        chunks.append({
            "chunk_id": f"{source_stem}_{idx:04d}",
            "text": text,
            "metadata": {
                "category": categoria,
                "source_file": str(pdf_path.relative_to(CORPUS_DIR)),
                "pages": str(doc.metadata.get("page", "")),
                "chunk_index": idx,
            },
        })

# Guardar a Drive
chunks_file = DRIVE_ROOT / "chunks.jsonl"
with chunks_file.open("w") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"Total chunks: {len(chunks)} -> {chunks_file}")
```

### 4.2 Embeddings + ChromaDB (replica `src/embeddings/embeddings.py`)

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # CPU esta bien para MiniLM
    encode_kwargs={"normalize_embeddings": True},
)

# Reconstruir Document objects desde chunks.jsonl
docs = []
with chunks_file.open() as f:
    for line in f:
        c = json.loads(line)
        meta = c["metadata"].copy()
        meta.pop("chunk_index", None)
        docs.append(Document(page_content=c["text"], metadata=meta, id=c["chunk_id"]))

# Construir/reindexar ChromaDB
import shutil
if VECTORDB_DIR.exists():
    shutil.rmtree(VECTORDB_DIR)

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=str(VECTORDB_DIR),
    collection_name="seguridad_mexico",
    collection_metadata={"hnsw:space": "cosine"},
)

print(f"ChromaDB con {vectordb._collection.count()} chunks en {VECTORDB_DIR}")
```

> **Tarda ~5-10 min** la primera vez. La proxima sesion ya esta en Drive
> y solo tarda ~10 s en abrir.

### 4.3 Funcion de RAG completa

```python
import time

PROMPT_TEMPLATE = """<|system|>
Eres un asistente experto en seguridad publica en Mexico. Responde SOLO con la informacion del contexto proporcionado.</s>
<|user|>
Contexto:
{context}

Pregunta: {question}</s>
<|assistant|>"""


def rag_query(question: str, k: int = 5) -> dict:
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    t0 = time.time()
    docs = retriever.invoke(question)
    retrieve_time = time.time() - t0

    context = "\n\n".join(d.page_content for d in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    t0 = time.time()
    answer = generate(prompt, max_new_tokens=256, temperature=0.1)
    gen_time = time.time() - t0

    return {
        "answer": answer,
        "sources": [
            {
                "content": d.page_content[:200] + "...",
                "metadata": d.metadata,
            }
            for d in docs
        ],
        "timing": {"retrieve_s": round(retrieve_time, 2), "generate_s": round(gen_time, 2)},
    }


# Test
result = rag_query("¿Cuantos homicidios hubo en Mexico en 2024?")
print("\n=== RESPUESTA ===")
print(result["answer"])
print("\n=== FUENTES ===")
for i, src in enumerate(result["sources"], 1):
    print(f"[{i}] {src['metadata']['source_file']} (cat: {src['metadata']['category']})")
print("\n=== TIMING ===")
print(result["timing"])
```

**Esperado**: respuesta en 3-10 s (T4 GPU), 5 fuentes con PDF y categoria.

---

## 5. Fine-tuning LoRA en T4 (el punto fuerte de Colab)

Esto es lo que justifica migrar: **LoRA en T4 tarda 50x menos que en CPU**.

### 5.1 Generar dataset comportamental (replica `behavioral_dataset.py`)

```python
# Si lo subiste a Drive, copia el archivo
!cp /content/drive/MyDrive/4-RAG/behavioral_dataset.py /content/
!python /content/behavioral_dataset.py

# O generarlo inline: 300 ejemplos, 4 comportements (citas/neutral/socratico/incertidumbre)
# Ver datasets/behavioral_dataset.py del proyecto local para el codigo completo
```

Output: `datasets/finetuning.jsonl` con 300 ejemplos.

### 5.2 Celda de training LoRA con QLoRA (4-bit)

```python
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = DRIVE_ROOT / "finetuning.jsonl"  # el de la celda anterior

# Quantizacion 4-bit: reduce memoria ~4x (T4 tiene 16 GB, el modelo pasa de 2 GB a 0.5 GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Cargar modelo quantizado
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
)

# Preparar para training con k-bit (LoRA sobre modelo quantizado)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Esperado: trainable params: ~1M || all params: ~500M || trainable%: ~0.2%

# Cargar dataset
records = [json.loads(l) for l in DATASET_PATH.open() if l.strip()]
texts = [
    f"<|system|>\nEres un asistente experto en seguridad publica en Mexico.</s>\n"
    f"<|user|>\n{r['instruction']}</s>\n"
    f"<|assistant|>\n{r['response']}</s>"
    for r in records
]
dataset = Dataset.from_list([{"text": t} for t in texts])

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# Training args optimizados para T4
training_args = TrainingArguments(
    output_dir=str(LORA_DIR / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=4,         # T4: 4 entra
    gradient_accumulation_steps=2,          # batch efectivo = 8
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    fp16=False,
    bf16=True,                              # T4: bf16 nativo
    dataloader_pin_memory=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

# Entrenar (~2-3 min por epoch en T4, vs ~1 hora en CPU)
import time
t0 = time.time()
trainer.train()
print(f"Training completo en {(time.time()-t0)/60:.1f} min")

# Guardar adapter en Drive
trainer.save_model(str(LORA_DIR))
tokenizer.save_pretrained(str(LORA_DIR))
print(f"Adapter guardado en {LORA_DIR}")
```

**Tiempos esperados en T4:**

| Dataset | Epochs | Tiempo total |
|---|---:|---:|
| 300 ejemplos | 3 | ~6-10 min |
| 70 ejemplos (rag_dataset) | 3 | ~2-3 min |

### 5.3 Verificar el adapter

```python
import os
print(os.listdir(LORA_DIR))
# Esperado: ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json', ...]

print(f"Tamano: {sum(f.stat().st_size for f in LORA_DIR.glob('**/*') if f.is_file()) / 1e6:.1f} MB")
# Esperado: ~5-8 MB
```

---

## 6. Integrar LoRA + RAG (recarga el modelo con el adapter)

```python
# Recargar el modelo con el adapter recien entrenado
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = LORA_DIR

# Limpiar modelo anterior de la GPU
del model
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)

# Aplicar adapter
model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
model.eval()
model = model.to("cuda")
print(f"Modelo + LoRA cargados en {next(model.parameters()).device}")

# Probar
def generate_v2(prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in answer:
        answer = answer.split("<|assistant|>")[-1].strip()
    return answer

# Test: pregunta del corpus
result = rag_query("¿Que instituciones publicaron el reporte de cifra negra?")
print(result["answer"])
```

---

## 7. Issues tipicos (y solucion copy-paste)

### Issue 1: `bitsandbytes` no instala

```
ERROR: Could not build wheels for bitsandbytes
```

**Solucion:**
```python
# En Celda 2, antes de bitsandbytes:
!pip install -q bitsandbytes --no-build-isolation

# Si sigue fallando:
import os
os.environ["BNB_CUDA_VERSION"] = "124"  # o la version que tengas
!pip install -q bitsandbytes
```

### Issue 2: `OutOfMemoryError` en T4 al cargar el modelo

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Solucion:**
```python
# Reducir batch size o habilitar quantizacion 4-bit
# En la celda 5.2, cambiar:
per_device_train_batch_size=1,        # en vez de 4
gradient_accumulation_steps=8,         # batch efectivo = 8 igual

# Y agregar antes de cargar el modelo:
torch.cuda.empty_cache()
```

### Issue 3: El modelo responde en ingles en vez de espanol

**Solucion:**
```python
# Forzar el idioma en el prompt
PROMPT_TEMPLATE = """<|system|>
Eres un asistente experto en seguridad publica en Mexico. 
SIEMPRE responde en espanol, citando instituciones y titulos del corpus.
Si el contexto no contiene la informacion, indicálo en espanol.</s>
<|user|>
Contexto:
{context}

Pregunta: {question}</s>
<|assistant|>"""
```

### Issue 4: `Drive not mounted`

**Solucion:**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)  # force_remount re-monta
```

### Issue 5: `HF_TOKEN` required para algunos modelos

**Solucion:**
```python
# Para modelos gated (Llama, Mistral), necesitas token:
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')

# Qwen 2.5 NO requiere token (es open).
```

### Issue 6: `CUDA_LAUNCH_BLOCKING=1` para debug

```python
# Si el training crashea y no sabes donde:
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Reejecutar la celda. El proximo error mostrara la linea exacta.
```

### Issue 7: El LoRA entrenado no mejora las respuestas

**Diagnostico:**
```python
# Comparar antes/despues con la misma pregunta
test_q = "¿Que instituciones publicaron datos sobre criminalidad?"
ctx = vectordb.as_retriever(search_kwargs={"k": 3}).invoke(test_q)
ctx_str = "\n\n".join(d.page_content for d in ctx)
prompt = PROMPT_TEMPLATE.format(context=ctx_str, question=test_q)

# Sin LoRA
model.disable_adapter_layers()
print("=== SIN LoRA ===")
print(generate_v2(prompt, max_new_tokens=200))

# Con LoRA
model.enable_adapter_layers()
print("\n=== CON LoRA ===")
print(generate_v2(prompt, max_new_tokens=200))
```

**Soluciones si no mejora:**
- Mas epochs (`num_train_epochs=5` o `10`)
- Mas datos (revisar que `finetuning.jsonl` tenga los 300 ejemplos)
- Learning rate mas alto (`5e-4` en vez de `2e-4`)
- Verificar que el adapter se cargo bien (`model.active_adapters` deberia ser `['default']`)

---

## 8. Checklist de defensa (antes de exponer)

```markdown
[ ] 1. Notebook ejecutado de punta a punta sin errores
[ ] 2. Celda "Setup" muestra GPU T4 (no CPU)
[ ] 3. Vectordb persistido en Drive (tamaño ~70 MB)
[ ] 4. Adapter LoRA persistido en Drive (tamaño 5-8 MB)
[ ] 5. Output de la query de prueba tiene:
       - Respuesta coherente en espanol
       - 5 fuentes listadas con PDF y categoria
       - Tiempo de respuesta razonable (< 30 s en T4)
[ ] 6. Tabla comparativa antes/despues del LoRA (Issue 7)
[ ] 7. Screenshots de:
       - nvidia-smi mostrando la T4
       - chunks.jsonl con al menos 7.000 lineas
       - Comparativa de respuestas con/sin LoRA
       - Grafica de training loss (Trainer ya la loggea)
[ ] 8. El profe puede abrir el notebook y re-ejecutar todo
[ ] 9. README del Colab con instrucciones de uso
[ ] 10. Link publico al notebook (opcional: "Compartir" en Colab)
```

### 8.1 Como entregar el notebook

1. **Archivo → Guardar una copia en GitHub** (si el repo es tuyo)
   O: **Archivo → Descargar .ipynb** y subirlo al repo en
   `4-RAG/notebooks/rag_colab.ipynb`
2. Asegurarse de que las **celdas de output** esten visibles
   (no solo el codigo)
3. Documentar en el README principal que ahora hay un `.ipynb` ejecutable
   en Colab como bonus

### 8.2 Como exportar el adapter a local (opcional)

Si queres bajar el adapter entrenado en Colab a tu maquina local para
usarlo con `integrate.py`:

```python
# En Colab, al final del notebook:
!zip -r lora_adapter.zip /content/drive/MyDrive/4-RAG-colab/lora/
# Descargar el zip manualmente desde Drive a tu maquina
```

Despues en local:
```bash
unzip lora_adapter.zip
cp -r lora/* 4-RAG/src/finetuning/lora_adapter/
python 4-RAG/src/finetuning/integrate.py -i
```

---

## 9. Resumen de tiempos (todo el pipeline en Colab)

| Etapa | Tiempo (primera vez) | Tiempo (sesiones siguientes, con Drive) |
|---|---:|---:|
| Setup (Drive, pip, GPU) | 2-3 min | 30 s |
| Descargar corpus (23 PDFs) | 5-10 min | 0 (ya en Drive) |
| Chunking | 1-2 min | 0 (chunks.jsonl ya en Drive) |
| Embeddings + ChromaDB | 5-10 min | 10-30 s (cargar desde Drive) |
| Generacion con Qwen 0.5B (sin LoRA) | 30 s | 30 s |
| Generar dataset comportamental (300 ej) | 1 min | 0 (jsonl en Drive) |
| **Fine-tuning LoRA (3 epochs)** | **6-10 min** | **0 (adapter ya en Drive)** |
| Recargar modelo con LoRA | 30 s | 30 s |
| Query de prueba RAG | 3-10 s | 3-10 s |
| **TOTAL primera vez** | **~25-40 min** | **~2-3 min por query** |

---

## 10. Diferencias clave: local vs Colab

| Aspecto | Local (Linux, sin GPU) | Colab (T4 GPU) |
|---|---|---|
| Embedding | CPU, ~5 min para 7595 chunks | CPU, igual |
| Retrieve | CPU, ~50 ms | CPU, igual |
| Ollama | Necesario instalar y servir | **No funciona**, usar transformers |
| LoRA training | ~1 h/epoch (infeasible para 3 epochs) | **~2-3 min/epoch** |
| LoRA inference | ~0.5 tok/s en CPU | ~10 tok/s en T4 |
| Ollama 7B inference | ~0.5 tok/s en CPU | ~30 tok/s en T4 |
| Persistencia | Disco local | Google Drive (manual mount) |
| Costo | 0 | 0 (free tier) |

**Conclusion**: en local hacé todo **excepto** el fine-tuning LoRA (que
se vuelve eterno). En Colab, corré el LoRA en T4 y exportá el adapter a
local para usar con `integrate.py`.

---

## Ver tambien

- `Guias/GUIA_RAG.md` - guia de defensa del proyecto (conceptos)
- `4-RAG/README.md` - README principal
- `4-RAG/docs/ENTORNO.md` - por que Python 3.11.9 local
- HuggingFace PEFT docs: https://huggingface.co/docs/peft
- Bitsandbytes QLoRA: https://huggingface.co/blog/4bit-transformers-bitsandbytes
