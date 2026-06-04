# El RAG y la pregunta por el autor

## Contexto

El profe puede preguntar **"¿qué piensa el autor del corpus?"** o variantes
("¿cuál es la postura del autor?", "¿de quién es esta idea?"). Esta es una
**pregunta meta** porque el corpus no tiene un autor único: tiene 23
documentos de 16 instituciones diferentes.

Este documento explica cómo está estructurado el RAG para responder a este
tipo de preguntas.

## Estructura de la respuesta

El sistema distingue **tres niveles de atribución**:

### 1. El curador (jojo)
jojo es el estudiante que **reunió y organizó** los documentos. Su rol fue
seleccionar qué PDFs incluir, no producir contenido.

> "El corpus fue curado por jojo, estudiante de la materia de IA, como
> parte del Proyecto 4. jojo no es experto en seguridad pública; organizó
> 23 documentos de 16 instituciones."

### 2. Los autores institucionales (16)
Cada documento tiene un `author` y una `institution` en su metadata. Cuando
el RAG responde sobre un tema, **atribuye cada afirmación a la institución
correspondiente** (ej. "Según INEGI en la ENVIPE 2024...").

### 3. El corpus en su conjunto (plural)
El corpus **no tiene postura única**. Sobre temas con disenso explícito
(militarización, cifra negra, subregistro), contiene tanto voces oficiales
como críticas. La convención es **presentar el pluralismo**, no sintetizar
artificialmente.

## Implementación técnica

### Archivos clave

| Archivo | Rol |
|---|---|
| `corpus/processed/MANIFIESTO.md` | Inyectado en el system prompt. Define la identidad del corpus y cómo responder sobre el autor. |
| `corpus/processed/AUTORES.md` | Catálogo de las 16 instituciones. Referencia. |
| `src/_format.py` | Helpers: `load_manifesto()`, `format_docs_with_attribution()`, `build_rag_prompt()`. |
| `src/ingest/chunking.py` | Propaga `author`, `institution`, `year` desde `documentos.json` a cada chunk. |
| `src/rag/rag.py` | Usa `format_docs_with_attribution()` y `build_rag_prompt()` con MANIFIESTO. |
| `src/finetuning/integrate.py` | Idem, para el RAG con LoRA. |
| `src/evaluation/compare.py` | Idem, para la comparativa before/after. |
| `datasets/atribution_examples.jsonl` | 30 ejemplos de atribución para fine-tuning. |
| `datasets/behavioral_dataset.py` | Merge: 300 base + 30 atribución = 330 ejemplos. |
| `src/finetuning/finetune.py` | Usa `build_training_prompt()` con MANIFIESTO. |
| `src/evaluation/official_questions.json` | Q11, Q12, Q13: meta-preguntas. |

### Flujo de una query

```
Pregunta del usuario
        ↓
Retriever (ChromaDB + MiniLM-L6-v2)
        ↓
Top-5 chunks con metadata {author, institution, year, ...}
        ↓
format_docs_with_attribution()  ←  agrega [Fuente: ...] visible
        ↓
build_rag_prompt(context, question, include_manifesto=True)
   ├── system: MANIFIESTO.md + instrucción
   ├── user:   contexto atribuido + pregunta
   └── target: respuesta del LLM
        ↓
Qwen 2.5 0.5B + LoRA adapter
        ↓
Respuesta con atribuciones explícitas
```

## Respuestas esperadas para el profe

### "¿Quién es el autor del corpus?"

> El corpus fue curado por **jojo**, estudiante de la materia de
> Inteligencia Artificial, como parte del Proyecto 4. jojo organizó 23
> documentos de 16 instituciones diferentes; su rol fue seleccionar y
> clasificar, no producir análisis editorial. Los autores del contenido
> son las instituciones listadas en `corpus/processed/AUTORES.md`.

### "¿Qué piensa el autor sobre la militarización?"

> El corpus no tiene una postura unificada sobre militarización, porque
> contiene documentos de instituciones con líneas distintas:
> - **MUCD** ('El negocio de la militarización', 2024) critica la
>   opacidad y el poder fáctico de las fuerzas armadas.
> - **Fundación Carolina** ('Militarización y militarismo en México',
>   2023) presenta un análisis académico más descriptivo.
> - El **disenso explícito** entre voces críticas y oficiales se preserva
>   en el corpus, no se sintetiza artificialmente.

### "¿Qué postura tiene el corpus?"

> El corpus es **plural por diseño**. Atribuirle una postura única sería
> simplificar lo que el sistema justamente intenta preservar: el
> pluralismo de fuentes. Los datos cuantitativos (cifras oficiales de
> INEGI) son la base común; los análisis críticos (México Evalúa, MUCD,
> Centro Prodh) cuestionan las políticas del Estado; y la academia
> (UNAM, Fundación Carolina) aporta perspectiva temporal.

## Cómo se entrena al LoRA para esto

El dataset de fine-tuning tiene **330 ejemplos** distribuidos así:

| Behavior | # | Ejemplo |
|---|---:|---|
| `citas` | 75 | Citar institución y título antes de responder |
| `neutralidad` | 75 | Presentar múltiples perspectivas |
| `socrático` | 75 | Hacer preguntas guía antes de responder |
| `incertidumbre` | 75 | Reconocer límites del corpus |
| **`atribucion`** | **30** | **Identificar al curador, listar autores, atribuir por institución, reconocer pluralidad** |

Los 30 ejemplos de `atribucion` están en `datasets/atribution_examples.jsonl`
y cubren:
- Identidad del curador (jojo) y su rol (5 ejemplos)
- Lista de los 16 autores institucionales (5 ejemplos)
- Atribución por institución específica: "qué piensa INEGI / México Evalúa / MUCD..." (10 ejemplos)
- Reconocimiento del pluralismo del corpus (5 ejemplos)
- Diferencia entre curador y autor (5 ejemplos)

## Métricas de validación

- **Q11** (oficial): ¿El RAG menciona a jojo y a las 16 instituciones?
- **Q12** (oficial): ¿El RAG presenta disenso entre autores sin sintetizar?
- **Q13** (oficial): ¿El RAG explica la diferencia entre autor único y corpus plural?

Para correrlas:
```bash
python 4-RAG/src/finetuning/compare.py --mode after --save
```

## Diferencias vs. versión anterior

| Aspecto | Antes | Ahora |
|---|---|---|
| Metadata de chunk | 5 campos (sin author) | 8 campos (con author, institution, year) |
| System prompt | "experto en seguridad pública..." | MANIFIESTO + instrucción |
| Formato de contexto | `"\n\n".join(text)` | `"[Fuente: INST — Título (año)]\n" + text` |
| Atribución en respuesta | No explícita | "Según INEGI...", "México Evalúa argumenta..." |
| Preguntas meta en evaluación | 0 | 3 (Q11, Q12, Q13) |
| Ejemplos de atribución en LoRA | 0 | 30 |

## Changelog

- **2026-06-03**: Reestructuración inicial. Creado MANIFIESTO.md, AUTORES.md,
  metadata enriquecida en chunks, 30 ejemplos de atribución para LoRA.
