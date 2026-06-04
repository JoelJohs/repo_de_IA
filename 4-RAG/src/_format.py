"""
Helpers compartidos de formato y carga de contexto para RAG.

Usado por:
  - src/rag/rag.py              (CLI interactivo + query única)
  - src/finetuning/integrate.py (RAG + LoRA)
  - src/finetuning/compare.py   (comparativa before/after)
  - src/finetuning/finetune.py  (training con MANIFIESTO en system prompt)
"""

import sys
from functools import lru_cache
from pathlib import Path

_BASE_DIR = Path(__file__).parent.parent
_MANIFESTO_PATH = _BASE_DIR / "corpus" / "processed" / "MANIFIESTO.md"
_AUTORES_PATH = _BASE_DIR / "corpus" / "processed" / "AUTORES.md"


def ensure_src_on_path() -> None:
    """Asegura que src/ esté en sys.path para poder hacer `from _format import ...`."""
    src = str(_BASE_DIR / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


@lru_cache(maxsize=1)
def load_manifesto() -> str:
    """Carga el MANIFIESTO del corpus (system prompt raíz).

    Cacheado en memoria porque se llama muchas veces por query.
    Devuelve string vacío si el archivo no existe (modo degraded).
    """
    if _MANIFESTO_PATH.exists():
        return _MANIFESTO_PATH.read_text(encoding="utf-8").strip()
    return ""


@lru_cache(maxsize=1)
def load_autores() -> str:
    """Carga el catálogo de autores del corpus."""
    if _AUTORES_PATH.exists():
        return _AUTORES_PATH.read_text(encoding="utf-8").strip()
    return ""


def format_docs_with_attribution(docs) -> str:
    """Concatena los chunks recuperados con un header de atribución visible.

    Esto es CRÍTICO para que el LLM pueda atribuir correctamente las
    afirmaciones a su fuente institucional.

    Ejemplo de salida:
        [Fuente: INEGI — Defunciones por Homicidio 2024 (2025), pág. 3]
        El texto del primer chunk va acá...

        [Fuente: México Evalúa — Hallazgos 2023 (2024), pág. 12]
        El texto del segundo chunk va acá...

    Args:
        docs: lista de objetos Document de LangChain (o cualquier objeto
              con .page_content y .metadata).

    Returns:
        String con todos los chunks formateados y separados por doble salto.
    """
    blocks = []
    for d in docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        institution = meta.get("institution") or meta.get("category", "?")
        title = meta.get("document_title") or meta.get("source_file", "?")
        year = meta.get("year", "?")
        pages = meta.get("pages", "?")
        author = meta.get("author")
        header = f"[Fuente: {institution}"
        if author and author != institution:
            header += f" / {author}"
        header += f" — {title} ({year}), pág. {pages}]"
        content = d.page_content if hasattr(d, "page_content") else str(d)
        blocks.append(f"{header}\n{content}")
    return "\n\n".join(blocks)


def build_rag_prompt(context: str, question: str, include_manifesto: bool = True) -> str:
    """Construye el prompt completo para una query RAG.

    Estructura:
        <|system|>
        {MANIFIESTO del corpus (si include_manifesto=True)}

        Eres un asistente experto en seguridad pública en México...
        <|user|>
        Contexto:
        {chunks formateados con atribución}

        Pregunta: {question}
        <|assistant|>

    Args:
        context: string devuelto por format_docs_with_attribution().
        question: pregunta del usuario.
        include_manifesto: si False, omite el MANIFIESTO (útil para
                           comparativas where se quiere aislar la variable).

    Returns:
        Prompt listo para tokenizer (Qwen2.5 chat format).
    """
    system_parts = []
    if include_manifesto:
        manifesto = load_manifesto()
        if manifesto:
            system_parts.append(manifesto)
    system_parts.append(
        "Eres un asistente experto en seguridad pública en México. "
        "Responde SOLO con la información del contexto proporcionado. "
        "Cuando el contexto incluya datos cuantitativos, cita la institución y el título. "
        "Si hay disenso entre autores, presenta las múltiples perspectivas. "
        "Si el contexto no contiene la información, indícalo claramente."
    )
    system_block = "\n\n".join(system_parts)

    return (
        f"<|system|>\n{system_block}</s>\n"
        f"<|user|>\nContexto:\n{context}\n\nPregunta: {question}</s>\n"
        f"<|assistant|>"
    )


def build_training_prompt(instruction: str, response: str) -> str:
    """Construye un ejemplo de training en formato Qwen2.5 chat.

    El system prompt también incluye el MANIFIESTO para que el LoRA
    aprenda a responder consistentemente con la identidad del corpus.

    Args:
        instruction: pregunta del usuario.
        response: respuesta esperada (target del training).

    Returns:
        String formateado para tokenización con field 'text'.
    """
    system_block = load_manifesto()
    if not system_block:
        system_block = "Eres un asistente experto en seguridad pública en México."
    system_block += (
        "\n\nResponde SOLO con la información del contexto proporcionado. "
        "Cuando el contexto incluya datos cuantitativos, cita la institución y el título."
    )
    return (
        f"<|system|>\n{system_block}</s>\n"
        f"<|user|>\n{instruction}</s>\n"
        f"<|assistant|>\n{response}</s>"
    )
