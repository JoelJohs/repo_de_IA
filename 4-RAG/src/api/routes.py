import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.embeddings.embeddings import get_vectorstore
from src.api.schemas import QueryRequest, QueryResponse, HealthResponse

router = APIRouter()

vectorstore = None
_gen_config = {}


def init_api(ollama_model: str, k: int = 5, temperature: float = 0.1):
    global vectorstore, _gen_config
    vectorstore = get_vectorstore()
    _gen_config = {"model": ollama_model, "temperature": temperature, "k": k}


@router.get("/health", response_model=HealthResponse)
def health():
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not initialized")
    count = vectorstore._collection.count()
    return HealthResponse(
        status="ok", vectorstore="ChromaDB (all-MiniLM-L6-v2)",
        total_chunks=count, ollama_model=_gen_config["model"],
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    retriever = vectorstore.as_retriever(search_kwargs={"k": req.k})
    t0 = time.time()
    docs = retriever.invoke(req.question)

    sources = [
        {"content": d.page_content[:300], "metadata": d.metadata}
        for d in docs
    ]

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.llms import Ollama

    llm = Ollama(model=_gen_config["model"], temperature=_gen_config["temperature"])

    prompt = ChatPromptTemplate.from_template(
        "Eres un asistente experto en seguridad pública en México. "
        "Responde basándote ÚNICAMENTE en el siguiente contexto.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}\n\nRespuesta:"
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    answer = chain.invoke(req.question)
    latency = round(time.time() - t0, 2)

    return QueryResponse(answer=answer, sources=sources, latency=latency)


@router.post("/query/stream")
def query_stream(req: QueryRequest):
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    retriever = vectorstore.as_retriever(search_kwargs={"k": req.k})

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.llms import Ollama

    llm = Ollama(model=_gen_config["model"], temperature=_gen_config["temperature"])

    prompt = ChatPromptTemplate.from_template(
        "Eres un asistente experto en seguridad pública en México. "
        "Responde basándote ÚNICAMENTE en el siguiente contexto.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}\n\nRespuesta:"
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    async def generate():
        async for chunk in chain.astream(req.question):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
