from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list
    latency: float


class FilteredQueryRequest(BaseModel):
    question: str
    category: str
    k: int = 5


class HealthResponse(BaseModel):
    status: str
    vectorstore: str
    total_chunks: int
    ollama_model: str
