# pydantic I/O schemas

from pydantic import BaseModel

class IngestResponse(BaseModel):
    document_id: int
    chunks: int

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
