from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ..db import get_db
from ..services import embeddings, retriever, generator

router = APIRouter()

class QueryReq(BaseModel):
    query: str
    top_k: int = 5

@router.post("/query")

def query_stub(body:QueryReq):
    return {"answer": "stub", "citations": []}

'''
def query_endpoint(body: QueryReq, db: Session = Depends(get_db)):
    qvec = embeddings.embed_texts([body.query])[0]
    rows = retriever.ann_search(db, qvec, body.top_k)
    passages = [(r.title, r.text, i) for i, r in enumerate(rows)]
    answer = generator.generate_answer(body.query, passages)
    citations = [f"{r.title} #{i}" for i, r in enumerate(rows)]
    return {"answer": answer, "citations": citations}
'''