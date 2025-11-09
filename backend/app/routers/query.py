from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..db import get_db
from ..services import embeddings, generator

router = APIRouter()

class QueryReq(BaseModel):
    query: str
    top_k: int = 5

@router.post("")
def query_endpoint(body: QueryReq, db: Session = Depends(get_db)):
    if not body.query.strip():
        raise HTTPException(400, "Query is empty.")

    qvec = embeddings.embed_texts([body.query])[0]

    # ANN search using pgvector operator <-> (L2 by default)
    sql = text("""
        SELECT c.text, d.title, c.seq
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.document_id
        ORDER BY e.vector <-> :qvec
        LIMIT :k
    """)
    rows = db.execute(sql, {"qvec": qvec, "k": body.top_k}).fetchall()
    if not rows:
        raise HTTPException(404, "No context available. Ingest a document first.")

    passages = [(r.title, r.text, r.seq) for r in rows]
    answer = generator.generate_answer(body.query, passages)
    citations = [f"{r.title} #{r.seq}" for r in rows]
    return {"answer": answer, "citations": citations}
