#pgvector search retriever


from sqlalchemy import text
from sqlalchemy.orm import Session

def ann_search(db: Session, qvec: list[float], top_k: int = 5):
    sql = text("""
        SELECT c.id, c.text, d.title
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.document_id
        ORDER BY e.vector <-> :qvec
        LIMIT :k
    """)
    return db.execute(sql, {"qvec": qvec, "k": top_k}).fetchall()
