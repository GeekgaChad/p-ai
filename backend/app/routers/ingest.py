
# app/routers/ingest.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Document, Chunk, Embedding
from ..services import s3, pdf, chunker, embeddings

router = APIRouter()

PDF_MIME = {"application/pdf"}
MAX_CHUNKS = 4_000  # guardrail to avoid runaway inserts on weird PDFs

@router.post("/file")
async def ingest_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 0) basic validation
    if file.content_type not in PDF_MIME:
        raise HTTPException(status_code=400, detail="Only PDF supported for MVP.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    # 1) stream raw file to S3 (no full read into RAM)
    try:
        # UploadFile.file is a SpooledTemporaryFile / file-like; perfect for streaming
        s3_uri = s3.put_fileobj(file.file, file.filename, file.content_type)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"S3 upload failed: {e}")

    # 2) re-open from S3 to extract text
    try:
        data = s3.get_bytes(s3_uri)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"S3 readback failed: {e}")

    if not data:
        raise HTTPException(status_code=400, detail="Empty file content.")

    try:
        text = pdf.extract_text(data)  # your existing parser that accepts bytes
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parse failed: {e}")

    # 3) chunk
    chunks = list(chunker.simple_chunks(text))
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found.")
    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(status_code=413, detail=f"Too many chunks (> {MAX_CHUNKS}).")

    # 4) embed (consider batching inside embeddings.embed_texts to keep payload small)
    try:
        vecs = embeddings.embed_texts(chunks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}")

    if len(vecs) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding service returned unexpected size.")

    # 5) persist to Postgres
    try:
        doc = Document(title=file.filename, s3_uri=s3_uri, mime=file.content_type)
        db.add(doc); db.flush()

        created = 0
        for i, (t, v) in enumerate(zip(chunks, vecs)):
            c = Chunk(document_id=doc.id, seq=i, text=t, meta_json={})
            db.add(c); db.flush()
            e = Embedding(chunk_id=c.id, vector=v)
            db.add(e)
            created += 1

        db.commit()
    except Exception:
        db.rollback()
        raise
    return {"document_id": doc.id, "chunks": created}
