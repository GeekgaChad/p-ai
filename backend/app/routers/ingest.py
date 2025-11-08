
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Document, Chunk, Embedding
from ..services import s3, pdf, chunker, embeddings

router = APIRouter()

@router.post("/file")

async def ingest_file_stub():
    return {"status": "ingest endpoint alive"}


'''
async def ingest_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ("application/pdf",):
        raise HTTPException(400, "Only PDF for MVP")
    data = await file.read()
    s3_uri = s3.put_bytes(file.filename, data, file.content_type)
    text = pdf.extract_text(data)
    doc = Document(title=file.filename, s3_uri=s3_uri, mime=file.content_type)
    db.add(doc); db.flush()

    chunks = list(chunker.simple_chunks(text))
    # embed in batches (MVP: one batch)
    vecs = embeddings.embed_texts(chunks)

    created = 0
    for i,(t,v) in enumerate(zip(chunks, vecs)):
        c = Chunk(document_id=doc.id, seq=i, text=t, meta_json={})
        db.add(c); db.flush()
        e = Embedding(chunk_id=c.id, vector=v)
        db.add(e)
        created += 1

    db.commit()
    return {"document_id": doc.id, "chunks": created}
'''