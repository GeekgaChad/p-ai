# SQLAlchemy models for Document, Chunk, and Embedding with pgvector support

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from .db import Base
# pgvector
from sqlalchemy import types as satypes

class Vector(satypes.UserDefinedType):
    def get_col_spec(self): return "vector(1024)"

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    s3_uri = Column(String, nullable=False)
    mime = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    seq = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    meta_json = Column(JSONB, default={})
    document = relationship("Document", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="CASCADE"), index=True, unique=True)
    vector = Column(Vector)  # pgvector(1024)
    chunk = relationship("Chunk", back_populates="embedding")
