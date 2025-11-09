# fastAPI app initialization

from fastapi import FastAPI
from .routers import ingest, query



#app = FastAPI(title="P_AI", version="0.1.0")
app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

def root():
    return {"message": "Welcome to the P_AI API", "ok": True, "see": ["/healthz", "/docs", "/ingest/file", "/query"]}


app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix ='/query', tags=["query"])


from .db import Base, engine
from . import models  # ensure models are imported/registered

Base.metadata.create_all(bind=engine)
