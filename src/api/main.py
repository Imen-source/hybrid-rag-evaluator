"""FastAPI ingestion service: trace create/retrieve + health check."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from . import crud, schemas
from .db import get_db
from .eval_routes import router as eval_router

app = FastAPI(title="LLM Eval Ingestion API")
app.include_router(eval_router)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.post("/traces", response_model=schemas.TraceRead, status_code=201)
def create_trace(trace: schemas.TraceCreate, db: Session = Depends(get_db)):
    return crud.create_trace(db, trace)


@app.get("/traces/{trace_id}", response_model=schemas.TraceRead)
def read_trace(trace_id: uuid.UUID, db: Session = Depends(get_db)):
    db_trace = crud.get_trace(db, trace_id)
    if db_trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return db_trace


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except OperationalError:
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ok"}
