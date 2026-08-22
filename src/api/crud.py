"""Persistence operations for traces."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from . import models, schemas


def create_trace(db: Session, trace: schemas.TraceCreate) -> models.Trace:
    db_trace = models.Trace(
        input=trace.input,
        output=trace.output,
        retrieved_context=trace.retrieved_context,
        meta=trace.metadata,
    )
    db.add(db_trace)
    db.commit()
    db.refresh(db_trace)
    return db_trace


def get_trace(db: Session, trace_id: uuid.UUID) -> models.Trace | None:
    return db.get(models.Trace, trace_id)
