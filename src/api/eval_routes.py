"""Eval-run endpoints: create a judge-scoring run, check its status/results."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.workers.config import get_judge_settings
from src.workers.queue import get_eval_queue
from src.workers.regression import NoCompletedResultsError, compare_runs

from . import models, schemas
from .db import get_db

router = APIRouter()


@router.post("/eval/run", response_model=schemas.EvalRunCreated, status_code=202)
def create_eval_run(payload: schemas.EvalRunCreate, db: Session = Depends(get_db)):
    if payload.trace_ids is not None:
        traces = (
            db.execute(select(models.Trace).where(models.Trace.id.in_(payload.trace_ids)))
            .scalars()
            .all()
        )
        found_ids = {t.id for t in traces}
        missing = set(payload.trace_ids) - found_ids
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown trace ids: {sorted(str(m) for m in missing)}",
            )
    else:
        completed_trace_ids = select(models.EvalResult.trace_id).where(
            models.EvalResult.status == "completed"
        )
        traces = (
            db.execute(select(models.Trace).where(models.Trace.id.notin_(completed_trace_ids)))
            .scalars()
            .all()
        )

    if not traces:
        raise HTTPException(status_code=400, detail="No traces available to evaluate.")

    settings = get_judge_settings()
    run = models.EvalRun(
        status="pending",
        judge_provider=settings["provider"],
        judge_model=settings["model"],
        trace_count=len(traces),
        worker_count=settings["worker_count"],
    )
    db.add(run)
    db.flush()

    for trace in traces:
        db.add(models.EvalResult(eval_run_id=run.id, trace_id=trace.id, status="pending"))

    # Commit BEFORE enqueueing: a worker could pick a job up almost
    # immediately, and its DB session must be able to see these rows.
    db.commit()
    db.refresh(run)

    queue = get_eval_queue()
    for trace in traces:
        queue.enqueue("src.workers.tasks.run_trace_eval", str(run.id), str(trace.id))

    run.status = "running"
    db.commit()
    db.refresh(run)

    return schemas.EvalRunCreated(run_id=run.id, status=run.status, trace_count=run.trace_count)


@router.get("/eval/runs/{run_id}", response_model=schemas.EvalRunRead)
def get_eval_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    run = db.get(models.EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return run


@router.post("/eval/runs/{run_id}/mark_baseline", response_model=schemas.EvalRunRead)
def mark_baseline(run_id: uuid.UUID, db: Session = Depends(get_db)):
    """Designate this run as THE baseline that /compare measures against.

    Marking a new baseline unsets whichever run was previously marked (a
    single mutable "current baseline" pointer), rather than rejecting the
    call if one already exists. A baseline is meant to move forward over
    time -- e.g. "the last known-good run" -- so requiring an explicit
    unmark step first would just add friction without protecting anything;
    the partial unique index added in migration 0003 (`is_baseline` can be
    true for at most one row) is what actually guarantees only one baseline
    exists, done here inside one transaction so the two writes can't race.
    """
    run = db.get(models.EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    if run.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Only a completed eval run can be marked as baseline.",
        )

    db.execute(
        update(models.EvalRun).where(models.EvalRun.is_baseline.is_(True)).values(is_baseline=False)
    )
    run.is_baseline = True
    db.commit()
    db.refresh(run)
    return run


@router.get("/eval/runs/{run_id}/compare", response_model=schemas.EvalRunCompareRead)
def compare_eval_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    """Compare this run's aggregate metrics against the current baseline run."""
    run = db.get(models.EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="Only a completed eval run can be compared.")

    baseline_run = db.execute(
        select(models.EvalRun).where(models.EvalRun.is_baseline.is_(True))
    ).scalar_one_or_none()
    if baseline_run is None:
        raise HTTPException(
            status_code=400,
            detail="No baseline run is set. Mark one via POST /eval/runs/{id}/mark_baseline first.",
        )
    if baseline_run.id == run.id:
        raise HTTPException(status_code=400, detail="This run is the baseline; nothing to compare.")

    try:
        return compare_runs(db, candidate_run_id=run.id, baseline_run_id=baseline_run.id)
    except NoCompletedResultsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
