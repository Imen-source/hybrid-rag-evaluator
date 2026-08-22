"""RQ job(s) executed by the eval worker process.

Reuses the existing, synchronous judge-scoring implementation in
evaluator.py unmodified -- this module is glue around it, not a
reimplementation.
"""

from __future__ import annotations

import time
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evaluator import EvaluatorConfig, evaluate_answer

from src.api import models
from src.api.db import SessionLocal
from src.workers import metrics
from src.workers.config import get_judge_settings
from src.workers.mlflow_utils import log_completed_run_to_mlflow

RESULT_FIELDS = (
    "correctness",
    "relevance",
    "groundedness",
    "hallucination",
    "explanation",
    "semantic_similarity",
    "keyword_overlap",
    "confidence",
)


def run_trace_eval(eval_run_id: str, trace_id: str) -> None:
    """Score one trace for one eval run and persist the result.

    After updating its own row, checks whether it was the last pending
    trace in the run and, if so, finalizes the run (status + MLflow log).
    """
    run_uuid = uuid.UUID(eval_run_id)
    trace_uuid = uuid.UUID(trace_id)
    settings = get_judge_settings()

    with SessionLocal() as session:
        result_row = session.execute(
            select(models.EvalResult).where(
                models.EvalResult.eval_run_id == run_uuid,
                models.EvalResult.trace_id == trace_uuid,
            )
        ).scalar_one()
        trace = session.get(models.Trace, trace_uuid)

        result_row.status = "running"
        session.commit()

        config = EvaluatorConfig(
            provider=settings["provider"],
            model=settings["model"],
            cache_dir=settings["cache_dir"],
            api_key_env=settings["api_key_env"],
            temperature=settings["temperature"],
            ollama_host=settings["ollama_host"],
            use_hybrid=settings["use_hybrid"],
            num_runs=settings["num_runs"],
        )

        start = time.monotonic()
        try:
            scores = evaluate_answer(
                question=trace.input,
                context=trace.retrieved_context or "",
                answer=trace.output,
                config=config,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            for field in RESULT_FIELDS:
                setattr(result_row, field, scores[field])
            result_row.latency_ms = elapsed_ms
            result_row.status = "completed"
            result_row.error = None
            metrics.JUDGE_LATENCY_SECONDS.labels(status="completed").observe(elapsed_ms / 1000.0)
            metrics.EVAL_CORRECTNESS.observe(scores["correctness"])
            metrics.EVAL_RELEVANCE.observe(scores["relevance"])
            metrics.EVAL_GROUNDEDNESS.observe(scores["groundedness"])
            metrics.EVAL_HALLUCINATION_TOTAL.labels(hallucination=str(scores["hallucination"])).inc()
        except Exception as exc:  # judge call failed even after evaluator.py's own retries/fallback
            elapsed_ms = (time.monotonic() - start) * 1000
            result_row.status = "failed"
            result_row.error = str(exc)
            result_row.latency_ms = elapsed_ms
            metrics.JUDGE_LATENCY_SECONDS.labels(status="failed").observe(elapsed_ms / 1000.0)
        session.commit()

        _maybe_finalize_run(session, run_uuid)


def _maybe_finalize_run(session: Session, run_uuid: uuid.UUID) -> None:
    # Lock the run row so that if two "last trace" jobs finish at nearly the
    # same time, the second one blocks here until the first has finalized
    # and committed, then sees run.status already terminal and no-ops.
    run = session.execute(
        select(models.EvalRun).where(models.EvalRun.id == run_uuid).with_for_update()
    ).scalar_one()

    if run.status in ("completed", "failed"):
        session.commit()
        return

    pending_count = session.execute(
        select(func.count())
        .select_from(models.EvalResult)
        .where(
            models.EvalResult.eval_run_id == run_uuid,
            models.EvalResult.status.in_(("pending", "running")),
        )
    ).scalar_one()

    if pending_count > 0:
        session.commit()
        return

    results = (
        session.execute(
            select(models.EvalResult).where(models.EvalResult.eval_run_id == run_uuid)
        )
        .scalars()
        .all()
    )
    succeeded = [r for r in results if r.status == "completed"]
    failed_count = len(results) - len(succeeded)

    run.status = "completed" if succeeded else "failed"
    if not succeeded:
        run.error = "All traces failed judge scoring."

    run.mlflow_run_id = log_completed_run_to_mlflow(
        run=run, results=succeeded, failed_count=failed_count
    )

    session.commit()
