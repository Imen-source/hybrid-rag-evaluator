"""Baseline-vs-candidate run comparison, backed by real EvalResult rows.

Thin DB-fetching layer over the pure functions in dataset_eval.py
(aggregate_metrics, compute_metric_deltas, evaluate_regression) -- see that
module for the aggregation logic and the regression-threshold reasoning.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from dataset_eval import aggregate_metrics, compute_metric_deltas, evaluate_regression
from src.api import models

# Same fields mlflow_utils.py pulls off a completed EvalResult row -- kept
# in sync deliberately since both feed the same aggregate_metrics().
SCORE_FIELDS = (
    "correctness",
    "relevance",
    "groundedness",
    "hallucination",
    "explanation",
    "semantic_similarity",
    "keyword_overlap",
    "confidence",
)


class NoCompletedResultsError(ValueError):
    """Raised when a run has no completed EvalResult rows to aggregate."""


def _completed_score_dicts(session: Session, run_id: uuid.UUID) -> list[dict]:
    results = (
        session.execute(
            select(models.EvalResult).where(
                models.EvalResult.eval_run_id == run_id,
                models.EvalResult.status == "completed",
            )
        )
        .scalars()
        .all()
    )
    return [{field: getattr(r, field) for field in SCORE_FIELDS} for r in results]


def compare_runs(session: Session, candidate_run_id: uuid.UUID, baseline_run_id: uuid.UUID) -> dict:
    """Aggregate both runs' completed results and return deltas + verdict.

    Raises NoCompletedResultsError if either run has zero completed results
    (aggregate_metrics() can't average an empty set).
    """
    candidate_scores = _completed_score_dicts(session, candidate_run_id)
    baseline_scores = _completed_score_dicts(session, baseline_run_id)

    if not candidate_scores:
        raise NoCompletedResultsError(f"Run {candidate_run_id} has no completed results to compare.")
    if not baseline_scores:
        raise NoCompletedResultsError(
            f"Baseline run {baseline_run_id} has no completed results to compare."
        )

    candidate_metrics = aggregate_metrics(candidate_scores)
    baseline_metrics = aggregate_metrics(baseline_scores)

    deltas = compute_metric_deltas(candidate_metrics, baseline_metrics)
    verdict = evaluate_regression(deltas)

    return {
        "candidate_run_id": candidate_run_id,
        "baseline_run_id": baseline_run_id,
        "metrics": deltas,
        "thresholds": verdict["thresholds"],
        "regressed": verdict["regressed"],
        "regressed_reasons": verdict["reasons"],
    }
