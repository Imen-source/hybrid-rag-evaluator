"""Dataset loading and metric aggregation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from schemas import AggregatedMetrics, DatasetSample, EvaluationResult


def load_dataset(dataset_path: str) -> list[DatasetSample]:
    """Load and validate a dataset of RAG evaluation samples."""
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Dataset JSON must contain a list of samples.")

    return [DatasetSample.model_validate(item) for item in payload]


def dataset_digest(dataset_path: str) -> str:
    """Return a stable hash of the dataset for reproducibility and logging."""
    path = Path(dataset_path)
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def aggregate_metrics(results: Iterable[dict]) -> dict:
    """Aggregate per-sample scores into dataset-level metrics."""
    validated_results = [EvaluationResult.model_validate(result) for result in results]
    if not validated_results:
        raise ValueError("At least one evaluation result is required to aggregate metrics.")

    total = len(validated_results)
    metrics = AggregatedMetrics(
        avg_correctness=sum(item.correctness for item in validated_results) / total,
        avg_relevance=sum(item.relevance for item in validated_results) / total,
        avg_groundedness=sum(item.groundedness for item in validated_results) / total,
        avg_semantic_similarity=sum(item.semantic_similarity for item in validated_results) / total,
        avg_keyword_overlap=sum(item.keyword_overlap for item in validated_results) / total,
        avg_confidence=sum(item.confidence for item in validated_results) / total,
        hallucination_rate=sum(1 for item in validated_results if item.hallucination) / total,
        sample_count=total,
    )
    return metrics.model_dump()


# Metrics compared between a candidate run and its baseline. Deliberately a
# subset of AggregatedMetrics: avg_semantic_similarity/avg_keyword_overlap/
# avg_confidence are judge-internals/fallback-scoring signals, not
# system-quality signals a regression gate should act on.
COMPARABLE_METRICS = ("avg_correctness", "avg_relevance", "avg_groundedness", "hallucination_rate")

# Regression gate thresholds (see evaluate_regression() docstring for the
# reasoning behind relative-vs-absolute and why these three metrics/values).
CORRECTNESS_RELATIVE_DROP_THRESHOLD = 0.05
GROUNDEDNESS_RELATIVE_DROP_THRESHOLD = 0.05
HALLUCINATION_RATE_ABSOLUTE_INCREASE_THRESHOLD = 0.05


def compute_metric_deltas(candidate: dict, baseline: dict) -> dict:
    """Pure per-metric diff between two aggregate_metrics()-shaped dicts.

    Only reads COMPARABLE_METRICS keys, so callers can pass either a full
    aggregate_metrics() output or a minimal dict containing just those keys.
    `relative_delta` is (candidate - baseline) / baseline; `None` when the
    baseline value is exactly 0, since a relative change is undefined there.
    """
    deltas = {}
    for metric in COMPARABLE_METRICS:
        baseline_value = baseline[metric]
        candidate_value = candidate[metric]
        delta = candidate_value - baseline_value
        relative_delta = (delta / baseline_value) if baseline_value else None
        deltas[metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
            "relative_delta": relative_delta,
        }
    return deltas


def evaluate_regression(deltas: dict) -> dict:
    """Apply the regression gate to a compute_metric_deltas() output.

    Threshold design (a real decision, not an arbitrary pick):

    - avg_correctness / avg_groundedness use a *relative* drop threshold
      (5%) rather than absolute, because both are 0-1 scores whose baseline
      level varies a lot by dataset/judge: a 0.05 absolute drop is huge
      against a 0.95 baseline and noise against a 0.20 one. Relative framing
      keeps the bar meaningful either way.
    - hallucination_rate uses an *absolute* increase threshold (5
      percentage points) instead, because it's already a rate: relative
      framing breaks down near a 0 baseline (any single hallucination on a
      previously-perfect baseline would be an "infinite" relative increase),
      and an absolute point-increase is what a human would actually mean by
      "hallucinations got noticeably more common."
    - avg_relevance is reported in the deltas but does not gate. It reflects
      topical/retrieval fit more than answer correctness or safety, and is
      the noisiest of the judge's scores in practice -- gating deployment on
      it risks blocking good changes over phrasing variance rather than real
      regressions. Correctness, groundedness, and hallucination rate are the
      three signals this project treats as "is the system still right and
      not making things up," which is what a regression gate should protect.

    A run only needs to trip ONE of the three gated metrics to count as
    regressed -- these are independent failure modes (a system can get less
    correct without hallucinating more, or vice versa), not a combined score.
    """
    reasons = []

    correctness = deltas["avg_correctness"]
    if (
        correctness["relative_delta"] is not None
        and correctness["relative_delta"] <= -CORRECTNESS_RELATIVE_DROP_THRESHOLD
    ):
        reasons.append(
            f"avg_correctness dropped {abs(correctness['relative_delta']):.1%} relative to "
            f"baseline (threshold: {CORRECTNESS_RELATIVE_DROP_THRESHOLD:.0%})"
        )

    groundedness = deltas["avg_groundedness"]
    if (
        groundedness["relative_delta"] is not None
        and groundedness["relative_delta"] <= -GROUNDEDNESS_RELATIVE_DROP_THRESHOLD
    ):
        reasons.append(
            f"avg_groundedness dropped {abs(groundedness['relative_delta']):.1%} relative to "
            f"baseline (threshold: {GROUNDEDNESS_RELATIVE_DROP_THRESHOLD:.0%})"
        )

    hallucination = deltas["hallucination_rate"]
    if hallucination["delta"] >= HALLUCINATION_RATE_ABSOLUTE_INCREASE_THRESHOLD:
        reasons.append(
            f"hallucination_rate increased {hallucination['delta']:.1%} points "
            f"(threshold: {HALLUCINATION_RATE_ABSOLUTE_INCREASE_THRESHOLD:.0%} points)"
        )

    return {
        "regressed": bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "correctness_relative_drop": CORRECTNESS_RELATIVE_DROP_THRESHOLD,
            "groundedness_relative_drop": GROUNDEDNESS_RELATIVE_DROP_THRESHOLD,
            "hallucination_rate_absolute_increase": HALLUCINATION_RATE_ABSOLUTE_INCREASE_THRESHOLD,
        },
    }
