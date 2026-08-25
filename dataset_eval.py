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
