"""MLflow logging for completed eval runs."""

from __future__ import annotations

import os

import mlflow

from dataset_eval import aggregate_metrics


def _configure_mlflow() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "trace-eval-runs"))


def log_completed_run_to_mlflow(run, results, failed_count: int) -> str | None:
    """Log one eval run as one MLflow run. `results` is the list of
    successfully-completed EvalResult ORM rows for this run."""
    _configure_mlflow()

    score_dicts = [
        {
            "correctness": r.correctness,
            "relevance": r.relevance,
            "groundedness": r.groundedness,
            "hallucination": r.hallucination,
            "explanation": r.explanation or "",
            "semantic_similarity": r.semantic_similarity,
            "keyword_overlap": r.keyword_overlap,
            "confidence": r.confidence,
        }
        for r in results
    ]

    with mlflow.start_run(run_name=f"eval-run-{run.id}") as mlflow_run:
        mlflow.log_param("judge_provider", run.judge_provider)
        mlflow.log_param("judge_model", run.judge_model)
        mlflow.log_param("worker_count", run.worker_count)
        mlflow.log_param("trace_count", run.trace_count)
        mlflow.log_param("eval_run_id", str(run.id))

        if score_dicts:
            aggregated = aggregate_metrics(score_dicts)
            for key, value in aggregated.items():
                mlflow.log_metric(key, value)

        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
        mlflow.log_metric("avg_judge_latency_ms", avg_latency_ms)
        mlflow.log_metric("succeeded_trace_count", len(results))
        mlflow.log_metric("failed_trace_count", failed_count)

        return mlflow_run.info.run_id
