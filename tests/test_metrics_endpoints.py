"""Confirm /metrics endpoints expose valid Prometheus exposition format with
the expected metric names -- no real judge call, no LLM involved.

The API check goes through a real (testcontainers) Postgres via the shared
api_app fixture, same as the other API tests, since importing src.api.main
binds the SQLAlchemy engine to whatever DATABASE_URL is set at that moment
(see conftest.py's docstring) -- /metrics itself doesn't touch the DB, but
the app has to be constructed the same sequenced way as every other test to
avoid binding the wrong DATABASE_URL.

The worker check is pure Python: it observes sample values directly into the
Histogram/Counter objects in src.workers.metrics and reads back the
exposition text, no Docker/network required.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client import generate_latest

# api_app comes from the root conftest.py.


def test_api_metrics_endpoint_returns_prometheus_exposition_format(api_app):
    with TestClient(api_app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body


def test_worker_metrics_expose_expected_names_after_observations():
    from src.workers import metrics

    metrics.JUDGE_LATENCY_SECONDS.labels(status="completed").observe(12.3)
    metrics.EVAL_CORRECTNESS.observe(0.9)
    metrics.EVAL_RELEVANCE.observe(0.8)
    metrics.EVAL_GROUNDEDNESS.observe(0.7)
    metrics.EVAL_HALLUCINATION_TOTAL.labels(hallucination="False").inc()

    body = generate_latest().decode("utf-8")

    for expected in (
        "judge_latency_seconds",
        "eval_correctness",
        "eval_relevance",
        "eval_groundedness",
        "eval_hallucination_total",
    ):
        assert expected in body, f"expected metric name {expected!r} in exposition output"

    assert 'status="completed"' in body
    assert 'hallucination="False"' in body


def test_start_metrics_server_is_importable_and_callable_signature():
    # Doesn't actually bind a port here (that's exercised live by run_worker.py
    # in the running worker container) -- just guards against an import-time
    # or signature regression in the small wrapper.
    from src.workers.metrics import start_metrics_server

    assert callable(start_metrics_server)
