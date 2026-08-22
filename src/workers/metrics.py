"""Prometheus metrics for the eval worker process.

The worker is not request-driven, so it runs its own metrics HTTP server
(`start_metrics_server`) on a separate port instead of exposing /metrics
through the (nonexistent) API framework -- see run_worker.py, which calls
this once at process startup.

These metrics are recorded from a single, persistent process, not a
forked-per-job one: run_worker.py runs an RQ `SimpleWorker` rather than RQ's
default `Worker`, which forks a child process per job. Prometheus counters
and histograms are plain in-process objects; under the default forking
Worker, increments made inside each forked child would be invisible to the
parent process actually serving /metrics (that's what prometheus_client's
multiprocess mode exists to solve). A single always-on process sidesteps
that complexity entirely, at the cost of RQ's per-job crash isolation --
an acceptable tradeoff here since this project's scaling story is already
"more worker containers" (`docker compose up --scale worker=N`), not
per-job process isolation within one container.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, start_http_server

JUDGE_LATENCY_SECONDS = Histogram(
    "judge_latency_seconds",
    "Judge scoring latency per trace, from evaluate_answer() call to return.",
    labelnames=("status",),
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 300),
)

# Score distributions, 0.0-1.0 range, only observed for completed evals.
_SCORE_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

EVAL_CORRECTNESS = Histogram(
    "eval_correctness", "Distribution of correctness scores.", buckets=_SCORE_BUCKETS
)
EVAL_RELEVANCE = Histogram(
    "eval_relevance", "Distribution of relevance scores.", buckets=_SCORE_BUCKETS
)
EVAL_GROUNDEDNESS = Histogram(
    "eval_groundedness", "Distribution of groundedness scores.", buckets=_SCORE_BUCKETS
)

EVAL_HALLUCINATION_TOTAL = Counter(
    "eval_hallucination_total",
    "Count of completed evals by hallucination flag.",
    labelnames=("hallucination",),
)


def start_metrics_server(port: int) -> None:
    start_http_server(port)
