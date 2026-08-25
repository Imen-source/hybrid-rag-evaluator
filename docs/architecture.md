# Architecture

This document covers the **service mode** of the repo (FastAPI + Postgres + Redis/RQ
worker + MLflow + Prometheus/Grafana, under `src/`, `infra/`, `alembic/`, `scripts/`).
The single-machine CLI (`cli.py`, `metaflow_pipeline.py`) reuses the same judge core
(`evaluator.py`) but is a separate, simpler entry point — see the root `README.md` for
that path.

## System diagram

```mermaid
flowchart LR
    Client -->|POST /traces| API[FastAPI ingestion API]
    Client -->|POST /eval/run<br>GET /eval/runs/id| API
    API -->|sync SQLAlchemy| PG[(Postgres<br>traces, eval_runs, eval_results)]
    API -->|enqueue job per trace| Redis[(Redis)]
    Redis -->|RQ SimpleWorker| Worker[Eval worker]
    Worker -->|evaluate_answer\|evaluator.py| Judge[Ollama or OpenAI]
    Worker -->|write scores| PG
    Worker -->|log run metrics<br>once all traces finish| MLflow[MLflow tracking server]
    API -->|GET /metrics| Prometheus
    Worker -->|:9100/metrics| Prometheus
    Prometheus --> Grafana
```

## Request flow

1. `POST /traces` writes one row to `traces` and returns it ([src/api/main.py](../src/api/main.py), [crud.py](../src/api/crud.py)).
2. `POST /eval/run` ([src/api/eval_routes.py](../src/api/eval_routes.py)) resolves the trace set (explicit `trace_ids`, or `scope: "all_pending"`), creates one `eval_runs` row plus one `pending` `eval_results` row per trace, commits, **then** enqueues one RQ job per trace on the `eval` queue. Commit-before-enqueue is deliberate: a worker can dequeue almost immediately, and its own DB session must already be able to see those rows.
3. Each worker job ([src/workers/tasks.py](../src/workers/tasks.py)) calls `evaluate_answer()` from `evaluator.py` — the same hybrid judge used by the CLI, unmodified — persists the per-trace result, records `judge_latency_seconds` and `evals_completed_total`, and checks (under a row lock on the `eval_runs` row) whether it was the last pending trace in the run. The trace that finishes last finalizes the run's status and logs the run's aggregate metrics to MLflow ([mlflow_utils.py](../src/workers/mlflow_utils.py)).
4. `GET /eval/runs/{id}` reads back the run plus all its per-trace results.

## API surface

| Endpoint | Method | Description |
|---|---|---|
| `/traces` | POST | Ingest one trace (`input`, `output`, `retrieved_context?`, `metadata?`) → 201 |
| `/traces/{id}` | GET | Fetch one trace → 200 / 404 |
| `/eval/run` | POST | Start a judge-scoring run over `trace_ids` **or** `scope: "all_pending"` (exactly one required) → 202, `{run_id, status, trace_count}` |
| `/eval/runs/{id}` | GET | Poll run status + all per-trace results → 200 / 404 |
| `/health` | GET | Liveness + a real `SELECT 1` DB round-trip → 200 / 503 |
| `/metrics` | GET | Prometheus exposition (request rate/latency), via `prometheus-fastapi-instrumentator` |

Data model (`src/api/models.py`): `traces` → `eval_runs` (1) → `eval_results` (N, one per trace in that run, FK to both `eval_runs` and `traces`).

## Data stores

- **Postgres**: system of record for traces, eval runs, and per-trace results. Migrated with Alembic (`alembic/versions/`), run automatically on API container boot.
- **Redis**: RQ's job queue only — no application data lives here.
- **MLflow**: one MLflow run per `eval_runs` row, logged once the run finishes (params: judge provider/model/trace/worker counts; metrics: `avg_*` scores, `hallucination_rate`, `avg_judge_latency_ms`, succeeded/failed counts). Backed by SQLite + local artifact dir inside the `mlflow` container ([infra/mlflow.Dockerfile](../infra/mlflow.Dockerfile)) — not a separate managed store.

## Observability

- **API**: `GET /metrics` — HTTP request rate/latency, standard `prometheus-fastapi-instrumentator` defaults.
- **Worker**: its own metrics HTTP server on `:9100` ([src/workers/metrics.py](../src/workers/metrics.py)), scraped by Prometheus over the internal Docker network (not published to the host — see comment in `metrics.py` on why a scaled worker fleet can't share one host port). Exposes:
  - `judge_latency_seconds` (histogram, labeled `status=completed|failed`)
  - `evals_completed_total` (counter, labeled `status`) — eval throughput (traces/sec) is `rate(evals_completed_total[...])` in Grafana, the same derivation pattern as the latency histogram's percentiles, not a separately precomputed number
  - `eval_correctness` / `eval_relevance` / `eval_groundedness` (histograms)
  - `eval_hallucination_total` (counter, labeled `hallucination`)
- **Grafana**: dashboard provisioned automatically from [infra/grafana/dashboards/hybrid-rag-evaluator.json](../infra/grafana/dashboards/hybrid-rag-evaluator.json) — request rate, judge latency p50/p95, score distributions (p10/p50/p90) for correctness/relevance/groundedness, hallucination rate, and eval throughput.
- Score-distribution and throughput panels only reflect traces evaluated *after* the worker started (Prometheus can't backfill from Postgres rows created before it was scraping).

## Judge detection benchmark

`scripts/generate_synthetic_traces.py` + `scripts/compute_detection_recall.py` are a
**one-time/on-demand benchmark**, not part of the running stack: they run from the host
against a live stack over HTTP, generate/post labeled synthetic traces, trigger a real
`/eval/run`, and compare the judge's output against the ground-truth labels to produce
recall/precision/false-positive-rate. This is a deliberately different lifecycle from
the Prometheus/Grafana monitoring above — the dashboard is meant to stay live for as
long as the stack runs; the benchmark is meant to be re-run by hand whenever someone
wants a fresh number (e.g. after changing the judge model or prompt), and its output
(`scripts/detection_recall_results.json`) is a point-in-time snapshot, not something CI
regenerates on every commit (see the CI note below).

## Known limitations

These are scoping decisions carried through deliberately, not bugs waiting to be fixed:

- **Sync DB access in the API/worker layer.** `src/api/db.py` uses SQLAlchemy's sync
  `create_engine` + `psycopg2`, and every FastAPI route is `def`, not `async def`.
  Concurrency in this system comes entirely from the Redis/RQ job queue — many workers,
  many jobs in flight — not from an async DB driver. FastAPI runs sync routes in a
  threadpool, so this isn't a correctness bug, but "async" accurately describes the job
  queue, not the database layer. Moving to `asyncpg`/`AsyncSession` would be a real
  rewrite (session lifecycle, Alembic async config, every route and query), not a small
  patch, and hasn't been judged worth it at this scale.
- **The recall benchmark is not CI-gated, on purpose.** It needs a full live stack plus
  a real Ollama instance with a model pulled — neither is available in a GitHub-hosted
  CI runner, and even if it were, re-running an LLM judge against the same fixed traces
  on every commit would produce noisy, non-deterministic pass/fail rather than a
  meaningful regression signal. CI mocks the judge call everywhere (see
  `tests/test_eval_api.py`); the benchmark is a separate, manually-triggered
  measurement.
- **Worker metrics don't survive `--scale worker=N`.** `infra/prometheus.yml` scrapes
  a single static target (`worker:9100`); with multiple worker replicas, Docker's
  round-robin DNS means Prometheus only ever sees whichever one replica a given scrape
  happens to resolve to. Correct multi-replica scraping would need Prometheus's Docker
  service discovery — out of scope here since the benchmark and demo usage in this repo
  only ever run one worker.
- **No persistent/public deployment.** Everything here runs via `docker compose up`
  locally. There is no hosted, always-on instance of this stack, and no public Grafana
  URL — despite the phrasing in some earlier drafts of this project's scope. If that
  changes, this section should be the first thing updated.
