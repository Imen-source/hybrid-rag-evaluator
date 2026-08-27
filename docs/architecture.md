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
| `/eval/runs/{id}/mark_baseline` | POST | Designate this run as the baseline `/compare` measures against → 200 / 400 (not `completed`) / 404 |
| `/eval/runs/{id}/compare` | GET | Diff this run's aggregate metrics against the current baseline run, with a regression verdict → 200 / 400 (not `completed`, is itself the baseline, or no baseline set) / 404 |
| `/health` | GET | Liveness + a real `SELECT 1` DB round-trip → 200 / 503 |
| `/metrics` | GET | Prometheus exposition (request rate/latency), via `prometheus-fastapi-instrumentator` |

Data model (`src/api/models.py`): `traces` → `eval_runs` (1) → `eval_results` (N, one per trace in that run, FK to both `eval_runs` and `traces`). `eval_runs.is_baseline` (migration `0003`) marks the single run `/compare` diffs against; a partial unique index (`is_baseline = true`) enforces at most one at a time.

## Baseline comparison, regression detection, and the gate

`POST /eval/runs/{id}/mark_baseline` marks a completed run as *the* baseline —
a single mutable pointer, not a set. Marking a new run unsets whichever run
was previously the baseline in the same transaction (see the docstring on
`mark_baseline()` in `src/api/eval_routes.py`), rather than requiring an
explicit unmark first: a baseline is meant to move forward over time (e.g.
"the last known-good run"), and requiring an unmark step would just add
friction without protecting anything real. The partial unique index from
migration `0003` is what actually guarantees at most one baseline exists,
independent of that application logic.

`GET /eval/runs/{id}/compare` aggregates both the target run's and the
baseline's completed results with the existing `aggregate_metrics()`
(`dataset_eval.py` — unchanged, reused as-is), diffs them
(`compute_metric_deltas()`), and applies a regression gate
(`evaluate_regression()`) to three of the four compared metrics:

- **`avg_correctness` / `avg_groundedness`**: flagged on a **relative** drop
  of more than 5%. Relative, not absolute, because both are 0–1 scores whose
  baseline level varies a lot by dataset/judge — a 0.05 absolute drop is
  huge against a 0.95 baseline and noise against a 0.20 one.
- **`hallucination_rate`**: flagged on an **absolute** increase of more than
  5 percentage points instead, because it's already a rate: relative framing
  breaks down near a 0 baseline (one hallucination on a previously-perfect
  baseline would be an "infinite" relative increase), and an absolute
  point-increase is what "hallucinations got noticeably more common"
  actually means.
- **`avg_relevance`** is reported in the diff but does not gate — it reflects
  topical/retrieval fit more than correctness or safety, and is the judge's
  noisiest score in practice. Gating on it risked blocking good changes over
  phrasing variance rather than catching real regressions.

A run trips the gate if it crosses **any one** of the three thresholds —
these are independent failure modes, not a combined score. The full
reasoning lives as a docstring on `evaluate_regression()` in
`dataset_eval.py`, next to the thresholds themselves, so the two can't drift
apart.

`scripts/check_regression.py` is the actual, callable gate: it calls
`GET /eval/runs/{id}/compare` for a given run against the current baseline
and exits `1` if `regressed: true` (or if the compare call itself fails —
e.g. no baseline set), `0` otherwise. **It is not wired into
`.github/workflows/ci.yml`.** Every call needs a real, completed eval run
scored by a live judge against a live API + Postgres — the same constraint
that already keeps `scripts/compute_detection_recall.py` out of CI (see
"Known limitations" below). It's meant to be run by hand, or from a
deployment pipeline that already has a live stack and a completed run to
check:

```bash
python -m scripts.check_regression --api-url http://localhost:8000 --run-id <uuid>
```

This is what backs the "regression detection, baselines, and gating" language
used to describe this project elsewhere (e.g. on a CV) — as of this feature,
those are real, callable code paths (`compute_metric_deltas`/
`evaluate_regression` in `dataset_eval.py`, the `/compare` endpoint, and
`scripts/check_regression.py`'s exit code), not just something a human could
eyeball across two Grafana panels or two MLflow runs.

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
- **The regression gate (`scripts/check_regression.py`) is not CI-gated either, for the
  same reason.** It calls a live judge through a live API for both the candidate and
  baseline runs it compares, which a GitHub-hosted runner can't provide without a real
  Ollama/OpenAI backend. Unlike the recall benchmark, this one doesn't need the *same*
  fixed traces to be meaningful — any live stack with a marked baseline works — so
  wiring it into a deployment pipeline that already has one is a real option; it's just
  not part of `ci.yml` today. The endpoint and exit code it relies on
  (`/eval/runs/{id}/compare`, `scripts/check_regression.py`) are tested against real
  Postgres and demonstrated against a real Ollama/phi judge — see the "Baseline
  comparison, regression detection, and the gate" section above.
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
