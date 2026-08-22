# Hybrid RAG Evaluator

Hybrid RAG Evaluator is a local-first evaluation pipeline for retrieval-augmented generation (RAG) systems. It combines lightweight LLM judging with deterministic scoring so the pipeline remains useful even on modest hardware using local models such as phi through Ollama.

## Features

- **Dataset Loading**: JSON dataset with question, context, and answer samples
- **Hybrid Evaluation**: Combines LLM judging with deterministic semantic similarity and keyword overlap
- **Metrics Computed**: correctness, relevance, groundedness, hallucination, explanation, semantic_similarity, keyword_overlap, confidence
- **Aggregate Metrics**: Average scores plus hallucination_rate
- **Caching**: LLM responses and embeddings cached for faster repeated runs
- **Provider Support**: Local Ollama (phi) and OpenAI API (optional)
- **Fallbacks**: Deterministic results if LLM fails or Ollama is unavailable

## Repository Layout

```
rag_evaluator/
├── cli.py               # Entry point
├── metaflow_pipeline.py # Pipeline runner (Metaflow + local fallback)
├── evaluator.py         # LLM evaluation, hybrid scoring, JSON handling
├── dataset_eval.py      # Dataset loading and aggregate metrics
├── prompts.py           # Evaluation prompt templates
├── schemas.py           # Pydantic schemas for structured output
├── similarity.py        # Semantic similarity & keyword overlap helpers
├── cache.py             # Clear cached artifacts
├── examples/            # Sample datasets
├── results.json         # Aggregated results
│
├── src/api/              # FastAPI ingestion + eval-run service (reuses evaluator.py)
├── src/workers/          # eval worker: judge scoring jobs + Prometheus metrics
├── alembic/              # Postgres migrations for the service's traces/eval tables
├── scripts/              # one-time synthetic trace generator + judge recall benchmark
└── infra/                # docker-compose stack: Postgres, Redis, MLflow, API, worker,
                           # Prometheus (infra/prometheus.yml), Grafana (infra/grafana/)
```

### Optional / Cleanup

- `metrics.py` → can be merged into dataset_eval.py (thin wrapper)
- Delete: `test_openai_key.py`, `debug.log`
- Ignore: `__pycache__`, `eval_cache/`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install ollama huggingface_hub sentence-transformers
```

## Usage

### Run with Ollama locally (phi)

```bash
python cli.py --provider ollama --model phi
```

**Optional flags:**

```bash
--use-hybrid true    # Combine LLM + semantic scoring
--num-runs 2         # Number of evaluation passes per sample
--verbose true       # Mirror debug info to console
```

### Run with OpenAI API

```bash
python cli.py --provider openai --model gpt-4o-mini
```

Set your API key:

```powershell
$env:OPENAI_API_KEY="your_openai_key_here"
```

## Service Mode (API + Worker + MLflow)

For team/CI use beyond the single-machine CLI, the same evaluator core (`evaluator.py`) is wrapped in a small async service: a FastAPI ingestion API, a Redis/RQ worker pool that runs judge scoring in the background, Postgres persistence for traces and results, and MLflow experiment tracking per eval run.

```
Client → API (FastAPI) → Postgres (traces, eval_runs, eval_results)
                       └→ Redis/RQ queue → Worker(s) → evaluator.py → MLflow
```

### Quick start

```bash
cp infra/.env.example infra/.env
docker compose -f infra/docker-compose.yml --env-file infra/.env up --build
```

This starts Postgres, Redis, MLflow, the API (Alembic migrations run automatically on boot), and one worker. Scale workers with:

```bash
docker compose -f infra/docker-compose.yml --env-file infra/.env up --build --scale worker=3
```

### API

| Endpoint | Description |
|---|---|
| `POST /traces` | Ingest one RAG trace (`input`, `output`, `retrieved_context`, `metadata`) |
| `GET /traces/{id}` | Fetch a trace |
| `POST /eval/run` | Start a judge-scoring run over `trace_ids` or `scope: "all_pending"` |
| `GET /eval/runs/{id}` | Poll run status and per-trace results |
| `GET /health` | Liveness + DB connectivity check |

Each `/eval/run` fans out one background job per trace to the worker pool; the worker calls `evaluate_answer()` from `evaluator.py` unmodified, persists per-trace scores, and — once every trace in the run has finished — logs the run's aggregate metrics to MLflow.

### Configuration

Judge provider/model and worker behavior are set via env vars (see `infra/.env.example`): `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_USE_HYBRID`, `JUDGE_NUM_RUNS`, `JUDGE_TEMPERATURE`, `OPENAI_API_KEY`, `OLLAMA_HOST`.

### Judge Detection Benchmark (synthetic traces)

`scripts/generate_synthetic_traces.py` and `scripts/compute_detection_recall.py` are one-time/on-demand scripts — not part of the running stack — that measure how well the judge actually catches bad RAG output. They run from the host against a live stack (`docker compose up` from the Quick start above), not inside a container.

```bash
pip install -r requirements-dev.txt   # for httpx, if not already installed

# 1. Generate ~70 labeled traces (12 deliberately bad) and post them to the API.
#    Writes scripts/synthetic_manifest.json (ground truth), checked into the repo.
python -m scripts.generate_synthetic_traces --api-url http://localhost:8000

# 2. Trigger a real eval run over exactly those traces, poll it to completion,
#    and compare the judge's output against the manifest's labels.
python -m scripts.compute_detection_recall --api-url http://localhost:8000
```

The second script prints recall / precision / false-positive-rate and writes the full per-trace breakdown to `scripts/detection_recall_results.json`. Re-running step 2 alone recomputes the numbers against the same traces without regenerating them, as long as `scripts/synthetic_manifest.json` still points at trace IDs that exist in the database.

### Monitoring (Prometheus + Grafana)

Unlike the one-time benchmark scripts above, this is meant to stay live: Prometheus and Grafana start with the rest of the stack (`docker compose up` from the Quick start above) and keep scraping/rendering as long as it's running — there's no separate script to invoke.

- **Grafana**: http://localhost:3000 (default `admin` / `admin`, set via `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`). The "Hybrid RAG Evaluator" dashboard is provisioned automatically on startup — no manual datasource setup or dashboard import.
- **Prometheus**: http://localhost:9090, scraping the API's `GET /metrics` (request rate + latency, via `prometheus-fastapi-instrumentator`) and the worker's own metrics server on port `9100` (judge latency histogram, correctness/relevance/groundedness score histograms, hallucination counter — see `src/workers/metrics.py`). Scrape config: `infra/prometheus.yml`.
- The worker's `/metrics` port (`WORKER_METRICS_PORT`, default 9100) is intentionally not published to the host, since scaling worker replicas (`--scale worker=N`) would collide on one host port; Prometheus reaches it over the internal Docker network instead.
- The dashboard's score-distribution and hallucination-rate panels only reflect traces evaluated *after* the instrumented worker started (Prometheus has no way to backfill historical DB rows) — running the judge detection benchmark above, or hitting `/eval/run` a few times, will populate them.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests spin up real, ephemeral Postgres and Redis containers via testcontainers (Docker required) and run actual Alembic migrations — only the LLM call itself is mocked.

## Hybrid Scoring Logic

Final score combines:

```
0.5 * llm_correctness + 0.3 * semantic_similarity + 0.2 * keyword_overlap
```

If LLM output is malformed or fails → deterministic scoring still returns a valid JSON payload

## Example Dataset

```json
[
  {
    "question": "What is the capital of France?",
    "context": "France's capital city is Paris.",
    "answer": "Paris is the capital of France."
  }
]
```

## Example Output

### Sample-level

```json
{
  "correctness": 0.9234,
  "relevance": 0.9235,
  "groundedness": 0.9234,
  "hallucination": false,
  "explanation": "Paris is the capital city of France.",
  "semantic_similarity": 0.9114,
  "keyword_overlap": 0.75,
  "confidence": 0.9645
}
```

### Aggregate metrics

```json
{
  "avg_correctness": 0.7877,
  "avg_relevance": 0.8552,
  "avg_groundedness": 0.8369,
  "avg_semantic_similarity": 0.8002,
  "avg_keyword_overlap": 0.6548,
  "avg_confidence": 0.9201,
  "hallucination_rate": 0.0,
  "sample_count": 3
}
```

## Logging & Debugging

- Raw LLM output, retries, and fallbacks logged to `debug.log`
- Use `--verbose true` to mirror logs to console
- Malformed output triggers retry with stricter prompt

## Caching

- **LLM results**: `eval_cache/`
- **Embeddings**: `eval_cache/embeddings/`

### Clear cache

```bash
python cache.py --cache-dir eval_cache
```

## Known Issues

- Small local models (e.g., phi) may produce inconsistent explanations
- Semantic similarity helps but is not a replacement for a gold reference
- Keyword overlap is lightweight; paraphrases can reduce accuracy
- Ollama must be running; otherwise deterministic fallback is used
- Windows users may see HuggingFace symlink warnings — non-blocking

## Future Improvements

- Support for larger local models if more system memory is available
- Improved fallback handling for multi-step or long-context questions
- Optional visualizations for metric distributions
- Better error reporting for hybrid mode inconsistencies
- Potential integration with cloud LLMs beyond OpenAI

## Sanity Checks

### Syntax check

```bash
python -m py_compile cli.py dataset_eval.py evaluator.py similarity.py schemas.py prompts.py metaflow_pipeline.py cache.py
```

### Reset cache

```bash
python cache.py
```

## Goals Achieved

- ✅ Fully functional local-first RAG evaluation pipeline
- ✅ Supports multiple LLM providers: OpenAI + Ollama (phi)
- ✅ Structured JSON output compatible with downstream metrics
- ✅ Optional hybrid semantic scoring
- ✅ Clean, modular, production-ready code
- ✅ Optional service mode: FastAPI + Postgres + Redis/RQ worker + MLflow tracking