# Hybrid RAG Evaluator

A local-first evaluation pipeline for retrieval-augmented generation (RAG) systems. It scores `(question, context, answer)` triples on correctness, relevance, groundedness, and hallucination, combining LLM-as-judge scoring with deterministic semantic-similarity and keyword-overlap checks so it stays useful even on modest hardware with a local model (Ollama's `phi`) or against OpenAI.

Two ways to use it:

- **CLI** (`cli.py`) — run the evaluator directly against a JSON dataset on one machine.
- **Service mode** (`src/api/`, `src/workers/`) — a FastAPI ingestion API, a Redis/RQ worker pool, Postgres persistence, and MLflow tracking, so traces can be ingested and scored asynchronously and the results tracked over time. Monitored with Prometheus + Grafana.

See [docs/architecture.md](docs/architecture.md) for the full request flow, data model, and known scoping limitations.

## Architecture

```
Client → API (FastAPI) → Postgres (traces, eval_runs, eval_results)
                       └→ Redis/RQ queue → Worker(s) → evaluator.py → MLflow
```

The worker and the CLI both call the same `evaluate_answer()` in `evaluator.py` — service mode is a thin async wrapper around the same judge core, not a separate implementation.

## Quick start (service mode)

```bash
cp infra/.env.example infra/.env
docker compose -f infra/docker-compose.yml --env-file infra/.env up --build
```

This is a cold start: it builds the API, worker, and MLflow images; starts Postgres, Redis, and MLflow; runs Alembic migrations automatically on the API container's boot; and brings up one worker plus Prometheus and Grafana. No manual setup steps beyond copying the env file.

```bash
# ingest a trace
curl -X POST localhost:8000/traces -H 'content-type: application/json' -d '{
  "input": "What is the capital of France?",
  "output": "Paris is the capital of France.",
  "retrieved_context": "France'"'"'s capital city is Paris."
}'

# score everything pending
curl -X POST localhost:8000/eval/run -H 'content-type: application/json' -d '{"scope": "all_pending"}'
```

| Endpoint | Description |
|---|---|
| `POST /traces` | Ingest one RAG trace (`input`, `output`, `retrieved_context`, `metadata`) |
| `GET /traces/{id}` | Fetch a trace |
| `POST /eval/run` | Start a judge-scoring run over `trace_ids` or `scope: "all_pending"` |
| `GET /eval/runs/{id}` | Poll run status and per-trace results |
| `POST /eval/runs/{id}/mark_baseline` | Mark a completed run as the baseline `/compare` measures against |
| `GET /eval/runs/{id}/compare` | Diff a run's aggregate metrics against the baseline, with a `regressed: true/false` verdict |
| `GET /health` | Liveness + DB connectivity check |
| `GET /metrics` | Prometheus exposition (request rate/latency) |

Scale workers with `--scale worker=N`. Judge provider/model and worker behavior are set via env vars in `infra/.env` (see `infra/.env.example`): `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_USE_HYBRID`, `JUDGE_NUM_RUNS`, `JUDGE_TEMPERATURE`, `OPENAI_API_KEY`, `OLLAMA_HOST`.

## CLI (single machine)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install ollama huggingface_hub sentence-transformers

python cli.py --provider ollama --model phi --dataset examples/sample.json
```

Flags: `--use-hybrid true` (combine LLM + semantic scoring), `--num-runs 2` (evaluation passes per sample, averaged), `--verbose true`. `--provider openai --model gpt-4o-mini` works the same way with `OPENAI_API_KEY` set.

Final correctness score: `0.5 * llm_correctness + 0.3 * semantic_similarity + 0.2 * keyword_overlap`. If the LLM call fails or produces malformed output, deterministic scoring still returns a valid result.

## Judge detection benchmark

`scripts/generate_synthetic_traces.py` and `scripts/compute_detection_recall.py` are a one-time/on-demand benchmark (not part of the running stack): they generate 70 labeled synthetic traces (58 good, 12 deliberately bad — via context-swap or a hallucinated answer), post them to a live stack, trigger a real eval run, and compare the judge's output against the ground-truth labels.

```bash
pip install -r requirements-dev.txt
python -m scripts.generate_synthetic_traces --api-url http://localhost:8000
python -m scripts.compute_detection_recall --api-url http://localhost:8000
```

**Measured result** (`scripts/detection_recall_results.json`, local Ollama `phi`, single CPU machine):

| Recall | Precision | False positive rate |
|---|---|---|
| 33.3% | 66.7% | 3.4% |

`phi` is a small model running on CPU — it misses about two-thirds of the injected bad traces. This is the judge's real, measured detection rate on this benchmark, not a target or an estimate; a larger or hosted model would be expected to score higher, but that hasn't been measured here.

## Baseline comparison, regression detection, and gating

```bash
# after a run finishes:
curl -X POST localhost:8000/eval/runs/<run_id>/mark_baseline

# after a later run finishes, compare it against whichever run is marked baseline:
curl localhost:8000/eval/runs/<other_run_id>/compare
```

`/compare` returns per-metric deltas (`avg_correctness`, `avg_relevance`, `avg_groundedness`, `hallucination_rate`: baseline value, candidate value, absolute delta, relative delta) and a `regressed: true/false` verdict. The gate: correctness/groundedness dropping more than 5% relative to baseline, or hallucination rate rising more than 5 percentage points, count as a regression (relevance is reported but doesn't gate — reasoning in [docs/architecture.md](docs/architecture.md)).

`scripts/check_regression.py` calls `/compare` for a given run and exits non-zero if `regressed: true` — something a CI step or deployment script can actually act on:

```bash
python -m scripts.check_regression --api-url http://localhost:8000 --run-id <uuid>
```

**Not wired into `.github/workflows/ci.yml`** — every call needs a real, completed eval run scored by a live judge, which a GitHub-hosted runner doesn't have (same reason the detection benchmark above isn't CI-gated). The gate itself is real and independently callable; see [docs/architecture.md](docs/architecture.md#baseline-comparison-regression-detection-and-the-gate) for the full design and a real run's `/compare` output.

## Monitoring (Prometheus + Grafana)

Prometheus and Grafana start with the rest of the stack and stay live for as long as it's running — no separate script to invoke.

- **Grafana**: http://localhost:3000 (default `admin`/`admin`). The dashboard is provisioned automatically: request rate, judge latency p50/p95, eval throughput, correctness/relevance/groundedness distributions, hallucination rate.
- **Prometheus**: http://localhost:9090, scraping the API's `/metrics` and the worker's own metrics server (`src/workers/metrics.py`).
- Score-distribution and throughput panels only reflect traces evaluated *after* the worker started — run the benchmark above, or hit `/eval/run` a few times, to populate them.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Tests spin up real, ephemeral Postgres and Redis containers via testcontainers (Docker required) and run the actual Alembic migrations; only the LLM call and the embedding-model load are mocked. CI (`.github/workflows/ci.yml`) runs lint (ruff) → this test suite → a build of all three Docker images, on every push. The Ollama-dependent detection benchmark above is not part of CI — see `docs/architecture.md` for why.

## Status

Runs locally via `docker compose`; there is no persistent or public deployment of this stack. See [docs/architecture.md#known-limitations](docs/architecture.md#known-limitations) for the scoping decisions behind that and a few other design tradeoffs (sync DB access under an otherwise-async job queue, single-target Prometheus scraping under `--scale`).

## Known issues

- Small local models (e.g. `phi`) can produce inconsistent explanations and, per the benchmark above, miss a meaningful share of bad traces.
- Semantic similarity is a helpful signal but not a replacement for a gold reference answer.
- Ollama must be running for the `ollama` provider; otherwise deterministic fallback scoring is used.
- Windows users may see HuggingFace symlink warnings from `sentence-transformers` — non-blocking.

## License

MIT — see [LICENSE](LICENSE).
