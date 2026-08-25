"""Eval-run tests backed by real Postgres + real Redis/RQ + real MLflow
(file-based, temp dir).

Only the LLM call and the sentence-transformers embedding load are mocked
(per the "CI shouldn't make real API calls" requirement, plus the embedding
model requires a real network download otherwise). Everything else --
job enqueue/dequeue through real RQ, DB status transitions, run
finalization, and MLflow run creation -- runs for real, and MLflow runs are
read back and asserted on, not just "no exception thrown".
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from redis import Redis
from rq import Queue
from rq.worker import SimpleWorker

# postgres_db_url/redis_db_url/api_app come from the root conftest.py.

FAILURE_MARKER = "TRIGGER_FAILURE"


@pytest.fixture(scope="session")
def eval_client(api_app, redis_db_url):
    # tempfile.mkdtemp(), not pytest's tmp_path_factory: this machine has a
    # stale, permission-locked leftover under the default pytest temp root
    # (C:\Users\<user>\AppData\Local\Temp\pytest-of-<user>) from an unrelated
    # prior session, which makes tmp_path_factory error out entirely.
    session_dir = tempfile.mkdtemp(prefix="eval_api_test_")
    mlruns_dir = os.path.join(session_dir, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{mlruns_dir}"
    os.environ["MLFLOW_EXPERIMENT"] = "test-eval-runs"
    os.environ["JUDGE_PROVIDER"] = "openai"
    os.environ["JUDGE_MODEL"] = "gpt-4o-mini-test"
    os.environ["JUDGE_NUM_RUNS"] = "1"
    os.environ["EVAL_WORKER_COUNT"] = "2"
    # A fresh, unique cache dir per test session: evaluate_answer() caches by
    # (question, context, answer, config) and would otherwise return a
    # stale cached result from a prior local test run without calling the
    # mocked judge/similarity functions at all, breaking the mock assertions.
    os.environ["JUDGE_CACHE_DIR"] = os.path.join(session_dir, "eval_cache")

    with TestClient(api_app) as client:
        yield client


def _drain_eval_queue() -> None:
    """Process every enqueued job synchronously, in-process (no fork -- RQ's
    default Worker requires os.fork, unavailable on Windows; SimpleWorker
    runs jobs in the same process, which is also what makes the mock.patch
    below actually take effect for the job)."""
    redis_url = os.environ["REDIS_URL"]
    conn = Redis.from_url(redis_url)
    queue = Queue("eval", connection=conn)
    worker = SimpleWorker([queue], connection=conn)
    worker.work(burst=True)


def _fake_call_model(prompt: str, config):
    payload = {
        "correctness": 0.9,
        "relevance": 0.85,
        "groundedness": 0.8,
        "hallucination": False,
        "explanation": "Mock judge response.",
        "confidence": 0.9,
    }
    return json.dumps(payload), config.model or "mock-judge-model"


def _fake_semantic_similarity(left_text, right_text, cache_dir, model_name=None):
    if FAILURE_MARKER in left_text or FAILURE_MARKER in right_text:
        raise RuntimeError("simulated embedding failure")
    return 0.75


def _create_trace(client, **overrides):
    payload = {
        "input": "What is the capital of France?",
        "output": "Paris is the capital of France.",
        "retrieved_context": "France's capital city is Paris.",
    }
    payload.update(overrides)
    resp = client.post("/traces", json=payload)
    assert resp.status_code == 201
    return resp.json()["id"]


@patch("similarity.semantic_similarity", side_effect=_fake_semantic_similarity)
@patch("evaluator.call_model", side_effect=_fake_call_model)
def test_eval_run_end_to_end_with_explicit_trace_ids(mock_call_model, mock_similarity, eval_client):
    trace_id_1 = _create_trace(eval_client)
    trace_id_2 = _create_trace(eval_client, output="Berlin is the capital of Germany.")

    create_resp = eval_client.post("/eval/run", json={"trace_ids": [trace_id_1, trace_id_2]})
    assert create_resp.status_code == 202
    created = create_resp.json()
    assert created["status"] == "running"
    assert created["trace_count"] == 2
    run_id = created["run_id"]

    _drain_eval_queue()

    get_resp = eval_client.get(f"/eval/runs/{run_id}")
    assert get_resp.status_code == 200
    run = get_resp.json()

    assert run["status"] == "completed"
    assert run["judge_provider"] == "openai"
    assert run["judge_model"] == "gpt-4o-mini-test"
    assert run["worker_count"] == 2
    assert run["trace_count"] == 2
    assert run["mlflow_run_id"] is not None
    assert len(run["results"]) == 2

    result_trace_ids = {r["trace_id"] for r in run["results"]}
    assert result_trace_ids == {trace_id_1, trace_id_2}
    for result in run["results"]:
        assert result["status"] == "completed"
        assert result["correctness"] is not None
        assert 0.0 <= result["correctness"] <= 1.0
        assert result["latency_ms"] is not None
        assert result["error"] is None

    assert mock_call_model.called


@patch("similarity.semantic_similarity", side_effect=_fake_semantic_similarity)
@patch("evaluator.call_model", side_effect=_fake_call_model)
def test_mlflow_run_is_created_and_queryable(mock_call_model, mock_similarity, eval_client):
    from mlflow.tracking import MlflowClient

    trace_id = _create_trace(eval_client)

    create_resp = eval_client.post("/eval/run", json={"trace_ids": [trace_id]})
    run_id = create_resp.json()["run_id"]

    _drain_eval_queue()

    run = eval_client.get(f"/eval/runs/{run_id}").json()
    mlflow_run_id = run["mlflow_run_id"]
    assert mlflow_run_id

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    mlflow_run = client.get_run(mlflow_run_id)  # raises if it doesn't really exist

    assert mlflow_run.data.params["judge_provider"] == "openai"
    assert mlflow_run.data.params["judge_model"] == "gpt-4o-mini-test"
    assert mlflow_run.data.params["trace_count"] == "1"
    assert mlflow_run.data.params["eval_run_id"] == run_id

    assert "avg_correctness" in mlflow_run.data.metrics
    assert "hallucination_rate" in mlflow_run.data.metrics
    assert mlflow_run.data.metrics["succeeded_trace_count"] == 1.0
    assert mlflow_run.data.metrics["failed_trace_count"] == 0.0
    assert mlflow_run.data.metrics["avg_judge_latency_ms"] >= 0.0


@patch("similarity.semantic_similarity", side_effect=_fake_semantic_similarity)
@patch("evaluator.call_model", side_effect=_fake_call_model)
def test_eval_run_partial_failure_still_completes(mock_call_model, mock_similarity, eval_client):
    ok_trace_id = _create_trace(eval_client)
    failing_trace_id = _create_trace(eval_client, output=f"{FAILURE_MARKER} answer")

    create_resp = eval_client.post("/eval/run", json={"trace_ids": [ok_trace_id, failing_trace_id]})
    run_id = create_resp.json()["run_id"]

    _drain_eval_queue()

    run = eval_client.get(f"/eval/runs/{run_id}").json()
    assert run["status"] == "completed"  # at least one trace succeeded

    results_by_id = {r["trace_id"]: r for r in run["results"]}
    assert results_by_id[ok_trace_id]["status"] == "completed"
    assert results_by_id[failing_trace_id]["status"] == "failed"
    assert results_by_id[failing_trace_id]["error"] is not None


def test_eval_run_scope_all_pending_excludes_already_completed(eval_client):
    with patch("similarity.semantic_similarity", side_effect=_fake_semantic_similarity), patch(
        "evaluator.call_model", side_effect=_fake_call_model
    ):
        completed_trace_id = _create_trace(eval_client, input="already scored question")
        pending_trace_id = _create_trace(eval_client, input="not yet scored question")

        first_run = eval_client.post("/eval/run", json={"trace_ids": [completed_trace_id]})
        first_run_id = first_run.json()["run_id"]
        _drain_eval_queue()
        assert eval_client.get(f"/eval/runs/{first_run_id}").json()["status"] == "completed"

        second_run_resp = eval_client.post("/eval/run", json={"scope": "all_pending"})
        assert second_run_resp.status_code == 202
        second_run = second_run_resp.json()
        _drain_eval_queue()

        run_detail = eval_client.get(f"/eval/runs/{second_run['run_id']}").json()
        result_trace_ids = {r["trace_id"] for r in run_detail["results"]}
        assert completed_trace_id not in result_trace_ids
        assert pending_trace_id in result_trace_ids


def test_eval_run_unknown_trace_id_returns_404(eval_client):
    resp = eval_client.post("/eval/run", json={"trace_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404


def test_eval_run_rejects_both_trace_ids_and_scope(eval_client):
    trace_id = _create_trace(eval_client)
    resp = eval_client.post(
        "/eval/run", json={"trace_ids": [trace_id], "scope": "all_pending"}
    )
    assert resp.status_code == 422


def test_eval_run_rejects_neither_trace_ids_nor_scope(eval_client):
    resp = eval_client.post("/eval/run", json={})
    assert resp.status_code == 422


def test_get_eval_run_missing_returns_404(eval_client):
    resp = eval_client.get(f"/eval/runs/{uuid.uuid4()}")
    assert resp.status_code == 404
