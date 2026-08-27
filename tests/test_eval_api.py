"""Eval-run tests backed by real Postgres + real Redis/RQ + real MLflow
(file-based, temp dir).

Only the LLM call and the sentence-transformers embedding load are mocked --
CI has no LLM API key and the embedding model would otherwise need a real
network download. Everything else -- job enqueue/dequeue through real RQ,
DB status transitions, run finalization, and MLflow run creation -- runs for
real, and MLflow runs are read back and asserted on, not just "no exception
thrown".
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
    # tempfile.mkdtemp(), not pytest's tmp_path_factory: on Windows a
    # permission-locked leftover under the default pytest temp root can make
    # tmp_path_factory error out entirely rather than just skip it.
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


def _make_fake_call_model(correctness: float, relevance: float, groundedness: float, hallucination: bool):
    """Build a call_model mock that returns fixed, caller-chosen scores --
    used by the compare/regression tests below to control aggregate metrics
    precisely instead of relying on the single fixed _fake_call_model above."""

    def _fake(prompt: str, config):
        payload = {
            "correctness": correctness,
            "relevance": relevance,
            "groundedness": groundedness,
            "hallucination": hallucination,
            "explanation": "Mock judge response.",
            "confidence": 0.9,
        }
        return json.dumps(payload), config.model or "mock-judge-model"

    return _fake


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
    mlflow_run = client.get_run(mlflow_run_id)

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


# --- baseline / compare / regression -----------------------------------


def _run_eval_with_fixed_scores(
    client, correctness: float, relevance: float, groundedness: float, hallucination: bool
) -> str:
    """Ingest one trace, score it with caller-chosen fixed judge scores, and
    return the resulting (completed) run id.

    Each call uses a unique question: evaluate_answer() caches by
    (question, context, answer, config), and this fixture's cache dir is
    shared for the whole test session -- reusing _create_trace's default
    question across calls with different fixed scores would return a stale
    cached result instead of exercising the newly-patched mock.

    Forces JUDGE_USE_HYBRID=false for the duration of the call: in hybrid
    mode (the fixture's default), evaluator.fuse_scores() blends the mocked
    judge's correctness/relevance/groundedness with semantic-similarity and
    keyword-overlap signals computed from the actual trace text, so the
    persisted EvalResult values would NOT equal the numbers passed in here.
    Non-hybrid mode still blends groundedness with the (mocked, constant
    0.75) semantic-context score -- see _expected_groundedness() below --
    but leaves correctness/relevance as pure pass-throughs of the mocked
    judge output, which is what makes the regression-math assertions in the
    tests below exact instead of approximate-and-hand-wavy.
    """
    trace_id = _create_trace(client, input=f"What is the capital of France? [{uuid.uuid4()}]")
    previous_use_hybrid = os.environ.get("JUDGE_USE_HYBRID")
    os.environ["JUDGE_USE_HYBRID"] = "false"
    try:
        with patch("similarity.semantic_similarity", side_effect=_fake_semantic_similarity), patch(
            "evaluator.call_model",
            side_effect=_make_fake_call_model(correctness, relevance, groundedness, hallucination),
        ):
            create_resp = client.post("/eval/run", json={"trace_ids": [trace_id]})
            run_id = create_resp.json()["run_id"]
            _drain_eval_queue()
    finally:
        if previous_use_hybrid is None:
            os.environ.pop("JUDGE_USE_HYBRID", None)
        else:
            os.environ["JUDGE_USE_HYBRID"] = previous_use_hybrid
    return run_id


def _expected_groundedness(llm_groundedness: float) -> float:
    """Mirrors evaluator.fuse_scores()'s non-hybrid groundedness formula:
    0.8 * llm_groundedness + 0.2 * semantic_context_score, where
    semantic_context_score is pinned to 0.75 by _fake_semantic_similarity."""
    return 0.8 * llm_groundedness + 0.2 * 0.75


# NOTE on ordering: eval_client is session-scoped and there is no
# "unmark baseline" endpoint (mark_baseline() only ever moves the single
# baseline pointer forward, see its docstring), so this "no baseline set"
# case can only be observed before any other test in this file marks one.
# It's deliberately defined first among the baseline/compare tests.
def test_compare_returns_400_when_no_baseline_set(eval_client):
    run_id = _run_eval_with_fixed_scores(eval_client, 0.9, 0.85, 0.9, False)

    resp = eval_client.get(f"/eval/runs/{run_id}/compare")
    assert resp.status_code == 400


def test_mark_baseline_marks_a_completed_run(eval_client):
    run_id = _run_eval_with_fixed_scores(eval_client, 0.9, 0.85, 0.9, False)

    resp = eval_client.post(f"/eval/runs/{run_id}/mark_baseline")
    assert resp.status_code == 200
    assert resp.json()["is_baseline"] is True

    fetched = eval_client.get(f"/eval/runs/{run_id}").json()
    assert fetched["is_baseline"] is True


def test_mark_baseline_rejects_run_that_is_not_completed(eval_client):
    trace_id = _create_trace(eval_client)
    # Deliberately not draining the queue -- the run stays "running".
    create_resp = eval_client.post("/eval/run", json={"trace_ids": [trace_id]})
    run_id = create_resp.json()["run_id"]
    assert create_resp.json()["status"] == "running"

    resp = eval_client.post(f"/eval/runs/{run_id}/mark_baseline")
    assert resp.status_code == 400

    # Drain so this run doesn't leak a stuck job into later tests.
    with patch("similarity.semantic_similarity", side_effect=_fake_semantic_similarity), patch(
        "evaluator.call_model", side_effect=_fake_call_model
    ):
        _drain_eval_queue()


def test_mark_baseline_missing_run_returns_404(eval_client):
    resp = eval_client.post(f"/eval/runs/{uuid.uuid4()}/mark_baseline")
    assert resp.status_code == 404


def test_mark_baseline_unsets_previous_baseline(eval_client):
    run_a = _run_eval_with_fixed_scores(eval_client, 0.9, 0.85, 0.9, False)
    run_b = _run_eval_with_fixed_scores(eval_client, 0.7, 0.85, 0.9, False)

    resp_a = eval_client.post(f"/eval/runs/{run_a}/mark_baseline")
    assert resp_a.json()["is_baseline"] is True

    resp_b = eval_client.post(f"/eval/runs/{run_b}/mark_baseline")
    assert resp_b.json()["is_baseline"] is True

    # Only run_b should be the baseline now -- run_a's flag was unset as
    # part of marking run_b, per mark_baseline()'s documented "single
    # mutable pointer" semantics.
    assert eval_client.get(f"/eval/runs/{run_a}").json()["is_baseline"] is False
    assert eval_client.get(f"/eval/runs/{run_b}").json()["is_baseline"] is True


def test_compare_missing_run_returns_404(eval_client):
    baseline_id = _run_eval_with_fixed_scores(eval_client, 0.9, 0.85, 0.9, False)
    eval_client.post(f"/eval/runs/{baseline_id}/mark_baseline")

    resp = eval_client.get(f"/eval/runs/{uuid.uuid4()}/compare")
    assert resp.status_code == 404


def test_compare_rejects_run_that_is_not_completed(eval_client):
    baseline_id = _run_eval_with_fixed_scores(eval_client, 0.9, 0.85, 0.9, False)
    eval_client.post(f"/eval/runs/{baseline_id}/mark_baseline")

    trace_id = _create_trace(eval_client)
    create_resp = eval_client.post("/eval/run", json={"trace_ids": [trace_id]})
    running_run_id = create_resp.json()["run_id"]

    resp = eval_client.get(f"/eval/runs/{running_run_id}/compare")
    assert resp.status_code == 400

    with patch("similarity.semantic_similarity", side_effect=_fake_semantic_similarity), patch(
        "evaluator.call_model", side_effect=_fake_call_model
    ):
        _drain_eval_queue()


def test_compare_detects_regression_against_baseline(eval_client):
    baseline_id = _run_eval_with_fixed_scores(eval_client, 0.90, 0.85, 0.90, False)
    eval_client.post(f"/eval/runs/{baseline_id}/mark_baseline")

    # Correctness/groundedness drop well past the 5% relative-drop bar, and
    # hallucination flips on -- all three gated metrics should trip.
    candidate_id = _run_eval_with_fixed_scores(eval_client, 0.50, 0.80, 0.40, True)

    resp = eval_client.get(f"/eval/runs/{candidate_id}/compare")
    assert resp.status_code == 200
    body = resp.json()

    assert body["candidate_run_id"] == candidate_id
    assert body["baseline_run_id"] == baseline_id
    assert body["regressed"] is True
    assert len(body["regressed_reasons"]) == 3

    correctness = body["metrics"]["avg_correctness"]
    assert correctness["baseline"] == pytest.approx(0.90)
    assert correctness["candidate"] == pytest.approx(0.50)
    assert correctness["delta"] == pytest.approx(-0.40)
    assert correctness["relative_delta"] == pytest.approx(-0.40 / 0.90)

    groundedness = body["metrics"]["avg_groundedness"]
    expected_baseline_groundedness = _expected_groundedness(0.90)
    expected_candidate_groundedness = _expected_groundedness(0.40)
    assert groundedness["baseline"] == pytest.approx(expected_baseline_groundedness)
    assert groundedness["candidate"] == pytest.approx(expected_candidate_groundedness)

    hallucination = body["metrics"]["hallucination_rate"]
    assert hallucination["baseline"] == pytest.approx(0.0)
    assert hallucination["candidate"] == pytest.approx(1.0)

    assert body["thresholds"] == {
        "correctness_relative_drop": 0.05,
        "groundedness_relative_drop": 0.05,
        "hallucination_rate_absolute_increase": 0.05,
    }


def test_compare_reports_no_regression_within_threshold(eval_client):
    baseline_id = _run_eval_with_fixed_scores(eval_client, 0.90, 0.85, 0.90, False)
    eval_client.post(f"/eval/runs/{baseline_id}/mark_baseline")

    # Correctness drops ~2.2% (raw, non-hybrid pass-through) and groundedness
    # (fused with the mocked 0.75 semantic-context score either way) drops
    # ~1.8% -- both under the 5% bar.
    candidate_id = _run_eval_with_fixed_scores(eval_client, 0.88, 0.85, 0.88, False)

    resp = eval_client.get(f"/eval/runs/{candidate_id}/compare")
    assert resp.status_code == 200
    body = resp.json()

    assert body["regressed"] is False
    assert body["regressed_reasons"] == []
