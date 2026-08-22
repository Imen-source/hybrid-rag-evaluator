"""API tests backed by a real, ephemeral Postgres (via testcontainers).

No mocking of the database: every test hits an actual Postgres instance
running in a Docker container, migrated with the real Alembic revisions.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

# postgres_db_url/api_app come from the root conftest.py: one Postgres
# container shared for the whole test session, not one per file.


@pytest.fixture(scope="session")
def test_client(api_app):
    with TestClient(api_app) as client:
        yield client


def _sample_payload(**overrides):
    payload = {
        "input": "What is the capital of France?",
        "output": "Paris is the capital of France.",
        "retrieved_context": "France's capital city is Paris.",
        "metadata": {"model": "gpt-4o-mini"},
    }
    payload.update(overrides)
    return payload


def test_create_then_retrieve_trace(test_client):
    create_resp = test_client.post("/traces", json=_sample_payload())
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["input"] == "What is the capital of France?"
    assert created["output"] == "Paris is the capital of France."
    assert created["retrieved_context"] == "France's capital city is Paris."
    assert created["metadata"] == {"model": "gpt-4o-mini"}
    uuid.UUID(created["id"])  # raises if not a valid UUID

    get_resp = test_client.get(f"/traces/{created['id']}")
    assert get_resp.status_code == 200
    fetched = get_resp.json()
    assert fetched["id"] == created["id"]
    assert fetched["output"] == "Paris is the capital of France."
    assert fetched["metadata"] == {"model": "gpt-4o-mini"}


def test_create_trace_without_metadata_or_context_defaults_ok(test_client):
    payload = {
        "input": "ping",
        "output": "pong",
    }
    resp = test_client.post("/traces", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["retrieved_context"] is None
    assert body["metadata"] == {}


def test_get_missing_trace_returns_404(test_client):
    missing_id = str(uuid.uuid4())
    resp = test_client.get(f"/traces/{missing_id}")
    assert resp.status_code == 404


def test_create_trace_validation_error_on_malformed_payload(test_client):
    # Missing required "output" field.
    resp = test_client.post("/traces", json={"input": "only input"})
    assert resp.status_code == 422


def test_health_checks_db_connectivity(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
