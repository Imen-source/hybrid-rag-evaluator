"""Pydantic request/response models for the ingestion API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TraceCreate(BaseModel):
    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    retrieved_context: str | None = None
    metadata: dict = Field(default_factory=dict)


class TraceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    timestamp: datetime
    input: str
    output: str
    retrieved_context: str | None
    metadata: dict = Field(validation_alias="meta")


class EvalRunCreate(BaseModel):
    trace_ids: list[uuid.UUID] | None = None
    scope: Literal["all_pending"] | None = None

    @model_validator(mode="after")
    def _check_target(self) -> EvalRunCreate:
        has_ids = bool(self.trace_ids)
        has_scope = self.scope is not None
        if has_ids == has_scope:
            raise ValueError(
                "Provide exactly one of a non-empty trace_ids list or scope='all_pending'."
            )
        return self


class EvalRunCreated(BaseModel):
    run_id: uuid.UUID
    status: str
    trace_count: int


class EvalResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    trace_id: uuid.UUID
    status: str
    correctness: float | None
    relevance: float | None
    groundedness: float | None
    hallucination: bool | None
    explanation: str | None
    semantic_similarity: float | None
    keyword_overlap: float | None
    confidence: float | None
    latency_ms: float | None
    error: str | None


class EvalRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    created_at: datetime
    updated_at: datetime
    judge_provider: str
    judge_model: str
    trace_count: int
    worker_count: int
    mlflow_run_id: str | None
    error: str | None
    is_baseline: bool
    results: list[EvalResultRead] = Field(default_factory=list)


class MetricDelta(BaseModel):
    baseline: float
    candidate: float
    delta: float
    relative_delta: float | None


class RegressionThresholds(BaseModel):
    correctness_relative_drop: float
    groundedness_relative_drop: float
    hallucination_rate_absolute_increase: float


class EvalRunCompareRead(BaseModel):
    candidate_run_id: uuid.UUID
    baseline_run_id: uuid.UUID
    metrics: dict[str, MetricDelta]
    thresholds: RegressionThresholds
    regressed: bool
    regressed_reasons: list[str]
