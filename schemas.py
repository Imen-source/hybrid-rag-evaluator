"""Pydantic schemas for validated pipeline artifacts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetSample(BaseModel):
    """One dataset row used in the evaluation pipeline."""

    question: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class EvaluationResult(BaseModel):
    """Validated judge output for one answer."""

    correctness: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    groundedness: float = Field(..., ge=0.0, le=1.0)
    hallucination: bool
    explanation: str
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    keyword_overlap: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class AggregatedMetrics(BaseModel):
    """Dataset-level summary produced after all samples are evaluated."""

    avg_correctness: float = Field(..., ge=0.0, le=1.0)
    avg_relevance: float = Field(..., ge=0.0, le=1.0)
    avg_groundedness: float = Field(..., ge=0.0, le=1.0)
    avg_semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    avg_keyword_overlap: float = Field(..., ge=0.0, le=1.0)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    hallucination_rate: float = Field(..., ge=0.0, le=1.0)
    sample_count: int = Field(..., ge=1)
