"""Deterministic similarity helpers for hybrid RAG evaluation."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import diskcache as dc

LOGGER = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace to stabilize caching and comparisons."""
    return " ".join(text.split())


def tokenize_words(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def keyword_overlap(left_text: str, right_text: str) -> float:
    """Return lexical overlap between two strings."""
    left_tokens = tokenize_words(left_text)
    right_tokens = tokenize_words(right_text)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def cosine_similarity_from_vectors(left_vector: list[float], right_vector: list[float]) -> float:
    """Compute cosine similarity using two dense vectors."""
    dot_product = sum(left * right for left, right in zip(left_vector, right_vector))
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str):
    """Load the sentence-transformer model lazily."""
    from sentence_transformers import SentenceTransformer

    LOGGER.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def build_embedding_cache_key(model_name: str, text: str) -> str:
    """Create a stable cache key for one embedding request."""
    text_digest = hashlib.sha256(normalize_whitespace(text).encode("utf-8")).hexdigest()
    return f"{model_name}:{text_digest}"


def get_cached_embedding(
    text: str,
    cache_dir: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float] | None:
    """Return a cached embedding vector for the given text."""
    normalized_text = normalize_whitespace(text)
    if not normalized_text:
        return None

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache = dc.Cache(cache_dir)
    cache_key = build_embedding_cache_key(model_name, normalized_text)
    try:
        cached_embedding = cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding

        embedding_model = load_embedding_model(model_name)
        embedding = embedding_model.encode([normalized_text])[0]
        embedding_vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        cache[cache_key] = embedding_vector
        return embedding_vector
    finally:
        cache.close()


def semantic_similarity(
    left_text: str,
    right_text: str,
    cache_dir: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> float:
    """Compute cosine similarity using locally cached sentence embeddings."""
    try:
        left_embedding = get_cached_embedding(left_text, cache_dir=cache_dir, model_name=model_name)
        right_embedding = get_cached_embedding(right_text, cache_dir=cache_dir, model_name=model_name)
        if left_embedding is None or right_embedding is None:
            return 0.0
        return float(cosine_similarity_from_vectors(left_embedding, right_embedding))
    except Exception as exc:
        LOGGER.exception("Semantic similarity failed: %s", exc)
        return 0.0


def similarity_breakdown(
    answer: str,
    question: str,
    context: str,
    cache_dir: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Compute deterministic signals used in hybrid evaluation."""
    semantic_answer_context = semantic_similarity(
        answer,
        context,
        cache_dir=cache_dir,
        model_name=model_name,
    )
    semantic_answer_question = semantic_similarity(
        answer,
        question,
        cache_dir=cache_dir,
        model_name=model_name,
    )
    keyword_answer_context = keyword_overlap(answer, context)
    keyword_answer_question = keyword_overlap(answer, question)

    return {
        "semantic_answer_context": semantic_answer_context,
        "semantic_answer_question": semantic_answer_question,
        "keyword_answer_context": keyword_answer_context,
        "keyword_answer_question": keyword_answer_question,
        "semantic_similarity": max(
            0.0,
            min(1.0, (semantic_answer_context + semantic_answer_question) / 2.0),
        ),
        "keyword_overlap": max(
            0.0,
            min(1.0, (keyword_answer_context + keyword_answer_question) / 2.0),
        ),
    }
