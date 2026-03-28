"""Hybrid LLM and deterministic evaluation logic with caching and schema validation."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import diskcache as dc
from pydantic import ValidationError

from prompts import EVALUATION_SYSTEM_PROMPT, build_evaluation_prompt
from schemas import EvaluationResult
from similarity import similarity_breakdown

LOGGER_NAME = "rag_evaluator"
JSON_REMINDER = "\nREMEMBER: RETURN ONLY VALID JSON."


def get_logger(log_path: str = "debug.log", verbose: bool = False) -> logging.Logger:
    """Return a configured logger for evaluator debugging."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    log_file = Path(log_path).resolve()
    file_handler_exists = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_file
        for handler in logger.handlers
    )
    if not file_handler_exists:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if verbose and not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(stream_handler)

    return logger


@dataclass(frozen=True)
class EvaluatorConfig:
    """Runtime configuration for the evaluator backend."""

    provider: str = "ollama"
    model: str | None = None
    cache_dir: str = "eval_cache"
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    ollama_host: str = "http://localhost:11434"
    use_hybrid: bool = True
    num_runs: int = 2
    log_path: str = "debug.log"
    embedding_model: str = "all-MiniLM-L6-v2"
    verbose: bool = False


def clamp_score(value: Any) -> float:
    """Normalize a numeric score into the inclusive [0, 1] interval."""
    return max(0.0, min(1.0, float(value)))


def extract_json(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from a model response."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No valid JSON found")


def get_model_name(config: EvaluatorConfig) -> str:
    """Resolve the model name based on the selected provider."""
    if config.provider == "ollama":
        return config.model or "phi"
    if config.provider == "openai":
        return config.model or "gpt-4o-mini"
    raise ValueError(f"Unsupported provider: {config.provider}")


def call_ollama(prompt: str, model_name: str, temperature: float, host: str) -> str:
    """Call a locally running Ollama instance and return the raw text response."""
    try:
        from ollama import ResponseError, chat
    except ImportError as exc:
        raise RuntimeError(
            "The 'ollama' Python package is not installed. Install it with 'pip install ollama'."
        ) from exc

    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature

    previous_host = os.environ.get("OLLAMA_HOST")
    os.environ["OLLAMA_HOST"] = host
    try:
        response = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options=options or None,
        )
    except ResponseError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    except Exception as exc:
        error_message = str(exc).lower()
        if any(token in error_message for token in ("connect", "connection", "refused", "localhost:11434")):
            raise RuntimeError(
                f"Could not connect to Ollama at {host}. "
                "Start the Ollama server and make sure the 'phi' model is available."
            ) from exc
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    finally:
        if previous_host is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = previous_host

    return response["message"]["content"]


def call_openai(prompt: str, model_name: str, config: EvaluatorConfig) -> str:
    """Call an OpenAI-compatible API backend and return the raw text response."""
    from openai import OpenAI

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {config.api_key_env} must be set for the openai provider."
        )

    client = OpenAI(api_key=api_key, base_url=config.api_base)
    response = client.chat.completions.create(
        model=model_name,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def call_model(prompt: str, config: EvaluatorConfig) -> tuple[str, str]:
    """Dispatch the evaluation prompt to the selected provider."""
    model_name = get_model_name(config)
    if config.provider == "ollama":
        return (
            call_ollama(prompt, model_name, config.temperature, config.ollama_host),
            model_name,
        )
    if config.provider == "openai":
        return call_openai(prompt, model_name, config), model_name
    raise ValueError(f"Unsupported provider: {config.provider}")


def log_raw_output(
    logger: logging.Logger,
    raw_response: str,
    run_index: int,
    failure_reason: str,
) -> None:
    """Persist raw model output when parsing or validation fails."""
    logger.warning(
        "LLM raw output failure on run %s (%s): %s",
        run_index,
        failure_reason,
        raw_response,
    )


def normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a model-produced evaluation payload into the result schema."""
    groundedness = clamp_score(payload.get("groundedness", 0.0))
    normalized_payload = {
        "correctness": clamp_score(payload.get("correctness", 0.0)),
        "relevance": clamp_score(payload.get("relevance", 0.0)),
        "groundedness": groundedness,
        "hallucination": bool(payload.get("hallucination", groundedness < 0.5)),
        "explanation": str(payload.get("explanation", "")),
        "semantic_similarity": clamp_score(payload.get("semantic_similarity", 0.0)),
        "keyword_overlap": clamp_score(payload.get("keyword_overlap", 0.0)),
        "confidence": clamp_score(payload.get("confidence", 0.0)),
    }
    return EvaluationResult.model_validate(normalized_payload).model_dump()


def parse_llm_response(raw_response: str, logger: logging.Logger, run_index: int) -> dict[str, Any]:
    """Parse and validate one raw LLM response."""
    if not raw_response.strip():
        log_raw_output(logger, raw_response, run_index, "empty output")
        raise ValueError("Model output was empty.")

    try:
        return normalize_llm_payload(extract_json(raw_response))
    except Exception as exc:
        log_raw_output(logger, raw_response, run_index, str(exc))
        raise ValueError(str(exc)) from exc


def call_with_retry(
    config: EvaluatorConfig,
    prompt: str,
    logger: logging.Logger,
    run_index: int,
    max_attempts: int = 2,
) -> tuple[dict[str, Any], str]:
    """Call the LLM and retry once when JSON parsing fails."""
    retry_prompt = prompt
    last_error: Exception | None = None
    model_name = get_model_name(config)

    for attempt_number in range(1, max_attempts + 1):
        raw_response, model_name = call_model(retry_prompt, config)
        logger.info(
            "Received raw response for run %s attempt %s",
            run_index,
            attempt_number,
        )
        logger.info("Raw model output: %s", raw_response)

        try:
            parsed_response = parse_llm_response(raw_response, logger, run_index)
            return parsed_response, model_name
        except Exception as exc:
            last_error = exc
            if attempt_number < max_attempts:
                logger.warning(
                    "Retrying run %s after parse failure on attempt %s: %s",
                    run_index,
                    attempt_number,
                    exc,
                )
                retry_prompt += JSON_REMINDER

    raise ValueError(f"Failed to parse JSON after retry: {last_error}")


def build_cache_key(question: str, context: str, answer: str, config: EvaluatorConfig) -> str:
    """Create a stable cache key that includes evaluator configuration."""
    return json.dumps(
        {
            "question": question,
            "context": context,
            "answer": answer,
            "provider": config.provider,
            "model": config.model,
            "api_base": config.api_base,
            "api_key_env": config.api_key_env,
            "temperature": config.temperature,
            "ollama_host": config.ollama_host,
            "use_hybrid": config.use_hybrid,
            "num_runs": config.num_runs,
            "embedding_model": config.embedding_model,
        },
        sort_keys=True,
    )


def fallback_result(
    explanation: str,
    similarity_scores: dict[str, Any] | None = None,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Build a conservative deterministic result when LLM output is unusable."""
    similarity_scores = similarity_scores or {"semantic_similarity": 0.0, "keyword_overlap": 0.0}
    deterministic_correctness = clamp_score(
        0.6 * similarity_scores["semantic_similarity"] + 0.4 * similarity_scores["keyword_overlap"]
    )
    return {
        "correctness": deterministic_correctness,
        "relevance": deterministic_correctness,
        "groundedness": clamp_score(similarity_scores["semantic_similarity"]),
        "hallucination": similarity_scores["semantic_similarity"] < 0.35,
        "explanation": explanation,
        "semantic_similarity": clamp_score(similarity_scores["semantic_similarity"]),
        "keyword_overlap": clamp_score(similarity_scores["keyword_overlap"]),
        "confidence": clamp_score(confidence),
    }


def evaluate_llm_once(
    prompt: str,
    config: EvaluatorConfig,
    logger: logging.Logger,
    run_index: int,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Run one LLM evaluation pass and return a validated payload or an error."""
    try:
        parsed_payload, model_name = call_with_retry(config, prompt, logger, run_index)
        return parsed_payload, model_name, None
    except Exception as exc:
        logger.exception("LLM evaluation run %s failed: %s", run_index, exc)
        return None, None, str(exc)


def aggregate_llm_runs(
    llm_results: list[dict[str, Any]],
    similarity_scores: dict[str, Any],
) -> tuple[dict[str, Any] | None, float]:
    """Average valid LLM runs and compute a consistency agreement score."""
    if not llm_results:
        return None, 0.0

    llm_correctness_values = [item["correctness"] for item in llm_results]
    averaged_llm_result = {
        "correctness": mean(llm_correctness_values),
        "relevance": mean(item["relevance"] for item in llm_results),
        "groundedness": mean(item["groundedness"] for item in llm_results),
        "hallucination": mean(float(item["hallucination"]) for item in llm_results) >= 0.5,
        "explanation": " | ".join(item["explanation"] for item in llm_results if item["explanation"]),
        "semantic_similarity": similarity_scores["semantic_similarity"],
        "keyword_overlap": similarity_scores["keyword_overlap"],
        "confidence": 0.0,
    }

    if len(llm_results) == 1:
        llm_agreement = 0.75
    else:
        llm_correctness_spread = max(llm_correctness_values) - min(llm_correctness_values)
        llm_agreement = clamp_score(1.0 - llm_correctness_spread)

    return averaged_llm_result, llm_agreement


def fuse_scores(
    averaged_llm_result: dict[str, Any] | None,
    similarity_scores: dict[str, Any],
    llm_agreement: float,
    config: EvaluatorConfig,
) -> dict[str, Any]:
    """Combine deterministic signals with optional LLM scores."""
    semantic_score = clamp_score(similarity_scores["semantic_similarity"])
    keyword_score = clamp_score(similarity_scores["keyword_overlap"])
    semantic_question_score = clamp_score(similarity_scores["semantic_answer_question"])
    semantic_context_score = clamp_score(similarity_scores["semantic_answer_context"])
    keyword_question_score = clamp_score(similarity_scores["keyword_answer_question"])
    keyword_context_score = clamp_score(similarity_scores["keyword_answer_context"])
    deterministic_correctness = clamp_score(0.6 * semantic_score + 0.4 * keyword_score)

    if averaged_llm_result and not config.use_hybrid:
        final_groundedness = clamp_score(
            0.8 * averaged_llm_result["groundedness"] + 0.2 * semantic_context_score
        )
        return {
            "correctness": clamp_score(averaged_llm_result["correctness"]),
            "relevance": clamp_score(averaged_llm_result["relevance"]),
            "groundedness": final_groundedness,
            "hallucination": bool(averaged_llm_result["hallucination"] or final_groundedness < 0.35),
            "explanation": averaged_llm_result["explanation"] or "LLM-only evaluation completed.",
            "semantic_similarity": semantic_score,
            "keyword_overlap": keyword_score,
            "confidence": clamp_score(0.7 * llm_agreement + 0.3 * semantic_score),
        }

    if averaged_llm_result and config.use_hybrid:
        llm_correctness = clamp_score(averaged_llm_result["correctness"])
        llm_relevance = clamp_score(averaged_llm_result["relevance"])
        llm_groundedness = clamp_score(averaged_llm_result["groundedness"])
        final_correctness = clamp_score(
            0.5 * llm_correctness + 0.3 * semantic_score + 0.2 * keyword_score
        )
        final_relevance = clamp_score(
            0.5 * llm_relevance + 0.3 * semantic_question_score + 0.2 * keyword_question_score
        )
        final_groundedness = clamp_score(
            0.5 * llm_groundedness + 0.3 * semantic_context_score + 0.2 * keyword_context_score
        )
        confidence = clamp_score(0.6 * llm_agreement + 0.4 * semantic_score)
        explanation = averaged_llm_result["explanation"] or "Hybrid evaluation completed."
        hallucination = bool(
            averaged_llm_result["hallucination"]
            or final_groundedness < 0.35
            or semantic_score < 0.3
        )
        return {
            "correctness": final_correctness,
            "relevance": final_relevance,
            "groundedness": final_groundedness,
            "hallucination": hallucination,
            "explanation": explanation,
            "semantic_similarity": semantic_score,
            "keyword_overlap": keyword_score,
            "confidence": confidence,
        }

    confidence = clamp_score(0.25 + 0.75 * semantic_score)
    return {
        "correctness": deterministic_correctness,
        "relevance": clamp_score(0.7 * semantic_question_score + 0.3 * keyword_question_score),
        "groundedness": clamp_score(0.7 * semantic_context_score + 0.3 * keyword_context_score),
        "hallucination": semantic_score < 0.3 and keyword_score < 0.2,
        "explanation": "LLM evaluation unavailable; used deterministic fallback scoring.",
        "semantic_similarity": semantic_score,
        "keyword_overlap": keyword_score,
        "confidence": confidence,
    }


def evaluate_answer(question: str, context: str, answer: str, config: EvaluatorConfig) -> dict:
    """Evaluate one answer using hybrid scoring and cache only valid results."""
    logger = get_logger(config.log_path, verbose=config.verbose)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    cache = dc.Cache(config.cache_dir)
    cache_key = build_cache_key(question, context, answer, config)

    if cache_key in cache:
        cached_result = cache[cache_key]
        cache.close()
        print("Cache hit for sample evaluation")
        logger.info("Cache hit for question: %s", question)
        return cached_result

    similarity_scores = similarity_breakdown(
        answer=answer,
        question=question,
        context=context,
        cache_dir=os.path.join(config.cache_dir, "embeddings"),
        model_name=config.embedding_model,
    )
    logger.info("Similarity scores: %s", similarity_scores)

    prompt = build_evaluation_prompt(question=question, context=context, answer=answer)
    llm_results: list[dict[str, Any]] = []
    model_name = get_model_name(config)
    llm_errors: list[str] = []

    for run_index in range(1, max(config.num_runs, 1) + 1):
        result, returned_model_name, error_message = evaluate_llm_once(
            prompt,
            config,
            logger,
            run_index,
        )
        if returned_model_name:
            model_name = returned_model_name
        if result is not None:
            llm_results.append(result)
        elif error_message:
            llm_errors.append(error_message)

    averaged_llm_result, llm_agreement = aggregate_llm_runs(llm_results, similarity_scores)
    final_result = fuse_scores(averaged_llm_result, similarity_scores, llm_agreement, config)

    if llm_results:
        print(
            f"Evaluating sample with provider={config.provider}, model={model_name}, "
            f"mode={'hybrid' if config.use_hybrid else 'llm-only'}"
        )
        logger.info("Used LLM-backed evaluation with %s successful runs.", len(llm_results))
    else:
        print("LLM fallback used: deterministic scoring only")
        logger.warning("All LLM runs failed; using deterministic fallback. Errors: %s", llm_errors)
        final_result["explanation"] = (
            f"{final_result['explanation']} "
            f"Errors: {' | '.join(llm_errors) if llm_errors else 'Unknown error.'}"
        )

    try:
        validated_result = normalize_llm_payload(final_result)
    except (ValidationError, TypeError, ValueError) as exc:
        logger.exception("Final schema validation failed: %s", exc)
        cache.close()
        return fallback_result(
            explanation=f"Schema validation failed: {exc}",
            similarity_scores=similarity_scores,
            confidence=final_result.get("confidence", 0.0),
        )

    cache[cache_key] = validated_result
    cache.close()
    return validated_result
