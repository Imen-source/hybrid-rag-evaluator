import pytest

from dataset_eval import aggregate_metrics, compute_metric_deltas, evaluate_regression


def test_aggregate_metrics_computes_expected_means():
    results = [
        {
            "correctness": 1.0,
            "relevance": 1.0,
            "groundedness": 1.0,
            "hallucination": False,
            "explanation": "Fully correct",
            "semantic_similarity": 1.0,
            "keyword_overlap": 1.0,
            "confidence": 0.95,
        },
        {
            "correctness": 0.0,
            "relevance": 0.5,
            "groundedness": 0.0,
            "hallucination": True,
            "explanation": "Unsupported answer",
            "semantic_similarity": 0.2,
            "keyword_overlap": 0.1,
            "confidence": 0.4,
        },
    ]

    aggregated = aggregate_metrics(results)

    assert aggregated["avg_correctness"] == 0.5
    assert aggregated["avg_relevance"] == 0.75
    assert aggregated["avg_groundedness"] == 0.5
    assert aggregated["avg_semantic_similarity"] == 0.6
    assert aggregated["avg_keyword_overlap"] == pytest.approx(0.55)
    assert aggregated["avg_confidence"] == pytest.approx(0.675)
    assert aggregated["hallucination_rate"] == 0.5
    assert aggregated["sample_count"] == 2


def _metrics(correctness: float, relevance: float, groundedness: float, hallucination_rate: float) -> dict:
    """Minimal aggregate_metrics()-shaped dict -- compute_metric_deltas only
    reads the four COMPARABLE_METRICS keys, so tests don't need the other
    aggregate_metrics fields (avg_semantic_similarity etc.)."""
    return {
        "avg_correctness": correctness,
        "avg_relevance": relevance,
        "avg_groundedness": groundedness,
        "hallucination_rate": hallucination_rate,
    }


def test_compute_metric_deltas_reports_baseline_candidate_delta_and_relative_delta():
    baseline = _metrics(0.90, 0.85, 0.90, 0.05)
    candidate = _metrics(0.80, 0.85, 0.90, 0.05)

    deltas = compute_metric_deltas(candidate, baseline)

    correctness = deltas["avg_correctness"]
    assert correctness["baseline"] == 0.90
    assert correctness["candidate"] == 0.80
    assert correctness["delta"] == pytest.approx(-0.10)
    assert correctness["relative_delta"] == pytest.approx(-0.10 / 0.90)

    assert deltas["avg_groundedness"]["delta"] == pytest.approx(0.0)
    assert deltas["hallucination_rate"]["delta"] == pytest.approx(0.0)


def test_compute_metric_deltas_relative_delta_is_none_when_baseline_is_zero():
    baseline = _metrics(0.0, 0.5, 0.5, 0.0)
    candidate = _metrics(0.5, 0.5, 0.5, 0.0)

    deltas = compute_metric_deltas(candidate, baseline)

    assert deltas["avg_correctness"]["delta"] == pytest.approx(0.5)
    assert deltas["avg_correctness"]["relative_delta"] is None


def test_evaluate_regression_flags_correctness_drop_past_threshold():
    baseline = _metrics(0.90, 0.85, 0.90, 0.05)
    candidate = _metrics(0.80, 0.85, 0.90, 0.05)  # -11.1% relative, past the 5% bar

    verdict = evaluate_regression(compute_metric_deltas(candidate, baseline))

    assert verdict["regressed"] is True
    assert len(verdict["reasons"]) == 1
    assert "avg_correctness" in verdict["reasons"][0]


def test_evaluate_regression_flags_groundedness_drop_past_threshold():
    baseline = _metrics(0.90, 0.85, 0.90, 0.05)
    candidate = _metrics(0.90, 0.85, 0.80, 0.05)  # -11.1% relative, past the 5% bar

    verdict = evaluate_regression(compute_metric_deltas(candidate, baseline))

    assert verdict["regressed"] is True
    assert len(verdict["reasons"]) == 1
    assert "avg_groundedness" in verdict["reasons"][0]


def test_evaluate_regression_flags_hallucination_rate_increase_past_threshold():
    baseline = _metrics(0.90, 0.85, 0.90, 0.05)
    candidate = _metrics(0.90, 0.85, 0.90, 0.15)  # +10 points, past the 5-point bar

    verdict = evaluate_regression(compute_metric_deltas(candidate, baseline))

    assert verdict["regressed"] is True
    assert len(verdict["reasons"]) == 1
    assert "hallucination_rate" in verdict["reasons"][0]


def test_evaluate_regression_no_regression_within_thresholds():
    baseline = _metrics(0.90, 0.85, 0.90, 0.05)
    # correctness/groundedness each drop ~2.2% (under 5%), hallucination_rate
    # rises 3 points (under 5) -- all inside the documented tolerance.
    candidate = _metrics(0.88, 0.85, 0.88, 0.08)

    verdict = evaluate_regression(compute_metric_deltas(candidate, baseline))

    assert verdict["regressed"] is False
    assert verdict["reasons"] == []


def test_evaluate_regression_does_not_gate_on_relevance():
    baseline = _metrics(0.90, 0.90, 0.90, 0.05)
    # Relevance craters; correctness/groundedness/hallucination are untouched.
    candidate = _metrics(0.90, 0.40, 0.90, 0.05)

    verdict = evaluate_regression(compute_metric_deltas(candidate, baseline))

    assert verdict["regressed"] is False
    assert verdict["reasons"] == []


def test_evaluate_regression_reports_every_metric_that_regresses():
    baseline = _metrics(0.90, 0.85, 0.90, 0.05)
    candidate = _metrics(0.80, 0.85, 0.80, 0.15)  # correctness, groundedness, hallucination all trip

    verdict = evaluate_regression(compute_metric_deltas(candidate, baseline))

    assert verdict["regressed"] is True
    assert len(verdict["reasons"]) == 3
    assert verdict["thresholds"] == {
        "correctness_relative_drop": 0.05,
        "groundedness_relative_drop": 0.05,
        "hallucination_rate_absolute_increase": 0.05,
    }
