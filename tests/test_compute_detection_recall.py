"""Pure-logic tests for the recall/precision/FPR comparison in
compute_detection_recall.py -- fixed fake labels and judge results, no real
LLM call, no network, no Docker."""

from __future__ import annotations

import pytest

from scripts.compute_detection_recall import compute_detection_metrics, is_detected


def test_is_detected_by_hallucination_flag():
    assert is_detected({"status": "completed", "hallucination": True, "groundedness": 0.9}) is True


def test_is_detected_by_low_groundedness_even_without_flag():
    assert is_detected({"status": "completed", "hallucination": False, "groundedness": 0.3}) is True


def test_not_detected_when_flag_false_and_groundedness_high():
    assert is_detected({"status": "completed", "hallucination": False, "groundedness": 0.9}) is False


def test_not_detected_when_groundedness_missing_and_flag_false():
    assert is_detected({"status": "completed", "hallucination": False, "groundedness": None}) is False


def test_compute_detection_metrics_on_fixed_fake_input():
    labels = {"t1": "bad", "t2": "bad", "t3": "bad", "t4": "good", "t5": "good"}
    results = {
        "t1": {"status": "completed", "hallucination": True, "groundedness": 0.9},  # TP (flag)
        "t2": {"status": "completed", "hallucination": False, "groundedness": 0.9},  # FN
        "t3": {"status": "completed", "hallucination": False, "groundedness": 0.4},  # TP (low groundedness)
        "t4": {"status": "completed", "hallucination": False, "groundedness": 0.9},  # TN
        "t5": {"status": "completed", "hallucination": True, "groundedness": 0.9},  # FP
    }

    metrics = compute_detection_metrics(labels, results)

    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["excluded_trace_ids"] == []
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 2)


def test_compute_detection_metrics_excludes_non_completed_results():
    labels = {"t1": "bad", "t2": "good"}
    results = {
        "t1": {"status": "failed", "hallucination": None, "groundedness": None},
        "t2": {"status": "completed", "hallucination": False, "groundedness": 0.9},
    }

    metrics = compute_detection_metrics(labels, results)

    assert metrics["excluded_trace_ids"] == ["t1"]
    assert metrics["bad_scored"] == 0
    assert metrics["good_scored"] == 1
    assert metrics["recall"] is None  # no bad traces were actually scored
    assert metrics["false_positive_rate"] == pytest.approx(0.0)


def test_compute_detection_metrics_missing_result_is_excluded():
    labels = {"t1": "bad"}
    metrics = compute_detection_metrics(labels, results={})
    assert metrics["excluded_trace_ids"] == ["t1"]
    assert metrics["recall"] is None


def test_compute_detection_metrics_respects_custom_threshold():
    labels = {"t1": "bad"}
    results = {"t1": {"status": "completed", "hallucination": False, "groundedness": 0.45}}

    assert compute_detection_metrics(labels, results, threshold=0.5)["true_positives"] == 1
    assert compute_detection_metrics(labels, results, threshold=0.4)["true_positives"] == 0
