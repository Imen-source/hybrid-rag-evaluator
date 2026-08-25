"""Pure-logic tests for the synthetic trace generator -- no network, no API,
no Docker. Checks build_traces() produces the expected counts and label
invariants."""

from __future__ import annotations

from scripts.generate_synthetic_traces import SEED, build_traces
from scripts.synthetic_facts import FACTS


def test_build_traces_total_count_in_range():
    traces = build_traces()
    assert 50 <= len(traces) <= 100
    assert len(traces) == len(FACTS)


def test_build_traces_bad_count_in_range_and_rest_good():
    traces = build_traces()
    bad = [t for t in traces if t.label == "bad"]
    good = [t for t in traces if t.label == "good"]
    assert 10 <= len(bad) <= 15
    assert len(good) == len(traces) - len(bad)
    assert all(t.label in ("good", "bad") for t in traces)


def test_bad_traces_use_at_least_two_distinct_failure_modes():
    traces = build_traces()
    bad_modes = {t.failure_mode for t in traces if t.label == "bad"}
    assert bad_modes == {"context_swap", "hallucinated_answer"}
    assert all(t.failure_mode is None for t in traces if t.label == "good")


def test_build_traces_is_deterministic_for_a_fixed_seed():
    assert build_traces(seed=SEED) == build_traces(seed=SEED)


def test_context_swap_trace_context_differs_from_original_and_answer_unchanged():
    traces = build_traces()
    facts_by_question = {f["question"]: f for f in FACTS}
    swapped = [t for t in traces if t.failure_mode == "context_swap"]
    assert swapped, "expected at least one context_swap trace"
    for trace in swapped:
        original = facts_by_question[trace.input]
        assert trace.retrieved_context != original["context"]
        assert trace.output == original["answer"]


def test_hallucinated_trace_answer_differs_and_context_unchanged():
    traces = build_traces()
    facts_by_question = {f["question"]: f for f in FACTS}
    hallucinated = [t for t in traces if t.failure_mode == "hallucinated_answer"]
    assert hallucinated, "expected at least one hallucinated_answer trace"
    for trace in hallucinated:
        original = facts_by_question[trace.input]
        assert trace.output != original["answer"]
        assert trace.output == original["bad_answer"]
        assert trace.retrieved_context == original["context"]


def test_good_traces_are_unmodified_facts():
    traces = build_traces()
    facts_by_question = {f["question"]: f for f in FACTS}
    for trace in traces:
        if trace.label == "good":
            original = facts_by_question[trace.input]
            assert trace.output == original["answer"]
            assert trace.retrieved_context == original["context"]
