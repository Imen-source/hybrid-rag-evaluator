"""Compute the judge's bad-trace detection recall/precision/false-positive
rate against the ground-truth manifest written by generate_synthetic_traces.py.

DETECTION RULE (decided before running the judge; not adjusted afterward):

    A trace is "detected as bad" if EITHER
        - result.hallucination is True, OR
        - result.groundedness < GROUNDEDNESS_THRESHOLD (0.5)

    Why: evaluator.py's own fuse_scores() already sets hallucination=True
    whenever groundedness < 0.35 (see evaluator.py), so re-checking
    groundedness at that same 0.35 cutoff would just restate the flag under
    a different name. Using a looser, independent threshold of 0.5 gives a
    second, genuinely additional signal -- it catches traces where
    groundedness degraded meaningfully without crossing the model's own
    internal hallucination cutoff -- while still being a deliberately
    conservative bar (below "somewhat grounded") rather than a coin flip
    at 0.5 chosen to flatter recall.

Both signals come directly off the persisted EvalResult row; nothing here
is re-derived from the raw LLM output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

GROUNDEDNESS_THRESHOLD = 0.5

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "scripts" / "synthetic_manifest.json"
DEFAULT_RESULTS_PATH = REPO_ROOT / "scripts" / "detection_recall_results.json"


def is_detected(result: dict[str, Any], threshold: float = GROUNDEDNESS_THRESHOLD) -> bool:
    """Apply the detection rule to one EvalResult-shaped dict."""
    if result.get("hallucination") is True:
        return True
    groundedness = result.get("groundedness")
    return groundedness is not None and groundedness < threshold


def compute_detection_metrics(
    labels: dict[str, str],
    results: dict[str, dict[str, Any]],
    threshold: float = GROUNDEDNESS_THRESHOLD,
) -> dict[str, Any]:
    """Compare ground-truth labels ("good"/"bad") against judge results.

    `results` maps trace_id -> {"status", "hallucination", "groundedness", ...}.
    Only results with status == "completed" are scored; others are counted
    separately as excluded (the judge produced no usable signal for them).
    """
    true_positives = false_negatives = false_positives = true_negatives = 0
    excluded: list[str] = []

    for trace_id, label in labels.items():
        result = results.get(trace_id)
        if result is None or result.get("status") != "completed":
            excluded.append(trace_id)
            continue

        detected = is_detected(result, threshold=threshold)
        if label == "bad":
            if detected:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if detected:
                false_positives += 1
            else:
                true_negatives += 1

    bad_scored = true_positives + false_negatives
    good_scored = false_positives + true_negatives
    flagged = true_positives + false_positives

    return {
        "threshold": threshold,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "excluded_trace_ids": excluded,
        "bad_scored": bad_scored,
        "good_scored": good_scored,
        "recall": (true_positives / bad_scored) if bad_scored else None,
        "precision": (true_positives / flagged) if flagged else None,
        "false_positive_rate": (false_positives / good_scored) if good_scored else None,
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def trigger_eval_run(trace_ids: list[str], api_url: str, client: httpx.Client) -> str:
    response = client.post(f"{api_url}/eval/run", json={"trace_ids": trace_ids})
    response.raise_for_status()
    return response.json()["run_id"]


def poll_eval_run(
    run_id: str, api_url: str, client: httpx.Client, poll_interval: float, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        response = client.get(f"{api_url}/eval/runs/{run_id}")
        response.raise_for_status()
        run = response.json()
        if run["status"] in ("completed", "failed"):
            return run
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Eval run {run_id} did not finish within {timeout}s (last status: {run['status']})"
            )
        time.sleep(poll_interval)


def print_report(manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    print("\n=== Judge bad-trace detection report ===")
    print(f"Total traces in manifest : {manifest['total']}")
    print(f"Labeled bad              : {manifest['bad_count']}")
    print(f"Labeled good              : {manifest['good_count']}")
    print(f"Detection rule            : hallucination == True OR groundedness < {metrics['threshold']}")
    print(f"Excluded (no usable result): {len(metrics['excluded_trace_ids'])}")
    print(
        f"Confusion matrix          : TP={metrics['true_positives']} FN={metrics['false_negatives']} "
        f"FP={metrics['false_positives']} TN={metrics['true_negatives']}"
    )
    print(f"Recall (of bad traces)    : {fmt(metrics['recall'])}")
    print(f"Precision (of flagged)    : {fmt(metrics['precision'])}")
    print(f"False positive rate       : {fmt(metrics['false_positive_rate'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--threshold", type=float, default=GROUNDEDNESS_THRESHOLD)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_path)
    labels = {e["trace_id"]: e["label"] for e in manifest["traces"]}
    trace_ids = list(labels.keys())

    with httpx.Client(timeout=30.0) as client:
        run_id = trigger_eval_run(trace_ids, args.api_url, client)
        print(f"Triggered eval run {run_id} over {len(trace_ids)} traces. Polling...")
        run = poll_eval_run(run_id, args.api_url, client, args.poll_interval, args.timeout)

    results = {r["trace_id"]: r for r in run["results"]}
    metrics = compute_detection_metrics(labels, results, threshold=args.threshold)
    print_report(manifest, metrics)

    per_trace = []
    for entry in manifest["traces"]:
        result = results.get(entry["trace_id"], {})
        per_trace.append(
            {
                "trace_id": entry["trace_id"],
                "label": entry["label"],
                "failure_mode": entry["failure_mode"],
                "domain": entry["domain"],
                "judge_status": result.get("status"),
                "hallucination": result.get("hallucination"),
                "groundedness": result.get("groundedness"),
                "correctness": result.get("correctness"),
                "detected": is_detected(result, threshold=args.threshold) if result else None,
            }
        )

    output = {
        "run_id": run_id,
        "run_status": run["status"],
        "judge_provider": run["judge_provider"],
        "judge_model": run["judge_model"],
        "metrics": metrics,
        "per_trace": per_trace,
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nFull results written to {args.results_path}")


if __name__ == "__main__":
    main()
