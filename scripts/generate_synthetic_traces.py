"""Generate a labeled synthetic trace set and post it to the ingestion API.

WHY / HOW BAD TRACES ARE CHOSEN (fixed before any judge run, not tuned after
seeing results -- see the module docstring in synthetic_facts.py for the
provenance of the `bad_answer` field used below):

  1. `BAD_COUNT` fact indices are chosen out of the 70-fact bank via
     `random.Random(SEED).sample(...)`. The seed is fixed and printed in the
     manifest, so the selection is reproducible.
  2. The lower half (sorted) of those indices become "context_swap" traces:
     `retrieved_context` is replaced with another fact's context, offset by
     exactly half the bank size (35) -- since facts are grouped 5-per-domain,
     a +35 offset always lands in a different domain, guaranteeing a real
     mismatch rather than an accidental near-duplicate context.
  3. The upper half become "hallucinated_answer" traces: `output` is
     replaced with that fact's pre-written `bad_answer` (authored alongside
     the correct answer, before any selection happened), while
     `retrieved_context` is left untouched -- so the failure is purely "the
     answer isn't supported by/contradicts the context", not a context
     problem.
  4. Everything else is a "good" trace: unmodified question/context/answer.

Every resulting trace's ground-truth label is written to a JSON manifest
(`--manifest-path`, checked into the repo) keyed by the trace_id the API
assigns on insert, so `compute_detection_recall.py` can recompute recall
against the same traces later without regenerating them.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from scripts.synthetic_facts import FACTS

SEED = 42
BAD_COUNT = 12  # within the requested 10-15 range
SWAP_OFFSET = len(FACTS) // 2  # 35; crosses domains since domains are grouped 5-per-block

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "scripts" / "synthetic_manifest.json"


@dataclass(frozen=True)
class TraceSpec:
    input: str
    output: str
    retrieved_context: str
    domain: str
    label: str  # "good" | "bad"
    failure_mode: str | None  # None | "context_swap" | "hallucinated_answer"
    reason: str


def select_bad_indices(
    n_facts: int, bad_count: int = BAD_COUNT, seed: int = SEED
) -> tuple[list[int], list[int]]:
    """Return (context_swap_indices, hallucinated_indices), each sorted."""
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(n_facts), bad_count))
    half = bad_count // 2
    return chosen[:half], chosen[half:]


def build_traces(seed: int = SEED) -> list[TraceSpec]:
    """Build the full labeled synthetic trace set (no network calls)."""
    facts = FACTS
    context_swap_idx, hallucinated_idx = select_bad_indices(len(facts), seed=seed)
    context_swap_set = set(context_swap_idx)
    hallucinated_set = set(hallucinated_idx)

    traces: list[TraceSpec] = []
    for i, fact in enumerate(facts):
        if i in context_swap_set:
            swap_source = facts[(i + SWAP_OFFSET) % len(facts)]
            traces.append(
                TraceSpec(
                    input=fact["question"],
                    output=fact["answer"],
                    retrieved_context=swap_source["context"],
                    domain=fact["domain"],
                    label="bad",
                    failure_mode="context_swap",
                    reason=(
                        f"retrieved_context swapped with a '{swap_source['domain']}' fact's "
                        f"context; it does not support the '{fact['domain']}' answer given."
                    ),
                )
            )
        elif i in hallucinated_set:
            traces.append(
                TraceSpec(
                    input=fact["question"],
                    output=fact["bad_answer"],
                    retrieved_context=fact["context"],
                    domain=fact["domain"],
                    label="bad",
                    failure_mode="hallucinated_answer",
                    reason="output replaced with a pre-authored claim that contradicts retrieved_context.",
                )
            )
        else:
            traces.append(
                TraceSpec(
                    input=fact["question"],
                    output=fact["answer"],
                    retrieved_context=fact["context"],
                    domain=fact["domain"],
                    label="good",
                    failure_mode=None,
                    reason="unmodified fact; answer is supported by retrieved_context.",
                )
            )
    return traces


def post_traces(traces: list[TraceSpec], api_url: str, client: httpx.Client) -> list[dict]:
    """POST each trace to /traces and return manifest entries with the assigned trace_id."""
    entries = []
    for spec in traces:
        response = client.post(
            f"{api_url}/traces",
            json={
                "input": spec.input,
                "output": spec.output,
                "retrieved_context": spec.retrieved_context,
                "metadata": {"synthetic": True, "domain": spec.domain},
            },
        )
        response.raise_for_status()
        trace_id = response.json()["id"]
        entries.append(
            {
                "trace_id": trace_id,
                "label": spec.label,
                "failure_mode": spec.failure_mode,
                "domain": spec.domain,
                "reason": spec.reason,
                "input": spec.input,
                "output": spec.output,
                "retrieved_context": spec.retrieved_context,
            }
        )
    return entries


def write_manifest(entries: list[dict], api_url: str, seed: int, manifest_path: Path) -> None:
    bad_count = sum(1 for e in entries if e["label"] == "bad")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "api_url": api_url,
        "seed": seed,
        "total": len(entries),
        "bad_count": bad_count,
        "good_count": len(entries) - bad_count,
        "failure_modes": sorted({e["failure_mode"] for e in entries if e["failure_mode"]}),
        "traces": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    traces = build_traces(seed=args.seed)
    bad = [t for t in traces if t.label == "bad"]
    print(f"Built {len(traces)} synthetic traces ({len(bad)} bad, {len(traces) - len(bad)} good).")

    with httpx.Client(timeout=30.0) as client:
        entries = post_traces(traces, args.api_url, client)

    write_manifest(entries, args.api_url, args.seed, args.manifest_path)
    print(f"Posted {len(entries)} traces to {args.api_url}. Manifest written to {args.manifest_path}")


if __name__ == "__main__":
    main()
