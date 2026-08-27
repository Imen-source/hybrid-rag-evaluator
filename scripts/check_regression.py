"""CI/deployment gate: exit non-zero if a run has regressed vs. the marked baseline.

Calls `GET /eval/runs/{run_id}/compare` on a live API and exits 1 if the
response's `regressed` field is true, or if the compare call itself fails
(e.g. no baseline set yet, or the run isn't `completed`) -- neither of those
is a "safe to act on this run" state either. Exits 0 only when the compare
call succeeds and `regressed` is false.

NOT wired into .github/workflows/ci.yml. Every call needs a real, completed
eval run scored by a live judge against a live API + Postgres -- the same
reason `scripts/compute_detection_recall.py` isn't CI-gated (see
docs/architecture.md's "known limitations"). This script is meant to be run
by hand, or from a deployment pipeline that already has a live stack and a
completed run to check, e.g. after a new prompt/model change has been
eval'd:

    python -m scripts.check_regression --api-url http://localhost:8000 --run-id <uuid>
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def fetch_comparison(run_id: str, api_url: str, client: httpx.Client) -> dict:
    response = client.get(f"{api_url}/eval/runs/{run_id}/compare")
    if response.status_code != 200:
        raise RuntimeError(f"Compare call failed ({response.status_code}): {response.text}")
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--run-id", required=True, help="Eval run id to check against the marked baseline."
    )
    args = parser.parse_args()

    with httpx.Client(timeout=30.0) as client:
        try:
            comparison = fetch_comparison(args.run_id, args.api_url, client)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    print(json.dumps(comparison, indent=2))

    if comparison["regressed"]:
        print("\nREGRESSION DETECTED vs. baseline run " + comparison["baseline_run_id"], file=sys.stderr)
        for reason in comparison["regressed_reasons"]:
            print(f"  - {reason}", file=sys.stderr)
        return 1

    print("\nNo regression vs. baseline run " + comparison["baseline_run_id"] + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
