"""Cache management utilities for local evaluator development."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def clear_cache(cache_dir: str = "eval_cache") -> bool:
    """Remove the evaluator cache directory if it exists."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return False

    shutil.rmtree(cache_path)
    return True


def main() -> None:
    """Provide a small CLI for clearing cached evaluator artifacts."""
    parser = argparse.ArgumentParser(description="Clear evaluator cache directories.")
    parser.add_argument(
        "--cache-dir",
        default="eval_cache",
        help="Cache directory to remove. Defaults to eval_cache.",
    )
    args = parser.parse_args()

    removed = clear_cache(args.cache_dir)
    if removed:
        print(f"Removed cache directory: {Path(args.cache_dir).resolve()}")
    else:
        print(f"Cache directory not found: {Path(args.cache_dir).resolve()}")


if __name__ == "__main__":
    main()
