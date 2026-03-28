"""Convenience CLI for running the evaluation pipeline on any platform."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_bool(value: str | bool) -> bool:
    """Parse a user-friendly boolean CLI flag."""
    if isinstance(value, bool):
        return value

    normalized_value = str(value).strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, received: {value}")


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the pipeline command from CLI arguments."""
    project_root = Path(__file__).resolve().parent
    flow_script = project_root / "metaflow_pipeline.py"
    selected_model = args.model
    if args.provider == "ollama" and not selected_model:
        selected_model = "phi"

    command = [
        sys.executable,
        str(flow_script),
        "run",
        "--dataset-path",
        args.file,
        "--results-path",
        args.results,
        "--cache-dir",
        args.cache_dir,
        "--llm-provider",
        args.provider,
        "--temperature",
        str(args.temperature),
        "--api-key-env",
        args.api_key_env,
        "--use-hybrid",
        str(args.use_hybrid).lower(),
        "--num-runs",
        str(args.num_runs),
        "--verbose",
        str(args.verbose).lower(),
    ]

    if selected_model:
        command.extend(["--model", selected_model])
    if args.api_base:
        command.extend(["--api-base", args.api_base])

    return command


def build_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    """Create an environment for the child process with reliable API key propagation."""
    env = os.environ.copy()
    if args.api_key:
        env[args.api_key_env] = args.api_key
    env.setdefault("PYTHONUTF8", "1")
    return env


def main() -> None:
    """Parse CLI arguments and run the evaluator pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the local hybrid RAG evaluation pipeline with Ollama or OpenAI."
    )
    parser.add_argument(
        "--file",
        default="examples/sample.json",
        help="Path to a JSON dataset containing question/context/answer samples.",
    )
    parser.add_argument(
        "--results",
        default="results.json",
        help="Path where the final evaluation report will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default="eval_cache",
        help="Directory used to store model responses and embedding caches.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend used for evaluation. Defaults to the local Ollama path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use. Defaults to 'phi' for Ollama and 'gpt-4o-mini' for OpenAI.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Optional OpenAI-compatible base URL for remote providers.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name that stores the API key for the openai provider.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key value to inject into the subprocess environment.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used by the judge model.",
    )
    parser.add_argument(
        "--use-hybrid",
        type=parse_bool,
        default=True,
        help="Combine LLM scoring with semantic similarity and keyword overlap. Default: true.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Number of LLM evaluation passes used for consistency scoring. Default: 2.",
    )
    parser.add_argument(
        "--verbose",
        type=parse_bool,
        default=False,
        help="Print debug logging to the console while still writing full logs to debug.log.",
    )

    args = parser.parse_args()
    command = build_command(args)
    env = build_subprocess_env(args)

    print("Running evaluation pipeline:")
    print(" ".join(command))
    if args.provider == "openai":
        key_present = bool(env.get(args.api_key_env))
        print(f"OpenAI key source: env var '{args.api_key_env}' present={key_present}")

    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
