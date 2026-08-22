"""Judge run configuration, read from the environment.

Kept dependency-light (stdlib only) on purpose: this module is imported by
the FastAPI process too (to record judge_provider/judge_model on the eval_run
row), and the API image intentionally does not install the heavy judge
dependencies (ollama/openai/sentence-transformers) that only the worker
needs.
"""

from __future__ import annotations

import os


def _default_model(provider: str) -> str:
    # Mirrors evaluator.get_model_name()'s two default cases. Duplicated
    # (rather than imported from evaluator.py) so this module stays free of
    # evaluator.py's heavy, worker-only dependencies.
    if provider == "ollama":
        return "phi"
    if provider == "openai":
        return "gpt-4o-mini"
    raise ValueError(f"Unsupported provider: {provider}")


def get_judge_settings() -> dict:
    provider = os.environ.get("JUDGE_PROVIDER", "openai")
    model = os.environ.get("JUDGE_MODEL") or _default_model(provider)
    use_hybrid = os.environ.get("JUDGE_USE_HYBRID", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "provider": provider,
        "model": model,
        "num_runs": int(os.environ.get("JUDGE_NUM_RUNS", "1")),
        "use_hybrid": use_hybrid,
        "worker_count": int(os.environ.get("EVAL_WORKER_COUNT", "1")),
        "cache_dir": os.environ.get("JUDGE_CACHE_DIR", "eval_cache"),
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434"),
        "api_key_env": os.environ.get("JUDGE_API_KEY_ENV", "OPENAI_API_KEY"),
        "temperature": float(os.environ.get("JUDGE_TEMPERATURE", "0.0")),
    }
