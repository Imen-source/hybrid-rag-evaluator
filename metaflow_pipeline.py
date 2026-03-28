"""Metaflow-compatible RAG evaluation pipeline with a Windows-safe fallback."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dataset_eval import aggregate_metrics, dataset_digest, load_dataset
from evaluator import EvaluatorConfig, evaluate_answer

METAFLOW_IMPORT_ERROR: str | None = None

try:
    from metaflow import FlowSpec, Parameter, current, step

    METAFLOW_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on local Windows Metaflow install
    FlowSpec = object  # type: ignore[assignment]
    Parameter = None  # type: ignore[assignment]
    current = None  # type: ignore[assignment]
    step = None  # type: ignore[assignment]
    METAFLOW_AVAILABLE = False
    METAFLOW_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def parse_bool(value: str | bool) -> bool:
    """Parse a user-friendly boolean argument."""
    if isinstance(value, bool):
        return value

    normalized_value = str(value).strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, received: {value}")


@dataclass
class PipelineArgs:
    """Normalized runtime arguments shared by Metaflow and local fallback."""

    dataset_path: str = "examples/sample.json"
    results_path: str = "results.json"
    cache_dir: str = "eval_cache"
    llm_provider: str = "ollama"
    model: str | None = None
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    use_hybrid: bool = True
    num_runs: int = 2
    verbose: bool = False
    orchestrator: str = "local-fallback"


def ensure_api_key_available(args: PipelineArgs) -> None:
    """Validate API key availability for the OpenAI provider."""
    if args.llm_provider != "openai":
        return

    if not os.getenv(args.api_key_env):
        raise RuntimeError(
            f"API key environment variable '{args.api_key_env}' is not set. "
            "Set it in PowerShell first or pass --api-key to cli.py."
        )


def build_report(
    args: PipelineArgs,
    run_id: str,
    flow_name: str,
    run_started_at: str,
    dataset_hash: str,
    sample_results: list[dict],
    metrics: dict,
) -> dict:
    """Build the final results payload."""
    return {
        "metadata": {
            "flow_name": flow_name,
            "run_id": run_id,
            "dataset_path": args.dataset_path,
            "dataset_hash": dataset_hash,
            "run_started_at": run_started_at,
            "run_completed_at": datetime.now(timezone.utc).isoformat(),
            "results_path": args.results_path,
            "orchestrator": args.orchestrator,
        },
        "config": {
            "dataset_path": args.dataset_path,
            "results_path": args.results_path,
            "cache_dir": args.cache_dir,
            "llm_provider": args.llm_provider,
            "model": args.model,
            "api_base": args.api_base,
            "api_key_env": args.api_key_env,
            "temperature": args.temperature,
            "use_hybrid": args.use_hybrid,
            "num_runs": args.num_runs,
            "verbose": args.verbose,
            "orchestrator": args.orchestrator,
        },
        "metrics": metrics,
        "samples": sample_results,
    }


def save_report(results_path: str, report: dict) -> str:
    """Persist the final report to JSON."""
    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(output_path.resolve())


def run_pipeline(args: PipelineArgs, run_id: str, flow_name: str) -> str:
    """Run the logical pipeline steps sequentially."""
    ensure_api_key_available(args)
    run_started_at = datetime.now(timezone.utc).isoformat()

    print("Step: load_dataset")
    dataset_hash = dataset_digest(args.dataset_path)
    dataset = [sample.model_dump() for sample in load_dataset(args.dataset_path)]
    print(f"Loaded {len(dataset)} samples from {args.dataset_path}")
    print(f"Dataset SHA256: {dataset_hash}")

    print("Step: evaluate_sample")
    evaluator_config = EvaluatorConfig(
        provider=args.llm_provider,
        model=args.model,
        cache_dir=args.cache_dir,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        use_hybrid=args.use_hybrid,
        num_runs=args.num_runs,
        verbose=args.verbose,
    )
    sample_results: list[dict] = []
    for index, sample in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] Evaluating question: {sample['question']}")
        evaluation = evaluate_answer(
            question=sample["question"],
            context=sample["context"],
            answer=sample["answer"],
            config=evaluator_config,
        )
        sample_results.append(
            {
                "question": sample["question"],
                "context": sample["context"],
                "answer": sample["answer"],
                "evaluation": evaluation,
            }
        )
        print(json.dumps(evaluation, indent=2))

    print("Step: aggregate_metrics")
    metrics = aggregate_metrics([item["evaluation"] for item in sample_results])
    print(json.dumps(metrics, indent=2))

    print("Step: save_results")
    report = build_report(
        args=args,
        run_id=run_id,
        flow_name=flow_name,
        run_started_at=run_started_at,
        dataset_hash=dataset_hash,
        sample_results=sample_results,
        metrics=metrics,
    )
    output_path = save_report(args.results_path, report)
    print(f"Saved final report to {output_path}")
    return output_path


def parse_fallback_args() -> PipelineArgs:
    """Parse CLI arguments for the Windows-safe local fallback."""
    parser = argparse.ArgumentParser(
        description="Windows-safe fallback runner for the RAG evaluation pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--dataset-path", default="examples/sample.json")
    run_parser.add_argument("--results-path", default="results.json")
    run_parser.add_argument("--cache-dir", default="eval_cache")
    run_parser.add_argument("--llm-provider", default="ollama", choices=["ollama", "openai"])
    run_parser.add_argument("--model", default="")
    run_parser.add_argument("--api-base", default="")
    run_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--use-hybrid", type=parse_bool, default=True)
    run_parser.add_argument("--num-runs", type=int, default=2)
    run_parser.add_argument("--verbose", type=parse_bool, default=False)

    parsed = parser.parse_args()
    if parsed.command != "run":
        parser.error("Only the 'run' command is supported in fallback mode.")

    return PipelineArgs(
        dataset_path=parsed.dataset_path,
        results_path=parsed.results_path,
        cache_dir=parsed.cache_dir,
        llm_provider=parsed.llm_provider,
        model=parsed.model or None,
        api_base=parsed.api_base or None,
        api_key_env=parsed.api_key_env,
        temperature=parsed.temperature,
        use_hybrid=parsed.use_hybrid,
        num_runs=parsed.num_runs,
        verbose=parsed.verbose,
        orchestrator="local-fallback",
    )


if METAFLOW_AVAILABLE:

    class RAGEvaluationFlow(FlowSpec):
        """Metaflow-managed pipeline that mirrors the local fallback steps."""

        dataset_path = Parameter("dataset-path", default="examples/sample.json")
        results_path = Parameter("results-path", default="results.json")
        cache_dir = Parameter("cache-dir", default="eval_cache")
        llm_provider = Parameter("llm-provider", default="ollama")
        model = Parameter("model", default="")
        api_base = Parameter("api-base", default="")
        api_key_env = Parameter("api-key-env", default="OPENAI_API_KEY")
        temperature = Parameter("temperature", type=float, default=0.0)
        use_hybrid = Parameter("use-hybrid", default=True, type=bool)
        num_runs = Parameter("num-runs", default=2, type=int)
        verbose = Parameter("verbose", default=False, type=bool)

        @step
        def start(self):
            """Initialize run configuration and validate environment."""
            self.pipeline_args = PipelineArgs(
                dataset_path=self.dataset_path,
                results_path=self.results_path,
                cache_dir=self.cache_dir,
                llm_provider=self.llm_provider,
                model=self.model or None,
                api_base=self.api_base or None,
                api_key_env=self.api_key_env,
                temperature=self.temperature,
                use_hybrid=self.use_hybrid,
                num_runs=self.num_runs,
                verbose=self.verbose,
                orchestrator="metaflow",
            )
            ensure_api_key_available(self.pipeline_args)
            self.run_started_at = datetime.now(timezone.utc).isoformat()
            print("Metaflow is available. Running the managed flow.")
            print(json.dumps(asdict(self.pipeline_args), indent=2))
            self.next(self.load_dataset)

        @step
        def load_dataset(self):
            """Load and validate the dataset."""
            self.dataset_hash = dataset_digest(self.pipeline_args.dataset_path)
            self.dataset = [
                sample.model_dump() for sample in load_dataset(self.pipeline_args.dataset_path)
            ]
            print("Step: load_dataset")
            print(f"Loaded {len(self.dataset)} samples from {self.pipeline_args.dataset_path}")
            print(f"Dataset SHA256: {self.dataset_hash}")
            self.next(self.evaluate_sample, foreach="dataset")

        @step
        def evaluate_sample(self):
            """Evaluate one sample."""
            evaluator_config = EvaluatorConfig(
                provider=self.pipeline_args.llm_provider,
                model=self.pipeline_args.model,
                cache_dir=self.pipeline_args.cache_dir,
                api_base=self.pipeline_args.api_base,
                api_key_env=self.pipeline_args.api_key_env,
                temperature=self.pipeline_args.temperature,
                use_hybrid=self.pipeline_args.use_hybrid,
                num_runs=self.pipeline_args.num_runs,
                verbose=self.pipeline_args.verbose,
            )
            sample = self.input
            print("Step: evaluate_sample")
            print(f"Evaluating question: {sample['question']}")
            self.sample_result = {
                "question": sample["question"],
                "context": sample["context"],
                "answer": sample["answer"],
                "evaluation": evaluate_answer(
                    question=sample["question"],
                    context=sample["context"],
                    answer=sample["answer"],
                    config=evaluator_config,
                ),
            }
            self.next(self.join_evaluations)

        @step
        def join_evaluations(self, inputs):
            """Collect sample-level outputs."""
            first = inputs[0]
            self.pipeline_args = first.pipeline_args
            self.dataset_hash = first.dataset_hash
            self.run_started_at = first.run_started_at
            self.sample_results = [branch.sample_result for branch in inputs]
            self.next(self.aggregate_metrics)

        @step
        def aggregate_metrics(self):
            """Aggregate final metrics."""
            print("Step: aggregate_metrics")
            self.metrics = aggregate_metrics([item["evaluation"] for item in self.sample_results])
            print(json.dumps(self.metrics, indent=2))
            self.next(self.save_results)

        @step
        def save_results(self):
            """Persist the final JSON report."""
            print("Step: save_results")
            report = build_report(
                args=self.pipeline_args,
                run_id=current.run_id,
                flow_name=current.flow_name,
                run_started_at=self.run_started_at,
                dataset_hash=self.dataset_hash,
                sample_results=self.sample_results,
                metrics=self.metrics,
            )
            self.output_path = save_report(self.pipeline_args.results_path, report)
            print(f"Saved final report to {self.output_path}")
            self.next(self.end)

        @step
        def end(self):
            """Complete the flow."""
            print("Pipeline finished successfully.")
            print(f"Results file: {self.output_path}")


if __name__ == "__main__":
    if METAFLOW_AVAILABLE:
        RAGEvaluationFlow()
    else:
        fallback_args = parse_fallback_args()
        print("Metaflow could not be imported cleanly in this environment.")
        print(f"Reason: {METAFLOW_IMPORT_ERROR}")
        print("Falling back to the local Windows-safe runner.")
        run_pipeline(
            args=fallback_args,
            run_id=f"local-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            flow_name="RAGEvaluationFlowLocal",
        )
