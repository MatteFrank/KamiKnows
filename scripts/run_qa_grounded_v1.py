"""Run KamiKnows Phase 2 QA grounded v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag.load_dataset import RagDatasetError
from kamiknows.rag.qa_grounded_v1 import DEFAULT_BASE_URL, DEFAULT_TOP_K, run_qa_grounded_v1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run citation-constrained QA grounded v1 over chunks_v1."
    )
    parser.add_argument("--rag-ready-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/task_reports/phase2_task04_qa_grounded_v1.md"),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Ollama model names to attempt. Defaults to qwen3:0.6b qwen3:4b mistral:7b.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_qa_grounded_v1(
            rag_ready_dir=args.rag_ready_dir,
            output_dir=args.output_dir,
            report_path=args.report_path,
            model_names=args.models,
            top_k=args.top_k,
            base_url=args.base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, RagDatasetError, ValueError, requests_error()) as exc:
        print(f"KamiKnows QA grounded v1 failed: {exc}", file=sys.stderr)
        return 1

    summary = result["eval_summary"]
    print("KamiKnows QA grounded v1 completed")
    print(f"Output directory: {args.output_dir}")
    print(f"Report: {args.report_path}")
    print(f"Questions: {summary['questions_total']}")
    print(f"Top-k: {summary['top_k']}")
    print(f"Models attempted: {', '.join(summary['models_attempted'])}")
    print(f"Models available: {', '.join(summary['models_available']) or '<none>'}")
    print(f"Answers succeeded: {summary['answers_succeeded']}")
    print(f"Backend errors: {summary['backend_errors']}")
    print(f"Expected-source hit rate: {summary['expected_source_hit_rate']:.3f}")
    print(f"Invalid citation records: {summary['invalid_citation_count']}")
    return 0


def requests_error() -> type[Exception]:
    try:
        import requests
    except ModuleNotFoundError:
        return RuntimeError
    return requests.RequestException


if __name__ == "__main__":
    raise SystemExit(main())
