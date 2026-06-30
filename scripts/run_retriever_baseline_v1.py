"""Run KamiKnows retrieval baseline v1 over a RAG-ready dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag.load_dataset import RagDatasetError
from kamiknows.rag.retriever_baseline_v1 import (
    default_v0_summary_path,
    run_retriever_baseline_v1,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic retrieval baseline v1 and diagnostics."
    )
    parser.add_argument("--rag-ready-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument(
        "--rag-v0-summary",
        type=Path,
        default=None,
        help="Optional path to rag_eval_summary.json for v0 comparison.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rag_v0_summary = args.rag_v0_summary
    if rag_v0_summary is None:
        candidate = default_v0_summary_path(args.rag_ready_dir)
        rag_v0_summary = candidate if candidate.exists() else None

    try:
        result = run_retriever_baseline_v1(
            rag_ready_dir=args.rag_ready_dir,
            output_dir=args.output_dir,
            top_k_values=args.top_k_values,
            rag_v0_summary_path=rag_v0_summary,
        )
    except (OSError, RagDatasetError, ValueError) as exc:
        print(f"KamiKnows retriever baseline v1 failed: {exc}", file=sys.stderr)
        return 1

    summary = result["eval_summary"]
    print("KamiKnows retriever baseline v1 completed")
    print(f"Output directory: {args.output_dir}")
    print(f"Questions: {summary['questions_total']}")
    print(f"Top-k values: {', '.join(str(value) for value in summary['top_k_values'])}")
    for top_k, hit_rate in summary["hit_rate_by_top_k"].items():
        print(f"Expected-source hit rate @ {top_k}: {hit_rate:.3f}")
    print(f"Manifest: {args.output_dir / 'retrieval_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
