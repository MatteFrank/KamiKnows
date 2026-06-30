"""Run KamiKnows Phase 2 embedding model selection v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag.embedding_selection import (
    default_tfidf_summary_path,
    run_embedding_model_selection_v1,
)
from kamiknows.rag.load_dataset import RagDatasetError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local/open embedding retrieval candidates on a RAG-ready dataset."
    )
    parser.add_argument("--rag-ready-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys or full names from the embedding registry. Defaults to all registry models.",
    )
    parser.add_argument(
        "--encoder-backend",
        choices=["auto", "hash", "sentence-transformers"],
        default="auto",
        help="auto uses local sentence-transformers if available, otherwise deterministic hash fallback.",
    )
    parser.add_argument(
        "--tfidf-summary",
        type=Path,
        default=None,
        help="Optional TF-IDF summary JSON for comparison.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tfidf_summary = args.tfidf_summary
    if tfidf_summary is None:
        candidate = default_tfidf_summary_path(args.rag_ready_dir)
        tfidf_summary = candidate if candidate.exists() else None

    try:
        result = run_embedding_model_selection_v1(
            rag_ready_dir=args.rag_ready_dir,
            output_dir=args.output_dir,
            model_names_or_keys=args.models,
            top_k=args.top_k,
            encoder_backend=args.encoder_backend,
            tfidf_summary_path=tfidf_summary,
        )
    except (OSError, RagDatasetError, RuntimeError, ValueError) as exc:
        print(f"KamiKnows embedding model selection v1 failed: {exc}", file=sys.stderr)
        return 1

    summary = result["eval_summary"]
    selected = summary["selected_provisional_embedding_model"]
    print("KamiKnows embedding model selection v1 completed")
    print(f"Output directory: {args.output_dir}")
    print(f"Questions: {summary['dataset_counts']['eval_questions']}")
    print(f"Chunks: {summary['dataset_counts']['chunks']}")
    print(f"Models compared: {summary['models_total']}")
    print(f"Top-k: {summary['top_k']}")
    print(f"Selected provisional model: {selected['model_name']} ({selected['encoder_backend']})")
    print(f"Selected hit@k: {selected['expected_source_hit_rate_at_k']:.3f}")
    print(f"Summary: {args.output_dir / 'embedding_eval_summary_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
