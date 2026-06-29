"""Run KamiKnows mini-RAG v0 over a RAG-ready dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag.load_dataset import RagDatasetError
from kamiknows.rag.outputs import run_mini_rag


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mini-RAG v0 retrieval plus optional grounded generation."
    )
    parser.add_argument("--rag-ready-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["ollama"], default="ollama")
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_mini_rag(
            rag_ready_dir=args.rag_ready_dir,
            output_dir=args.output_dir,
            backend=args.backend,
            model=args.model,
            top_k=args.top_k,
            retrieval_only=args.retrieval_only,
            base_url=args.base_url,
            temperature=args.temperature,
        )
    except (OSError, RagDatasetError, ValueError) as exc:
        print(f"KamiKnows mini-RAG failed: {exc}", file=sys.stderr)
        return 1

    summary = result["eval_summary"]
    generation = summary["generation"]
    retrieval = summary["retrieval"]
    print("KamiKnows mini-RAG v0 completed")
    print(f"Output directory: {args.output_dir}")
    print(f"Questions: {summary['questions_total']}")
    print(f"Retrieved contexts: {len(result['retrieved_contexts'])}")
    print(f"Answers: {len(result['answers'])}")
    print(f"Retrieval expected-source hit rate: {retrieval['expected_source_hit_rate']:.3f}")
    print(f"Retrieval-only: {generation['retrieval_only']}")
    print(f"Answers succeeded: {generation['answers_succeeded']}")
    print(f"Backend errors: {generation['backend_errors']}")
    print(f"Manifest: {args.output_dir / 'rag_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
