"""Step 3 - retrieve: rank the chunks of a dataset for every question of an eval set.

Example:
    python scripts/retrieve.py --dataset-dir outputs/chunks_fastcalo_semantic_fd100_nobm \\
        --questions data/questions/fastcalo_paired.jsonl \\
        --output-dir outputs/retrieval_semantic_fd100_nobm_dense_explicit_paired_top5 \\
        --retriever dense --source-policy explicit_arxiv --top-k 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.embeddings import EMBEDDING_MODELS
from kamiknows.retrieval import run_retrieval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="output of chunk.py")
    parser.add_argument("--questions", type=Path, required=True, help="JSONL with question_id and question")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--retriever", choices=["tfidf", "dense"], required=True,
                        help="tfidf: lexical cosine; dense: sentence-transformers embeddings")
    parser.add_argument("--embedding-model", choices=sorted(EMBEDDING_MODELS), default="all-MiniLM-L6-v2",
                        help="used by --retriever dense (default: all-MiniLM-L6-v2)")
    parser.add_argument("--device", default="cpu", help="torch device for the encoder (default: cpu)")
    parser.add_argument("--top-k", type=int, default=5, help="passages per question (default: 5)")
    parser.add_argument("--source-policy", choices=["none", "explicit_arxiv"], default="explicit_arxiv",
                        help="explicit_arxiv: restrict to the arXiv ids written in the question (default)")
    args = parser.parse_args(argv)
    try:
        summary = run_retrieval(dataset_dir=args.dataset_dir, questions_file=args.questions,
                                output_dir=args.output_dir, retriever=args.retriever,
                                embedding_model=args.embedding_model, device=args.device,
                                top_k=args.top_k, source_policy=args.source_policy)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"retrieve failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
