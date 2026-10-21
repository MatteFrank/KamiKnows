"""Step 2 - chunk: split parsed sections into chunks with the chosen chunker.

Examples:
    python scripts/chunk.py word --parsed-dir outputs/parsed_fastcalo \\
        --output-dir outputs/chunks_fastcalo_word

    python scripts/chunk.py semantic --parsed-dir outputs/parsed_fastcalo \\
        --output-dir outputs/chunks_fastcalo_semantic_fd100_nobm \\
        --exclude-backmatter --cut-policy first_drop --min-words 100 --plots
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.chunking import SemanticChunker, make_sentence_chunker, make_word_chunker, run_chunking
from kamiknows.embeddings import EMBEDDING_MODELS


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--parsed-dir", type=Path, required=True, help="output of parse.py")
    common.add_argument("--output-dir", type=Path, required=True, help="new directory for this run")
    common.add_argument("--target-words", type=int, default=400, help="maximum chunk size (default: 400)")
    common.add_argument("--overlap-words", type=int, default=60, help="overlap between chunks (default: 60)")
    common.add_argument("--exclude-backmatter", action="store_true",
                        help="drop acknowledgment and bibliography sections before chunking")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    chunkers = parser.add_subparsers(dest="chunker", required=True)
    chunkers.add_parser("word", parents=[common], help="fixed word windows")
    chunkers.add_parser("sentence", parents=[common], help="whole sentences up to --target-words")
    semantic = chunkers.add_parser("semantic", parents=[common], help="cut where embedding similarity drops")
    semantic.add_argument("--embedding-model", choices=sorted(EMBEDDING_MODELS), default="all-MiniLM-L6-v2",
                          help="sentence encoder (default: all-MiniLM-L6-v2)")
    semantic.add_argument("--device", default="cpu", help="torch device for the encoder (default: cpu)")
    semantic.add_argument("--min-words", type=int, default=250, help="minimum chunk size (default: 250)")
    semantic.add_argument("--window-sentences", type=int, default=2,
                          help="sentences averaged on each side of a boundary (default: 2)")
    semantic.add_argument("--threshold-quantile", type=float, default=0.25,
                          help="a boundary is a candidate below this per-section quantile (default: 0.25)")
    semantic.add_argument("--cut-policy", choices=["lowest_similarity", "first_drop"], default="lowest_similarity",
                          help="cut at the lowest candidate, or at the first candidate with a large drop")
    semantic.add_argument("--min-score-drop", type=float, default=0.10,
                          help="minimum similarity decrease for first_drop (default: 0.10)")
    semantic.add_argument("--plots", action="store_true", help="also write PNG diagnostics in plots/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target_words < 100 or not 0 <= args.overlap_words < args.target_words:
        print("chunk failed: require --target-words >= 100 and 0 <= --overlap-words < --target-words",
              file=sys.stderr)
        return 1
    params = {"target_words": args.target_words, "overlap_words": args.overlap_words}
    try:
        if args.chunker == "word":
            chunker = make_word_chunker(**params)
        elif args.chunker == "sentence":
            chunker = make_sentence_chunker(**params)
        else:
            from kamiknows.embeddings import Encoder
            encoder = Encoder(args.embedding_model, device=args.device)
            semantic = {key: getattr(args, key) for key in
                        ("min_words", "window_sentences", "threshold_quantile", "cut_policy", "min_score_drop")}
            chunker = SemanticChunker(lambda sentences: encoder.encode(sentences, kind="passage"),
                                      **params, **semantic)
            params.update(semantic, embedding_model=args.embedding_model)
        summary = run_chunking(parsed_dir=args.parsed_dir, output_dir=args.output_dir, chunker_name=args.chunker,
                               chunker=chunker, exclude_backmatter=args.exclude_backmatter, params=params,
                               plots=getattr(args, "plots", False))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"chunk failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
