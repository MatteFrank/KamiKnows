"""Build sentence groups from existing full-text sections, without retrieval."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.rag_ready.sentence_dataset import build_sentence_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-words", type=int, default=400)
    parser.add_argument("--overlap-words", type=int, default=60)
    parser.add_argument("--exclude-backmatter", action="store_true",
                        help="Experimental: exclude acknowledgments and bibliography before chunking.")
    args = parser.parse_args()
    try:
        result = build_sentence_dataset(**vars(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Sentence dataset build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Papers: {result['papers_count']}; chunks: {result['chunks_v1_count']}")
    print(f"Lengths (whitespace words): {result['lengths']}")
    print(f"Validation: {result['validation']}")
    print(f"Manifest: {args.output_dir / 'rag_manifest_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
