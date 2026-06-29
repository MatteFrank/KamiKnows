"""Build KamiKnows RAG-ready dataset v0 from Fase 1F full-text outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag_ready.build_dataset import RagReadyDatasetError, build_rag_ready_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG-ready dataset v0 from existing full-text parsing outputs."
    )
    parser.add_argument(
        "--fulltext-dir",
        type=Path,
        required=True,
        help="Fase 1F full-text output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for RAG-ready dataset files.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain label stored in chunk metadata and manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_rag_ready_dataset(
            fulltext_dir=args.fulltext_dir,
            output_dir=args.output_dir,
            domain=args.domain,
        )
    except (OSError, RagReadyDatasetError, ValueError) as exc:
        print(f"KamiKnows RAG-ready dataset build failed: {exc}", file=sys.stderr)
        return 1

    counts = manifest["counts"]
    validation = manifest["validation"]
    print("KamiKnows RAG-ready dataset v0 built")
    print(f"Output directory: {args.output_dir}")
    print(f"Papers: {counts['papers']}")
    print(f"Chunks: {counts['chunks']}")
    print(f"Equations: {counts['equations']}")
    print(f"Eval questions: {counts['eval_questions']}")
    print(f"Validation status: {validation['status']}")
    print(f"Manifest: {args.output_dir / 'rag_manifest_v0.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
