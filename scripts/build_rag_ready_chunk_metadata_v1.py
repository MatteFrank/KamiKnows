"""Build RAG-ready chunking/metadata v1 outputs from a RAG-ready v0 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag.load_dataset import RagDatasetError
from kamiknows.rag_ready.chunk_metadata_v1 import build_rag_ready_chunk_metadata_v1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chunking and metadata v1 artifacts from RAG-ready v0."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_rag_ready_chunk_metadata_v1(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    except (OSError, RagDatasetError, ValueError) as exc:
        print(f"KamiKnows chunk metadata v1 build failed: {exc}", file=sys.stderr)
        return 1

    audit = result["audit_summary"]
    smoke = result["retrieval_smoke_summary"]
    print("KamiKnows chunk metadata v1 build completed")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Chunks v0: {audit['chunks_count']}")
    print(f"Chunks v1: {len(result['chunks_v1'])}")
    print(f"Chunks with flags: {result['manifest']['quality_summary']['chunks_with_quality_flags']}")
    print(f"Retrieval smoke hit rate: {smoke['expected_source_hit_rate']:.3f}")
    print(f"Manifest: {args.output_dir / 'rag_manifest_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
