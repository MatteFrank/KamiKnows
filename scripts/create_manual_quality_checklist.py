"""Create a Markdown manual quality checklist from KamiKnows JSONL records.

The script is intentionally not an automatic scientific judge. It prepares a
small review sheet so a human can compare the paper abstract with the extracted
``main_claim``, ``method`` and ``limitations``.

Example from the repository root:

    python scripts/create_manual_quality_checklist.py \
      outputs/model_mini_benchmark/query_a_ollama_qwen3_0_6b.jsonl \
      --output outputs/manual_quality_checklist.md

If the JSONL records do not contain ``source.abstract``, pass the metadata file
used to create them:

    python scripts/create_manual_quality_checklist.py \
      outputs/qwen_calorimeter_extractions.jsonl \
      --metadata-list outputs/metadata/calorimeter_metadata.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the script runnable with: python scripts/create_manual_quality_checklist.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.ingestion.arxiv_metadata import (
    ArxivMetadataError,
    load_arxiv_metadata_list_file,
)
from kamiknows.quality.manual_checklist import (
    DEFAULT_MAX_ABSTRACT_CHARS,
    DEFAULT_REVIEW_LIMIT,
    ManualChecklistError,
    build_abstract_index,
    build_manual_quality_checklist_from_jsonl,
)
from kamiknows.storage.jsonl import JsonlStorageError

DEFAULT_OUTPUT_PATH = Path("outputs/manual_quality_checklist.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Markdown manual review checklist from KamiKnows JSONL records."
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to a KamiKnows extraction JSONL file.",
    )
    parser.add_argument(
        "--metadata-list",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional arXiv metadata JSON list used to recover abstracts when "
            "the JSONL does not include source.abstract. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_REVIEW_LIMIT,
        help=f"Maximum records to include. Default: {DEFAULT_REVIEW_LIMIT}.",
    )
    parser.add_argument(
        "--max-abstract-chars",
        type=int,
        default=DEFAULT_MAX_ABSTRACT_CHARS,
        help=(
            "Maximum characters of abstract to include per record. "
            f"Default: {DEFAULT_MAX_ABSTRACT_CHARS}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination Markdown file. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args(argv)


def load_optional_abstract_index(metadata_paths: list[Path]) -> dict[str, str]:
    records = []
    for metadata_path in metadata_paths:
        records.extend(load_arxiv_metadata_list_file(metadata_path))
    return build_abstract_index(records)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        abstract_index = load_optional_abstract_index(args.metadata_list)
        checklist = build_manual_quality_checklist_from_jsonl(
            jsonl_path=args.jsonl_path,
            limit=args.limit,
            abstract_index=abstract_index,
            max_abstract_chars=args.max_abstract_chars,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(checklist, encoding="utf-8")
    except (
        ArxivMetadataError,
        JsonlStorageError,
        ManualChecklistError,
        OSError,
        ValueError,
    ) as exc:
        print(f"KamiKnows manual checklist creation failed: {exc}", file=sys.stderr)
        return 1

    print("Manual quality checklist created")
    print(f"Source JSONL: {args.jsonl_path}")
    if args.metadata_list:
        print("Metadata files:")
        for metadata_path in args.metadata_list:
            print(f"  {metadata_path}")
    print(f"Output Markdown: {args.output}")
    print("")
    print("Review manually: this file is not an automatic scientific score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
