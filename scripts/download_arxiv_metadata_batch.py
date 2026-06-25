"""Download or collect arXiv metadata and save it as JSON.

This script performs only the ingestion phase:

    arXiv query/IDs -> metadata records -> validated JSON list

It does not call Ollama, Qwen, or any other LLM. The saved JSON file can be used
later by extraction or benchmark scripts.

Examples from the repository root:

    python scripts/download_arxiv_metadata_batch.py \
      --query "cat:hep-ex AND calorimeter" \
      --max-results 3 \
      --output outputs/metadata/calorimeter_metadata.json

    python scripts/download_arxiv_metadata_batch.py \
      --ids 2301.00001 2301.00002 \
      --output outputs/metadata/selected_ids_metadata.json

    python scripts/download_arxiv_metadata_batch.py \
      --ids-file data/examples/arxiv_ids_example.txt \
      --output outputs/metadata/selected_ids_metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/download_arxiv_metadata_batch.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.ingestion.arxiv_downloader import (
    ArxivIngestionError,
    fetch_arxiv_metadata_by_id,
    search_arxiv_metadata,
)
from kamiknows.ingestion.arxiv_metadata import (
    ArxivMetadataError,
    load_arxiv_id_list_file,
    validate_arxiv_metadata,
    write_arxiv_metadata_list_file,
)

DEFAULT_OUTPUT_PATH = Path("outputs/metadata/arxiv_metadata_download.json")
DEFAULT_MAX_RESULTS = 3
DEFAULT_TIMEOUT_SECONDS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download arXiv metadata only and save a validated JSON list."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--query",
        help="Remote arXiv API query, for example 'cat:hep-ex AND calorimeter'.",
    )
    source.add_argument(
        "--ids",
        nargs="+",
        help="One or more arXiv IDs or abs/pdf URLs.",
    )
    source.add_argument(
        "--ids-file",
        type=Path,
        help=(
            "Plain-text file with one arXiv ID or arXiv abs/pdf URL per line. "
            "Blank lines and lines starting with # are ignored."
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"Maximum results when --query is used. Default: {DEFAULT_MAX_RESULTS}",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout for arXiv calls. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination metadata JSON file. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args(argv)


def _validate_positive_int(value: int, *, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def download_metadata_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    """Download metadata records from a query or explicit IDs."""
    _validate_positive_int(args.timeout_seconds, name="--timeout-seconds")

    if args.query:
        _validate_positive_int(args.max_results, name="--max-results")
        records = search_arxiv_metadata(
            args.query,
            max_results=args.max_results,
            timeout_seconds=args.timeout_seconds,
        )
        if not records:
            raise ArxivIngestionError("arXiv search returned no records")
        return [validate_arxiv_metadata(record) for record in records], args.query

    input_ids = args.ids if args.ids is not None else load_arxiv_id_list_file(args.ids_file)

    records: list[dict[str, Any]] = []
    for arxiv_id in input_ids:
        record = fetch_arxiv_metadata_by_id(
            arxiv_id,
            timeout_seconds=args.timeout_seconds,
        )
        records.append(validate_arxiv_metadata(record))
    source_label = (
        f"{len(input_ids)} explicit arXiv ID(s)"
        if args.ids is not None
        else f"arXiv ID list file: {args.ids_file}"
    )
    return records, source_label


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        records, source_label = download_metadata_records(args)
        output_path = write_arxiv_metadata_list_file(records, args.output)
    except (ArxivIngestionError, ArxivMetadataError, RuntimeError, ValueError) as exc:
        print(f"KamiKnows metadata download failed: {exc}", file=sys.stderr)
        return 1

    print("Metadata download completed")
    print(f"Source: {source_label}")
    print(f"Records: {len(records)}")
    print(f"Output file: {output_path}")
    print("")
    print("First record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))
    print("")
    print("Use later with:")
    print(
        "  python scripts/run_batch_arxiv_extraction.py "
        f"--metadata-list {output_path} --backend ollama --model qwen3:0.6b"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
