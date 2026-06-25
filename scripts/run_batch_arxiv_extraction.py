"""Run a tiny batch arXiv-style extraction demo.

This script processes multiple arXiv-style metadata records and writes one
traceable JSONL record per paper:

    metadata records -> title/abstract -> ModelPlugin -> extraction JSON -> JSONL

Supported metadata sources in Fase 0:

- local JSON list, deterministic and offline
- remote arXiv search query
- remote list of arXiv IDs

The default backend is still ``fake`` so the batch path is stable and runnable
without Ollama. Use ``--backend ollama`` only when you explicitly want to call a
local Ollama model.

Examples from the repository root:

    python scripts/run_batch_arxiv_extraction.py

    python scripts/run_batch_arxiv_extraction.py \
      --metadata-list data/examples/arxiv_metadata_batch_example.json \
      --output outputs/batch_arxiv_extractions.jsonl

    python scripts/run_batch_arxiv_extraction.py \
      --query "cat:hep-ex AND calorimeter" \
      --max-results 3

    python scripts/run_batch_arxiv_extraction.py \
      --ids 2301.00001 2301.00002

    python scripts/run_batch_arxiv_extraction.py \
      --ids-file data/examples/arxiv_ids_example.txt

    python scripts/run_batch_arxiv_extraction.py \
      --query "cat:hep-ex AND calorimeter" \
      --max-results 2 \
      --backend ollama \
      --model qwen3:0.6b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/run_batch_arxiv_extraction.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.extraction.abstract_to_json import ExtractionError, abstract_to_json
from kamiknows.ingestion.arxiv_downloader import (
    ArxivIngestionError,
    fetch_arxiv_metadata_by_id,
    search_arxiv_metadata,
)
from kamiknows.ingestion.arxiv_metadata import (
    ArxivMetadataError,
    load_arxiv_id_list_file,
    load_arxiv_metadata_list_file,
    validate_arxiv_metadata,
)
from kamiknows.models.base import ModelPlugin
from kamiknows.models.fake import FakeExtractionModel
from kamiknows.models.ollama_plugin import OllamaPlugin
from kamiknows.run_metadata import DEFAULT_PROMPT_VERSION, build_run_metadata
from kamiknows.storage.jsonl import JsonlStorageError, append_jsonl_record

DEFAULT_BATCH_METADATA_PATH = Path("data/examples/arxiv_metadata_batch_example.json")
DEFAULT_BATCH_OUTPUT_PATH = Path("outputs/batch_arxiv_extractions.jsonl")
DEFAULT_OLLAMA_MODEL = "qwen3:0.6b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_NETWORK_TIMEOUT_SECONDS = 30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny batch extraction demo over local or remote arXiv-style "
            "metadata records."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--metadata-list",
        type=Path,
        default=None,
        help=(
            "Path to a JSON list of arXiv-style metadata records. "
            f"Default when no remote source is given: {DEFAULT_BATCH_METADATA_PATH}"
        ),
    )
    source.add_argument(
        "--query",
        help=(
            "Remote arXiv API query, for example "
            "'cat:hep-ex AND calorimeter'. Processes up to --max-results records."
        ),
    )
    source.add_argument(
        "--ids",
        nargs="+",
        help=(
            "One or more remote arXiv IDs or abs/pdf URLs, for example "
            "2301.00001 arXiv:2301.00002v1."
        ),
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
        default=3,
        help="Maximum remote arXiv search results to process when --query is used. Default: 3",
    )
    parser.add_argument(
        "--backend",
        choices=["fake", "ollama"],
        default="fake",
        help="Model backend to use. Default: fake",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help=(
            "Ollama model name used when --backend ollama. "
            f"Default: {DEFAULT_OLLAMA_MODEL}"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama base URL used when --backend ollama.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Default: 0.0",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_NETWORK_TIMEOUT_SECONDS,
        help=(
            "Network timeout for remote arXiv calls. "
            f"Default: {DEFAULT_NETWORK_TIMEOUT_SECONDS}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BATCH_OUTPUT_PATH,
        help=f"Destination JSONL file. Default: {DEFAULT_BATCH_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help=(
            "Prompt/schema version label recorded in JSONL run metadata. "
            f"Default: {DEFAULT_PROMPT_VERSION}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of loaded records to process. "
            "For --query, --max-results controls the remote fetch size first."
        ),
    )
    return parser.parse_args(argv)


def _validate_limit(limit: int | None) -> None:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >= 1 when provided")


def _validate_positive_int(value: int, *, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def load_batch_metadata_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    """Load a batch of metadata records from local JSON or remote arXiv.

    Returns ``(records, source_label)`` where source_label is a human-readable
    description used only for CLI output.
    """
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
        validated = [validate_arxiv_metadata(record) for record in records]
        return validated, f"remote arXiv query: {args.query}"

    if args.ids or args.ids_file:
        input_ids = args.ids if args.ids is not None else load_arxiv_id_list_file(args.ids_file)
        records: list[dict[str, Any]] = []
        for arxiv_id in input_ids:
            record = fetch_arxiv_metadata_by_id(
                arxiv_id,
                timeout_seconds=args.timeout_seconds,
            )
            records.append(validate_arxiv_metadata(record))
        source_label = (
            f"remote arXiv IDs: {len(input_ids)}"
            if args.ids is not None
            else f"remote arXiv IDs file: {args.ids_file}"
        )
        return records, source_label

    metadata_list_path = args.metadata_list or DEFAULT_BATCH_METADATA_PATH
    return (
        load_arxiv_metadata_list_file(metadata_list_path),
        f"local metadata list: {metadata_list_path}",
    )


def build_traceable_batch_record(
    *,
    metadata: dict[str, Any],
    extraction: dict[str, Any],
    backend: str = "fake",
    model_name: str = "fake",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    """Wrap one batch extraction with source metadata and run metadata."""
    return {
        "source": {
            "arxiv_id": metadata.get("arxiv_id", ""),
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "abstract": metadata.get("abstract", ""),
            "categories": metadata.get("categories", []),
            "published": metadata.get("published", ""),
            "url": metadata.get("url", ""),
        },
        "extraction": extraction,
        "run": build_run_metadata(
            backend=backend,
            model=model_name,
            prompt_version=prompt_version,
        ),
    }


def build_model_for_record(
    *,
    backend: str,
    title: str,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> ModelPlugin:
    """Create the selected model backend for one record."""
    if backend == "fake":
        return FakeExtractionModel(title=title)
    if backend == "ollama":
        return OllamaPlugin(model=model_name, base_url=base_url)
    raise ValueError(f"unsupported backend: {backend}")


def run_batch(
    *,
    metadata_records: list[dict[str, Any]],
    output_path: Path,
    backend: str = "fake",
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = 0.0,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Process metadata records and append traceable JSONL records."""
    _validate_limit(limit)

    if limit is not None:
        metadata_records = metadata_records[:limit]

    if not metadata_records:
        raise ArxivMetadataError("no metadata records to process")

    saved_records: list[dict[str, Any]] = []
    for metadata in metadata_records:
        validated_metadata = validate_arxiv_metadata(metadata)
        title = str(validated_metadata["title"])
        abstract = str(validated_metadata["abstract"])
        model = build_model_for_record(
            backend=backend,
            title=title,
            model_name=model_name,
            base_url=base_url,
        )
        extraction = abstract_to_json(
            model=model,
            title=title,
            abstract=abstract,
            temperature=temperature,
        )
        record = build_traceable_batch_record(
            metadata=validated_metadata,
            extraction=extraction,
            backend=backend,
            model_name=model_name if backend == "ollama" else "fake",
            prompt_version=prompt_version,
        )
        append_jsonl_record(record, output_path)
        saved_records.append(record)

    return saved_records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        _validate_limit(args.limit)
        metadata_records, source_label = load_batch_metadata_records(args)
        records = run_batch(
            metadata_records=metadata_records,
            output_path=args.output,
            backend=args.backend,
            model_name=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            prompt_version=args.prompt_version,
            limit=args.limit,
        )
    except (
        ArxivIngestionError,
        ArxivMetadataError,
        ExtractionError,
        JsonlStorageError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"KamiKnows batch extraction failed: {exc}", file=sys.stderr)
        if args.backend == "ollama":
            print(
                "Hint: check that Ollama is running and the model is installed, "
                f"for example: ollama pull {args.model}",
                file=sys.stderr,
            )
        return 1

    print(f"Batch source: {source_label}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model if args.backend == 'ollama' else 'fake'}")
    print(f"Processed records: {len(records)}")
    print(f"Output file: {args.output}")
    print("")
    for index, record in enumerate(records, start=1):
        source = record["source"]
        extraction = record["extraction"]
        print(f"[{index}] {source['arxiv_id']} - {source['title']}")
        print(f"    confidence: {extraction['confidence']}")
        print(f"    main_claim: {extraction['main_claim']}")

    print("\nInspect with:")
    print(f"  python scripts/inspect_jsonl.py {args.output}")
    print("\nSummarize with:")
    print(f"  python scripts/summarize_jsonl.py {args.output}")
    print("\nLast saved record:")
    print(json.dumps(records[-1], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
