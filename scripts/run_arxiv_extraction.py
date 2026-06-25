"""Run one minimal arXiv -> extraction -> JSONL workflow.

This is the first script that connects the two Fase 0 branches:

arXiv metadata lookup -> title/abstract -> model extraction -> JSONL save

It is intentionally small and processes only one selected arXiv metadata record.
It does not download PDFs, parse LaTeX, create chunks, build embeddings, or run
RAG.

Usage examples from the repository root:

    python scripts/run_arxiv_extraction.py --id 2301.00001

    python scripts/run_arxiv_extraction.py \
      --query "cat:hep-ex AND calorimeter" \
      --backend fake

    python scripts/run_arxiv_extraction.py \
      --metadata-file data/examples/arxiv_metadata_example.json

Use Qwen through local Ollama:

    python scripts/run_arxiv_extraction.py \
      --query "cat:hep-ex AND calorimeter" \
      --backend ollama \
      --model qwen3:0.6b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/run_arxiv_extraction.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.extraction.abstract_to_json import ExtractionError, abstract_to_json
from kamiknows.ingestion.arxiv_metadata import (
    ArxivMetadataError,
    load_arxiv_metadata_file,
    validate_arxiv_metadata,
)
from kamiknows.ingestion.arxiv_downloader import (
    ArxivIngestionError,
    fetch_arxiv_metadata_by_id,
    search_arxiv_metadata,
)
from kamiknows.models.base import ModelPlugin
from kamiknows.models.fake import FakeExtractionModel
from kamiknows.models.ollama_plugin import OllamaPlugin
from kamiknows.run_metadata import DEFAULT_PROMPT_VERSION, build_run_metadata
from kamiknows.storage.jsonl import append_jsonl_record

DEFAULT_ARXIV_EXTRACTIONS_PATH = Path("outputs/arxiv_extractions.jsonl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one minimal arXiv metadata to JSON extraction demo."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--id",
        dest="arxiv_id",
        help="arXiv ID or arXiv abs/pdf URL, for example 2301.00001.",
    )
    source.add_argument(
        "--query",
        help="Small arXiv API query, for example 'cat:hep-ex AND calorimeter'.",
    )
    source.add_argument(
        "--metadata-file",
        type=Path,
        help=(
            "Path to one offline arXiv-style metadata JSON file. "
            "Useful for demos without network access."
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1,
        help="Number of arXiv search results to fetch before selecting the first. Default: 1",
    )
    parser.add_argument(
        "--backend",
        choices=["fake", "ollama"],
        default="fake",
        help="Model backend to use. Default: fake",
    )
    parser.add_argument(
        "--model",
        default="qwen3:0.6b",
        help="Ollama model name used when --backend ollama. Default: qwen3:0.6b",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL used when --backend ollama.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Default: 0.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARXIV_EXTRACTIONS_PATH,
        help="Destination JSONL file. Default: outputs/arxiv_extractions.jsonl",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help=(
            "Prompt/schema version label recorded in JSONL run metadata. "
            f"Default: {DEFAULT_PROMPT_VERSION}"
        ),
    )
    return parser.parse_args(argv)


def load_one_arxiv_record(args: argparse.Namespace) -> dict[str, Any]:
    """Load one metadata record from arXiv, query search, or an offline file."""
    if args.metadata_file:
        return load_arxiv_metadata_file(args.metadata_file)
    if args.arxiv_id:
        return validate_arxiv_metadata(fetch_arxiv_metadata_by_id(args.arxiv_id))

    records = search_arxiv_metadata(args.query, max_results=args.max_results)
    if not records:
        raise ArxivIngestionError("arXiv search returned no records")
    return validate_arxiv_metadata(records[0])


def build_model(args: argparse.Namespace, title: str) -> ModelPlugin:
    """Create the selected model backend."""
    if args.backend == "fake":
        return FakeExtractionModel(title=title)
    if args.backend == "ollama":
        return OllamaPlugin(model=args.model, base_url=args.base_url)
    raise ValueError(f"unsupported backend: {args.backend}")


def build_traceable_record(
    *,
    metadata: dict[str, Any],
    extraction: dict[str, Any],
    backend: str,
    model_name: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    """Wrap extraction with source metadata and run metadata for traceability."""
    return {
        "source": {
            "arxiv_id": metadata.get("arxiv_id", ""),
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        metadata = load_one_arxiv_record(args)
        title = str(metadata["title"])
        abstract = str(metadata["abstract"])
        model = build_model(args, title=title)
        extraction = abstract_to_json(
            model=model,
            title=title,
            abstract=abstract,
            temperature=args.temperature,
        )
        record = build_traceable_record(
            metadata=metadata,
            extraction=extraction,
            backend=args.backend,
            model_name=args.model if args.backend == "ollama" else "fake",
            prompt_version=args.prompt_version,
        )
        saved_path = append_jsonl_record(record, args.output)
    except (
        ArxivIngestionError,
        ArxivMetadataError,
        ExtractionError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"KamiKnows arXiv extraction failed: {exc}", file=sys.stderr)
        if args.backend == "ollama":
            print(
                "Hint: check that Ollama is running and the model is installed, "
                f"for example: ollama pull {args.model}",
                file=sys.stderr,
            )
        return 1

    print(f"Selected arXiv paper: {metadata['arxiv_id']}")
    print(f"Title: {metadata['title']}")
    print(f"Backend: {args.backend}")
    if args.backend == "ollama":
        print(f"Model: {args.model}")
    else:
        print("Note: fake backend ignores the abstract and is only a software demo.")
    print("Traceable JSONL record:")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\nSaved one JSONL record to: {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
