"""Inspect a small KamiKnows JSONL output file.

This script is intentionally read-only. It helps during Fase 0 tutorials by
printing a compact human-readable summary of records produced by scripts such as:

    python scripts/run_fake_extraction.py
    python scripts/run_arxiv_extraction.py
    python scripts/smoke_test_offline.py

Usage from the repository root:

    python scripts/inspect_jsonl.py outputs/arxiv_extractions.jsonl

Optional limit:

    python scripts/inspect_jsonl.py outputs/arxiv_extractions.jsonl --limit 5

Markdown export:

    python scripts/inspect_jsonl.py outputs/arxiv_extractions.jsonl \
      --mode markdown --output outputs/arxiv_extractions_readable.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/inspect_jsonl.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.storage.jsonl import JsonlStorageError, read_jsonl_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a small KamiKnows JSONL file and print a readable summary."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a KamiKnows .jsonl file, for example outputs/arxiv_extractions.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of records to display in inspect mode. Default: 10.",
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "markdown"],
        default="inspect",
        help="Output mode. Default: inspect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Markdown output path when --mode markdown. Default: "
            "<input_stem>_readable.md next to the input file."
        ),
    )
    return parser.parse_args(argv)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return value if it is a dictionary; otherwise return an empty dictionary."""
    if isinstance(value, dict):
        return value
    return {}


def _shorten(text: Any, max_chars: int = 90) -> str:
    """Convert a value to one compact single-line string."""
    value = str(text or "-").replace("\n", " ").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def summarize_record(record: dict[str, Any], index: int) -> str:
    """Create a compact, human-readable summary for one KamiKnows JSONL record."""
    source = _as_dict(record.get("source"))
    extraction = _as_dict(record.get("extraction"))
    run = _as_dict(record.get("run"))

    title = source.get("title") or extraction.get("title") or "-"
    arxiv_id = source.get("arxiv_id", "-")
    confidence = extraction.get("confidence", "-")
    backend = run.get("backend", "-")
    model = run.get("model", "-")
    created_at = run.get("created_at", "-")
    main_claim = extraction.get("main_claim", "-")

    lines = [
        f"[{index}] {title}",
        f"    source: {arxiv_id}",
        f"    confidence: {confidence}",
        f"    backend/model: {backend} / {model}",
        f"    created_at: {created_at}",
        f"    main_claim: {main_claim}",
    ]
    return "\n".join(lines)


def inspect_jsonl(path: Path, limit: int = 10) -> str:
    """Return a readable summary for a small KamiKnows JSONL file."""
    if limit < 1:
        raise ValueError("--limit must be >= 1")

    records = read_jsonl_records(path)
    shown_records = records[:limit]

    lines = [
        f"JSONL file: {path}",
        f"Records: {len(records)}",
        f"Showing: {len(shown_records)}",
    ]

    if not records:
        lines.append("No records found.")
        return "\n".join(lines)

    lines.append("")
    for index, record in enumerate(shown_records, start=1):
        lines.append(summarize_record(record, index=index))
        lines.append("")

    if len(records) > limit:
        lines.append(f"... {len(records) - limit} more record(s) not shown. Use --limit to display more.")

    return "\n".join(lines).rstrip()


def default_markdown_output_path(path: Path) -> Path:
    """Return the default readable Markdown path for a JSONL input file."""
    return path.with_name(f"{path.stem}_readable.md")


def _metadata_line(label: str, value: Any) -> str:
    text = str(value or "-").replace("\n", " ").strip() or "-"
    return f"- {label}: `{text}`"


def format_records_as_markdown(records: list[dict[str, Any]], *, input_path: Path) -> str:
    """Format traceable JSONL records as a readable Markdown document."""
    lines = [
        f"# Readable export for {input_path.name}",
        "",
        f"Source JSONL: `{input_path}`",
        f"Records: {len(records)}",
        "",
    ]

    if not records:
        lines.append("No records found.")
        return "\n".join(lines).rstrip() + "\n"

    for index, record in enumerate(records, start=1):
        source = _as_dict(record.get("source"))
        extraction = _as_dict(record.get("extraction"))
        run = _as_dict(record.get("run"))

        title = source.get("title") or extraction.get("title") or "-"
        extraction_json = json.dumps(
            extraction,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

        lines.extend(
            [
                f"## Record {index}",
                "",
                _metadata_line("arXiv ID", source.get("arxiv_id")),
                _metadata_line("Title", title),
                _metadata_line("Backend", run.get("backend")),
                _metadata_line("Model", run.get("model")),
                _metadata_line("Prompt version", run.get("prompt_version")),
                _metadata_line("Schema version", run.get("extraction_schema_version")),
                "",
                "### Extraction",
                "",
                "```json",
                extraction_json,
                "```",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def export_markdown(path: Path, output_path: Path | None = None) -> Path:
    """Write a readable Markdown export for all records in a JSONL file."""
    records = read_jsonl_records(path)
    destination = output_path or default_markdown_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        format_records_as_markdown(records, input_path=path),
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.mode == "inspect":
            if args.output is not None:
                raise ValueError("--output is only supported with --mode markdown")
            summary = inspect_jsonl(path=args.path, limit=args.limit)
            print(summary)
        else:
            output_path = export_markdown(path=args.path, output_path=args.output)
            print(f"Markdown export written to: {output_path}")
    except (JsonlStorageError, OSError, ValueError) as exc:
        print(f"KamiKnows JSONL inspection failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
