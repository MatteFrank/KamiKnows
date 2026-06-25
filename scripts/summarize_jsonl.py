"""Summarize and lightly evaluate KamiKnows JSONL extraction records.

This script is intentionally small and offline. It reads a JSONL file produced by
Fase 0 scripts and reports:

    - number of records
    - confidence label counts
    - backend/model/prompt_version usage
    - missing required fields

It does not judge scientific correctness. It only checks record shape and basic
traceability fields.

Usage from the repository root:

    python scripts/summarize_jsonl.py

Custom file:

    python scripts/summarize_jsonl.py outputs/batch_arxiv_extractions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/summarize_jsonl.py without
# requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.storage.jsonl import JsonlStorageError, read_jsonl_records

DEFAULT_INPUT_PATH = Path("outputs/batch_arxiv_extractions.jsonl")
VALID_CONFIDENCE_LABELS = {"low", "medium", "high"}

REQUIRED_SOURCE_FIELDS = (
    "source.arxiv_id",
    "source.title",
    "source.authors",
    "source.categories",
    "source.published",
    "source.url",
)

REQUIRED_EXTRACTION_FIELDS = (
    "extraction.title",
    "extraction.field",
    "extraction.main_claim",
    "extraction.method",
    "extraction.limitations",
    "extraction.confidence",
)

REQUIRED_RUN_FIELDS = (
    "run.run_id",
    "run.created_at",
    "run.backend",
    "run.model",
    "run.prompt_version",
    "run.prompt_template_sha256",
    "run.extraction_schema_version",
)


class JsonlSummaryError(RuntimeError):
    """Raised when JSONL summary cannot be computed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a KamiKnows JSONL output file offline."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=f"JSONL file to summarize. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--allow-simple-records",
        action="store_true",
        help=(
            "Do not require the source.* block. Use this for JSONL files produced "
            "by scripts/run_fake_extraction.py."
        ),
    )
    parser.add_argument(
        "--show-missing-limit",
        type=int,
        default=10,
        help="Maximum number of record indexes to print for each missing field.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return exit code 1 if missing fields or invalid confidence labels are found.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help=(
            "Optional path where the summary is also written as machine-readable JSON. "
            "Parent directories are created automatically."
        ),
    )
    return parser.parse_args(argv)


def _lookup_path(record: dict[str, Any], dotted_path: str) -> Any:
    current: Any = record
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list) and not value:
        return True
    return False


def summarize_records(
    records: list[dict[str, Any]],
    *,
    require_source: bool = True,
) -> dict[str, Any]:
    """Compute a small offline summary for KamiKnows JSONL records.

    Args:
        records: Parsed JSONL records.
        require_source: Whether arXiv-style source.* fields are required.

    Returns:
        A dictionary with counts and warning details.
    """
    if not records:
        raise JsonlSummaryError("JSONL file contains no records")

    required_fields = list(REQUIRED_EXTRACTION_FIELDS) + list(REQUIRED_RUN_FIELDS)
    if require_source:
        required_fields = list(REQUIRED_SOURCE_FIELDS) + required_fields

    confidence_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    prompt_version_counts: Counter[str] = Counter()
    prompt_template_sha256_counts: Counter[str] = Counter()
    extraction_schema_version_counts: Counter[str] = Counter()
    missing_fields: dict[str, list[int]] = defaultdict(list)
    invalid_confidence_records: list[int] = []
    records_with_source = 0

    for index, record in enumerate(records, start=1):
        if isinstance(record.get("source"), dict):
            records_with_source += 1

        for field_path in required_fields:
            value = _lookup_path(record, field_path)
            if _is_missing(value):
                missing_fields[field_path].append(index)

        confidence = _lookup_path(record, "extraction.confidence")
        if _is_missing(confidence):
            confidence_counts["<missing>"] += 1
        else:
            confidence_text = str(confidence)
            confidence_counts[confidence_text] += 1
            if confidence_text not in VALID_CONFIDENCE_LABELS:
                invalid_confidence_records.append(index)

        backend = _lookup_path(record, "run.backend")
        model = _lookup_path(record, "run.model")
        prompt_version = _lookup_path(record, "run.prompt_version")
        prompt_template_sha256 = _lookup_path(record, "run.prompt_template_sha256")
        extraction_schema_version = _lookup_path(record, "run.extraction_schema_version")

        backend_counts[str(backend) if not _is_missing(backend) else "<missing>"] += 1
        model_counts[str(model) if not _is_missing(model) else "<missing>"] += 1
        prompt_version_counts[
            str(prompt_version) if not _is_missing(prompt_version) else "<missing>"
        ] += 1
        prompt_template_sha256_counts[
            str(prompt_template_sha256)
            if not _is_missing(prompt_template_sha256)
            else "<missing>"
        ] += 1
        extraction_schema_version_counts[
            str(extraction_schema_version)
            if not _is_missing(extraction_schema_version)
            else "<missing>"
        ] += 1

    has_warnings = bool(missing_fields or invalid_confidence_records)

    return {
        "total_records": len(records),
        "records_with_source": records_with_source,
        "require_source": require_source,
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "prompt_version_counts": dict(sorted(prompt_version_counts.items())),
        "prompt_template_sha256_counts": dict(
            sorted(prompt_template_sha256_counts.items())
        ),
        "extraction_schema_version_counts": dict(
            sorted(extraction_schema_version_counts.items())
        ),
        "missing_fields": dict(sorted(missing_fields.items())),
        "invalid_confidence_records": invalid_confidence_records,
        "evaluation_status": "WARN" if has_warnings else "PASS",
    }


def _format_counter(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"{title}:"]
    if not counts:
        lines.append("  <none>")
        return lines
    for label, count in counts.items():
        lines.append(f"  {label}: {count}")
    return lines


def format_summary(
    summary: dict[str, Any],
    *,
    input_path: Path,
    show_missing_limit: int = 10,
) -> str:
    """Format a summary dictionary for CLI output."""
    lines: list[str] = [
        f"JSONL file: {input_path}",
        f"Records: {summary['total_records']}",
        f"Records with source block: {summary['records_with_source']}",
        f"Source required: {str(summary['require_source']).lower()}",
        "",
    ]

    lines.extend(_format_counter("Confidence labels", summary["confidence_counts"]))
    lines.append("")
    lines.extend(_format_counter("Backends", summary["backend_counts"]))
    lines.append("")
    lines.extend(_format_counter("Models", summary["model_counts"]))
    lines.append("")
    lines.extend(_format_counter("Prompt versions", summary["prompt_version_counts"]))
    lines.append("")
    lines.extend(
        _format_counter(
            "Prompt template SHA-256", summary["prompt_template_sha256_counts"]
        )
    )
    lines.append("")
    lines.extend(
        _format_counter(
            "Extraction schema versions",
            summary["extraction_schema_version_counts"],
        )
    )
    lines.append("")

    missing_fields: dict[str, list[int]] = summary["missing_fields"]
    invalid_confidence_records: list[int] = summary["invalid_confidence_records"]

    if missing_fields:
        lines.append("Missing fields:")
        for field_path, indexes in missing_fields.items():
            shown = indexes[:show_missing_limit]
            suffix = "" if len(indexes) <= show_missing_limit else " ..."
            shown_text = ", ".join(str(item) for item in shown)
            lines.append(f"  {field_path}: {len(indexes)} record(s) [{shown_text}{suffix}]")
    else:
        lines.append("Missing fields: none")

    if invalid_confidence_records:
        shown = invalid_confidence_records[:show_missing_limit]
        suffix = "" if len(invalid_confidence_records) <= show_missing_limit else " ..."
        shown_text = ", ".join(str(item) for item in shown)
        lines.append(
            "Invalid confidence labels: "
            f"{len(invalid_confidence_records)} record(s) [{shown_text}{suffix}]"
        )
    else:
        lines.append("Invalid confidence labels: none")

    lines.append(f"Evaluation status: {summary['evaluation_status']}")
    return "\n".join(lines)


def write_summary_json(summary: dict[str, Any], output_path: Path) -> Path:
    """Write a machine-readable summary JSON file.

    Args:
        summary: Summary dictionary returned by summarize_records.
        output_path: Destination JSON path.

    Returns:
        The written path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        records = read_jsonl_records(args.input_path)
        summary = summarize_records(
            records,
            require_source=not args.allow_simple_records,
        )
    except (JsonlStorageError, JsonlSummaryError) as exc:
        print(f"KamiKnows JSONL summary failed: {exc}", file=sys.stderr)
        return 1

    print(
        format_summary(
            summary,
            input_path=args.input_path,
            show_missing_limit=args.show_missing_limit,
        )
    )

    if args.json_output is not None:
        write_summary_json(summary, args.json_output)
        print(f"Summary JSON written to: {args.json_output}")

    has_warnings = summary["evaluation_status"] == "WARN"
    if args.fail_on_warnings and has_warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
