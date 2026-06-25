"""Build a manual quality checklist for KamiKnows extraction records.

This module intentionally does not judge scientific correctness automatically.
It prepares a compact Markdown review sheet so a human can check whether the
extracted ``main_claim``, ``method`` and ``limitations`` are faithful to the
paper abstract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from kamiknows.storage.jsonl import read_jsonl_records

DEFAULT_REVIEW_LIMIT = 3
DEFAULT_MAX_ABSTRACT_CHARS = 1800


class ManualChecklistError(RuntimeError):
    """Raised when a manual checklist cannot be built."""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _single_line(value: Any) -> str:
    return str(value or "-").replace("\n", " ").strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars < 100:
        raise ManualChecklistError("max abstract chars must be >= 100")
    normalized = str(text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 24].rstrip() + "\n\n[abstract truncated]"


def build_abstract_index(metadata_records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Build ``arxiv_id -> abstract`` lookup from metadata records."""
    index: dict[str, str] = {}
    for record in metadata_records:
        if not isinstance(record, dict):
            continue
        arxiv_id = str(record.get("arxiv_id", "")).strip()
        abstract = str(record.get("abstract", "")).strip()
        if arxiv_id and abstract:
            index[arxiv_id] = abstract
    return index


def _find_abstract(record: dict[str, Any], abstract_index: dict[str, str]) -> str:
    source = _as_dict(record.get("source"))
    inline_abstract = str(source.get("abstract", "")).strip()
    if inline_abstract:
        return inline_abstract

    arxiv_id = str(source.get("arxiv_id", "")).strip()
    if arxiv_id and arxiv_id in abstract_index:
        return abstract_index[arxiv_id]

    return ""


def build_record_review_section(
    *,
    record: dict[str, Any],
    index: int,
    abstract_index: dict[str, str] | None = None,
    max_abstract_chars: int = DEFAULT_MAX_ABSTRACT_CHARS,
) -> str:
    """Build one Markdown review section."""
    if not isinstance(record, dict):
        raise ManualChecklistError("record must be a dictionary")

    abstract_index = abstract_index or {}
    source = _as_dict(record.get("source"))
    extraction = _as_dict(record.get("extraction"))
    run = _as_dict(record.get("run"))

    title = _single_line(source.get("title") or extraction.get("title"))
    arxiv_id = _single_line(source.get("arxiv_id"))
    url = _single_line(source.get("url"))
    backend = _single_line(run.get("backend"))
    model = _single_line(run.get("model"))
    prompt_version = _single_line(run.get("prompt_version"))
    confidence = _single_line(extraction.get("confidence"))
    abstract = _find_abstract(record, abstract_index)

    if abstract:
        abstract_block = _truncate(abstract, max_chars=max_abstract_chars)
    else:
        abstract_block = (
            "[abstract missing from this JSONL record. Regenerate with the current "
            "scripts or pass --metadata-list to create_manual_quality_checklist.py]"
        )

    lines = [
        f"## Record {index}: {title}",
        "",
        f"- arXiv ID: `{arxiv_id}`",
        f"- URL: {url}",
        f"- backend/model: `{backend}` / `{model}`",
        f"- prompt version: `{prompt_version}`",
        f"- extraction confidence: `{confidence}`",
        "",
        "### Abstract used for review",
        "",
        abstract_block,
        "",
        "### Extracted fields to check",
        "",
        f"- `main_claim`: {_single_line(extraction.get('main_claim'))}",
        f"- `method`: {_single_line(extraction.get('method'))}",
        f"- `limitations`: {_single_line(extraction.get('limitations'))}",
        "",
        "### Manual checklist",
        "",
        "- [ ] `main_claim` is directly supported by the abstract.",
        "- [ ] `method` is directly supported by the abstract.",
        "- [ ] `limitations` are directly supported by the abstract.",
        "- [ ] No unsupported scientific claim was introduced.",
        "- [ ] The confidence label is plausible for this extraction.",
        "- [ ] Notes:",
        "",
        "Review outcome: `pass | revise | reject | unclear`",
        "",
    ]
    return "\n".join(lines)


def build_manual_quality_checklist(
    *,
    records: list[dict[str, Any]],
    source_path: str | Path,
    limit: int = DEFAULT_REVIEW_LIMIT,
    abstract_index: dict[str, str] | None = None,
    max_abstract_chars: int = DEFAULT_MAX_ABSTRACT_CHARS,
) -> str:
    """Build a Markdown checklist for a small sample of extraction records."""
    if limit < 1:
        raise ManualChecklistError("limit must be >= 1")
    if not records:
        raise ManualChecklistError("no records available for manual review")

    abstract_index = abstract_index or {}
    selected = records[:limit]

    lines = [
        "# KamiKnows manual quality checklist",
        "",
        "Purpose: manually check whether extracted fields are faithful to the abstract.",
        "This is not an automatic scientific evaluation and not a benchmark score.",
        "",
        f"Source JSONL: `{source_path}`",
        f"Records selected: {len(selected)} / {len(records)}",
        "",
        "Review focus:",
        "",
        "- `main_claim`",
        "- `method`",
        "- `limitations`",
        "- unsupported claims or overstatements",
        "",
    ]

    for index, record in enumerate(selected, start=1):
        lines.append(
            build_record_review_section(
                record=record,
                index=index,
                abstract_index=abstract_index,
                max_abstract_chars=max_abstract_chars,
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def build_manual_quality_checklist_from_jsonl(
    *,
    jsonl_path: str | Path,
    limit: int = DEFAULT_REVIEW_LIMIT,
    abstract_index: dict[str, str] | None = None,
    max_abstract_chars: int = DEFAULT_MAX_ABSTRACT_CHARS,
) -> str:
    """Read a JSONL file and build a Markdown manual quality checklist."""
    records = read_jsonl_records(jsonl_path)
    return build_manual_quality_checklist(
        records=records,
        source_path=jsonl_path,
        limit=limit,
        abstract_index=abstract_index,
        max_abstract_chars=max_abstract_chars,
    )
