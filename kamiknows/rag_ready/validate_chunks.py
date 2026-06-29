"""Validation helpers for RAG-ready chunk records."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from kamiknows.fulltext.chunking import count_words

ALLOWED_CHUNK_SOURCE_TYPES = {"section", "abstract", "equation_context"}
REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "paper_id",
    "arxiv_id",
    "title",
    "source_type",
    "section_id",
    "section_heading",
    "text",
    "word_count",
    "metadata",
)
REQUIRED_METADATA_FIELDS = (
    "domain",
    "source_pilot",
    "paper_dir",
    "source_file",
    "parsing_status",
    "source_type_original",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, dict) and not value:
        return True
    return False


def missing_fields_for_chunk(chunk: dict[str, Any]) -> list[str]:
    """Return missing required top-level or metadata fields for one chunk."""
    missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk or _is_missing(chunk[field])]
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        if "metadata" not in missing:
            missing.append("metadata")
        return missing
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata or _is_missing(metadata[field]):
            missing.append(f"metadata.{field}")
    return missing


def validate_chunks(
    chunks: list[dict[str, Any]],
    *,
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate RAG-ready chunk records and return a compact report."""
    paper_ids = {str(paper.get("paper_id")) for paper in papers if paper.get("paper_id")}
    missing_counter: Counter[str] = Counter()
    source_type_counter: Counter[str] = Counter()
    chunk_id_counter: Counter[str] = Counter()
    warnings: list[str] = []
    empty_chunks = 0
    orphan_chunks = 0
    word_count_mismatch = 0
    word_counts: list[int] = []
    chunks_per_paper: Counter[str] = Counter()

    for chunk in chunks:
        for field in missing_fields_for_chunk(chunk):
            missing_counter[field] += 1

        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id:
            chunk_id_counter[chunk_id] += 1

        paper_id = str(chunk.get("paper_id") or "")
        if paper_id:
            chunks_per_paper[paper_id] += 1
        if paper_id not in paper_ids:
            orphan_chunks += 1

        text = str(chunk.get("text") or "")
        if not text.strip():
            empty_chunks += 1

        source_type = str(chunk.get("source_type") or "<missing>")
        source_type_counter[source_type] += 1
        if source_type not in ALLOWED_CHUNK_SOURCE_TYPES:
            warnings.append(f"invalid source_type {source_type!r} in chunk {chunk_id or '<missing>'}")

        word_count = chunk.get("word_count")
        actual_word_count = count_words(text)
        if isinstance(word_count, int):
            word_counts.append(word_count)
            tolerance = max(3, int(actual_word_count * 0.2))
            if abs(word_count - actual_word_count) > tolerance:
                word_count_mismatch += 1
        else:
            missing_counter["word_count"] += 1

    duplicate_chunk_ids = sorted(
        chunk_id for chunk_id, count in chunk_id_counter.items() if count > 1
    )
    papers_without_chunks = sorted(
        str(paper.get("paper_id"))
        for paper in papers
        if paper.get("paper_id") and chunks_per_paper.get(str(paper.get("paper_id")), 0) == 0
    )

    missing_required_fields = dict(sorted(missing_counter.items()))
    if word_count_mismatch:
        warnings.append(f"{word_count_mismatch} chunk(s) have word_count outside tolerance")

    status = "PASS"
    if (
        orphan_chunks
        or empty_chunks
        or duplicate_chunk_ids
        or any(missing_required_fields.values())
    ):
        status = "FAIL"
    elif papers_without_chunks or warnings:
        status = "REVISE"

    if word_counts:
        word_stats = {
            "min": min(word_counts),
            "median": median(word_counts),
            "max": max(word_counts),
        }
    else:
        word_stats = {"min": 0, "median": 0, "max": 0}

    return {
        "status": status,
        "orphan_chunks": orphan_chunks,
        "empty_chunks": empty_chunks,
        "missing_required_fields": missing_required_fields,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "papers_without_chunks": papers_without_chunks,
        "warnings": warnings,
        "chunks_per_paper": dict(sorted(chunks_per_paper.items())),
        "source_type_counts": dict(sorted(source_type_counter.items())),
        "word_count_stats": word_stats,
    }


def build_chunk_quality_report(
    *,
    papers: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Build the machine-readable chunk quality report."""
    parsing_status_counts = Counter(str(paper.get("parsing_status") or "<missing>") for paper in papers)
    recommendation = "Dataset passes validation; inspect chunks manually before Fase 1H mini-RAG."
    if validation["status"] == "FAIL":
        recommendation = "Fix failed chunk validation issues before any retrieval experiment."
    elif validation["status"] == "REVISE":
        recommendation = "Review warnings or papers without chunks before moving to Fase 1H."

    return {
        "report_version": "chunk_quality_report_v0",
        "scope": "RAG-ready dataset validation only; this is not RAG yet.",
        "papers": len(papers),
        "chunks": len(chunks),
        "chunks_per_paper": validation["chunks_per_paper"],
        "word_count": validation["word_count_stats"],
        "empty_chunk_count": validation["empty_chunks"],
        "duplicate_chunk_ids": validation["duplicate_chunk_ids"],
        "missing_field_counts": validation["missing_required_fields"],
        "orphan_chunk_count": validation["orphan_chunks"],
        "papers_without_chunks": validation["papers_without_chunks"],
        "source_type_counts": validation["source_type_counts"],
        "parsing_status_counts": dict(sorted(parsing_status_counts.items())),
        "validation_status": validation["status"],
        "warnings": validation["warnings"],
        "recommendation": recommendation,
    }


def format_chunk_quality_report_markdown(report: dict[str, Any]) -> str:
    """Format a readable chunk quality report."""
    lines = [
        "# KamiKnows RAG-Ready Dataset v0 Chunk Quality Report",
        "",
        "This is not RAG yet. No embeddings, vector DB, retrieval, LLM calls, or generated answers are included.",
        "",
        "## Summary",
        "",
        f"- Papers: {report['papers']}",
        f"- Chunks: {report['chunks']}",
        f"- Validation status: {report['validation_status']}",
        f"- Empty chunks: {report['empty_chunk_count']}",
        f"- Orphan chunks: {report['orphan_chunk_count']}",
        f"- Duplicate chunk IDs: {len(report['duplicate_chunk_ids'])}",
        f"- Word count min/median/max: {report['word_count']['min']} / {report['word_count']['median']} / {report['word_count']['max']}",
        "",
        "## Per-Paper Chunks",
        "",
        "| Paper ID | Chunks |",
        "|---|---:|",
    ]
    for paper_id, count in report["chunks_per_paper"].items():
        lines.append(f"| `{paper_id}` | {count} |")

    lines.extend(["", "## Source Types", ""])
    for source_type, count in report["source_type_counts"].items():
        lines.append(f"- `{source_type}`: {count}")

    lines.extend(["", "## Parsing Status Counts", ""])
    for status, count in report["parsing_status_counts"].items():
        lines.append(f"- `{status}`: {count}")

    lines.extend(["", "## Observed Risks", ""])
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    if report["papers_without_chunks"]:
        lines.append("- Papers without chunks: " + ", ".join(report["papers_without_chunks"]))
    if report["duplicate_chunk_ids"]:
        lines.append("- Duplicate chunk IDs: " + ", ".join(report["duplicate_chunk_ids"]))
    if report["missing_field_counts"]:
        lines.append("- Missing field counts:")
        for field, count in report["missing_field_counts"].items():
            lines.append(f"  - `{field}`: {count}")
    if (
        not report["warnings"]
        and not report["papers_without_chunks"]
        and not report["duplicate_chunk_ids"]
        and not report["missing_field_counts"]
    ):
        lines.append("- No validation risks detected.")

    lines.extend(
        [
            "",
            "## Next Step Recommendation",
            "",
            report["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)
