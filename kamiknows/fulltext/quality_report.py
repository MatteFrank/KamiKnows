"""Quality reporting for the Fase 1F full-text parsing pilot."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from kamiknows.fulltext.arxiv_source import ERROR_CATEGORIES
from kamiknows.fulltext.paper_outputs import write_json


def build_fulltext_manifest(
    *,
    output_dir: Path,
    ids_file: Path,
    paper_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a small manifest for full-text pilot artifacts."""
    paper_entries = []
    for result in paper_results:
        files = result.get("files", {})
        paper_entries.append(
            {
                "paper_id": result.get("paper_id"),
                "arxiv_id": result.get("arxiv_id"),
                "parsing_status": result.get("parsing_status"),
                "source_type": result.get("source_type"),
                "files": files,
            }
        )
    return {
        "manifest_version": "fulltext_manifest_v0",
        "scope": "full-text parsing artifacts only; not RAG, not embeddings, not scientific QA",
        "ids_file": str(ids_file),
        "output_dir": str(output_dir),
        "papers": paper_entries,
    }


def _all_errors(result: dict[str, Any]) -> list[dict[str, str]]:
    errors = list(result.get("errors", []) or [])
    warnings = list(result.get("warnings", []) or [])
    return errors + warnings


def build_quality_report(
    *,
    requested_ids: list[str],
    paper_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a machine-readable full-text parsing quality report."""
    status_counts = Counter(result.get("parsing_status", "failed") for result in paper_results)
    source_counts = Counter(result.get("source_type", "unavailable") for result in paper_results)
    taxonomy_counts = {category: 0 for category in ERROR_CATEGORIES}

    for result in paper_results:
        for error in _all_errors(result):
            category = str(error.get("category") or "unknown_error")
            if category not in taxonomy_counts:
                category = "unknown_error"
            taxonomy_counts[category] += 1

    if status_counts.get("failed", 0):
        recommendation = "Fix failed source downloads or parsing failures before building RAG-ready chunks."
    elif status_counts.get("partial", 0):
        recommendation = "Inspect partial papers, then improve LaTeX parsing only where errors repeat."
    else:
        recommendation = "Proceed to inspect chunks manually before any RAG experiment."

    return {
        "report_version": "fulltext_parsing_quality_report_v0",
        "scope": "full-text parsing quality only; this is not RAG and not scientific QA",
        "requested_papers": len(requested_ids),
        "successfully_processed": status_counts.get("success", 0),
        "partially_processed": status_counts.get("partial", 0),
        "failed": status_counts.get("failed", 0),
        "latex_source_available": source_counts.get("latex_source", 0),
        "pdf_fallback_count": source_counts.get("pdf_fallback", 0),
        "unavailable_or_error": source_counts.get("unavailable", 0) + source_counts.get("error", 0),
        "error_taxonomy": taxonomy_counts,
        "papers": [
            {
                "paper_id": result.get("paper_id"),
                "arxiv_id": result.get("arxiv_id"),
                "title": result.get("title"),
                "parsing_status": result.get("parsing_status"),
                "source_type": result.get("source_type"),
                "sections_count": result.get("sections_count"),
                "equations_count": result.get("equations_count"),
                "chunks_count": result.get("chunks_count"),
                "plain_text_word_count": result.get("plain_text_word_count"),
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),
            }
            for result in paper_results
        ],
        "recommendation": recommendation,
    }


def format_quality_report_markdown(report: dict[str, Any]) -> str:
    """Format the full-text parsing quality report as Markdown."""
    lines = [
        "# KamiKnows Fase 1F Full-Text Parsing Quality Report",
        "",
        "This report covers parsing artifacts only. It is not RAG and not scientific QA.",
        "",
        "## Summary",
        "",
        f"- Requested papers: {report['requested_papers']}",
        f"- Successfully processed: {report['successfully_processed']}",
        f"- Partially processed: {report['partially_processed']}",
        f"- Failed: {report['failed']}",
        f"- LaTeX source available: {report['latex_source_available']}",
        f"- PDF fallback count: {report['pdf_fallback_count']}",
        f"- Unavailable/error count: {report['unavailable_or_error']}",
        "",
        "## Per-Paper Status",
        "",
        "| arXiv ID | Status | Source | Sections | Equations | Chunks | Words |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for paper in report["papers"]:
        lines.append(
            "| {arxiv_id} | {status} | {source} | {sections} | {equations} | {chunks} | {words} |".format(
                arxiv_id=paper.get("arxiv_id", ""),
                status=paper.get("parsing_status", ""),
                source=paper.get("source_type", ""),
                sections=paper.get("sections_count", 0),
                equations=paper.get("equations_count", 0),
                chunks=paper.get("chunks_count", 0),
                words=paper.get("plain_text_word_count", 0),
            )
        )

    lines.extend(["", "## Parsing Error Taxonomy", ""])
    for category, count in report["error_taxonomy"].items():
        lines.append(f"- `{category}`: {count}")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            report["recommendation"],
            "",
            "## Boundary",
            "",
            "This pilot does not add RAG, embeddings, vector DB, PDF parsing, LaTeX semantic interpretation, LoRA, fine-tuning, model calls, or scientific QA.",
            "",
        ]
    )
    return "\n".join(lines)


def write_quality_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown quality reports."""
    json_path = write_json(report, output_dir / "parsing_quality_report.json")
    markdown_path = output_dir / "parsing_quality_report.md"
    markdown_path.write_text(format_quality_report_markdown(report), encoding="utf-8")
    return json_path, markdown_path
