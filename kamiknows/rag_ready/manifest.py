"""Manifest helpers for RAG-ready dataset v0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kamiknows.run_metadata import utc_now_iso


def build_rag_manifest(
    *,
    source_fulltext_dir: Path,
    output_dir: Path,
    domain: str,
    papers: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    equations: list[dict[str, Any]],
    eval_questions: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Build the RAG-ready dataset manifest."""
    return {
        "dataset_name": "rag_ready_fastcalo_v0",
        "source_fulltext_dir": str(source_fulltext_dir),
        "output_dir": str(output_dir),
        "created_at": utc_now_iso(),
        "domain": domain,
        "counts": {
            "papers": len(papers),
            "chunks": len(chunks),
            "equations": len(equations),
            "eval_questions": len(eval_questions),
        },
        "files": {
            "chunks": str(output_dir / "chunks.jsonl"),
            "papers": str(output_dir / "papers.jsonl"),
            "equations": str(output_dir / "equations.jsonl"),
            "eval_questions": str(output_dir / "eval_questions_v0.jsonl"),
            "chunk_quality_report_json": str(output_dir / "chunk_quality_report.json"),
            "chunk_quality_report_md": str(output_dir / "chunk_quality_report.md"),
        },
        "validation": {
            "status": validation["status"],
            "orphan_chunks": validation["orphan_chunks"],
            "empty_chunks": validation["empty_chunks"],
            "missing_required_fields": validation["missing_required_fields"],
            "duplicate_chunk_ids": validation["duplicate_chunk_ids"],
            "papers_without_chunks": validation["papers_without_chunks"],
            "warnings": validation["warnings"],
        },
        "scope": "RAG-ready dataset only; no embeddings, no vector DB, no retrieval, no generated answers.",
    }
