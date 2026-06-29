"""Build RAG-ready dataset v0 from Fase 1F full-text outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kamiknows.fulltext.paper_outputs import write_json, write_jsonl
from kamiknows.rag_ready.eval_questions import generate_eval_questions
from kamiknows.rag_ready.manifest import build_rag_manifest
from kamiknows.rag_ready.validate_chunks import (
    build_chunk_quality_report,
    format_chunk_quality_report_markdown,
    validate_chunks,
)

INCLUDED_PARSING_STATUSES = {"success", "partial"}


class RagReadyDatasetError(RuntimeError):
    """Raised when the RAG-ready dataset cannot be built."""


def read_json(path: Path) -> Any:
    """Read JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a small JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise RagReadyDatasetError(f"{path}:{line_number} is not a JSON object")
            records.append(parsed)
    return records


def discover_paper_dirs(fulltext_dir: Path) -> list[Path]:
    """Return sorted per-paper directories from a Fase 1F output tree."""
    papers_dir = fulltext_dir / "processed" / "papers"
    if not papers_dir.exists():
        raise RagReadyDatasetError(f"missing papers directory: {papers_dir}")
    return sorted(path for path in papers_dir.iterdir() if path.is_dir())


def load_paper_record(paper_dir: Path) -> dict[str, Any]:
    """Load and normalize one paper.json into papers.jsonl schema."""
    paper_path = paper_dir / "paper.json"
    if not paper_path.exists():
        raise RagReadyDatasetError(f"missing paper.json: {paper_path}")
    paper = read_json(paper_path)
    return {
        "paper_id": paper.get("paper_id", paper_dir.name),
        "arxiv_id": paper.get("arxiv_id", ""),
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "parsing_status": paper.get("parsing_status", ""),
        "source_type": paper.get("source_type", ""),
        "sections_count": int(paper.get("sections_count", 0) or 0),
        "equations_count": int(paper.get("equations_count", 0) or 0),
        "chunks_count": int(paper.get("chunks_count", 0) or 0),
        "plain_text_word_count": int(paper.get("plain_text_word_count", 0) or 0),
        "paper_dir": str(paper_dir),
        "files": paper.get("files", {}),
    }


def _resolve_source_file(paper_dir: Path, raw_path: str | None, fallback_name: str) -> Path:
    if raw_path:
        path = Path(raw_path)
        if path.exists():
            return path
    return paper_dir / fallback_name


def load_rag_chunks(
    *,
    paper: dict[str, Any],
    paper_dir: Path,
    source_fulltext_dir: Path,
    domain: str,
) -> list[dict[str, Any]]:
    """Load source chunks and enrich them with RAG-ready metadata."""
    if paper.get("parsing_status") not in INCLUDED_PARSING_STATUSES:
        return []
    files = paper.get("files", {})
    chunks_path = _resolve_source_file(paper_dir, files.get("chunks"), "chunks.jsonl")
    if not chunks_path.exists():
        return []

    chunks = []
    for chunk in read_jsonl(chunks_path):
        chunks.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "paper_id": paper.get("paper_id", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title") or chunk.get("title", ""),
                "source_type": chunk.get("source_type", ""),
                "section_id": chunk.get("section_id", ""),
                "section_heading": chunk.get("section_heading", ""),
                "text": chunk.get("text", ""),
                "word_count": int(chunk.get("word_count", 0) or 0),
                "metadata": {
                    "domain": domain,
                    "source_pilot": str(source_fulltext_dir),
                    "paper_dir": str(paper_dir),
                    "source_file": str(chunks_path),
                    "parsing_status": paper.get("parsing_status", ""),
                    "source_type_original": paper.get("source_type", ""),
                },
            }
        )
    return chunks


def load_equation_records(
    *,
    paper: dict[str, Any],
    paper_dir: Path,
) -> list[dict[str, Any]]:
    """Flatten one paper's equations into global equation records."""
    if paper.get("parsing_status") not in INCLUDED_PARSING_STATUSES:
        return []
    files = paper.get("files", {})
    equations_path = _resolve_source_file(paper_dir, files.get("equations"), "equations.json")
    if not equations_path.exists():
        return []
    equations = read_json(equations_path)
    if not isinstance(equations, list):
        raise RagReadyDatasetError(f"equations file must contain a list: {equations_path}")
    records = []
    for equation in equations:
        if not isinstance(equation, dict):
            continue
        equation_id = str(equation.get("equation_id", ""))
        records.append(
            {
                "equation_id": equation_id,
                "global_equation_id": f"{paper.get('paper_id')}_{equation_id}",
                "paper_id": paper.get("paper_id", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title", ""),
                "section_id": equation.get("section_id"),
                "raw_latex": equation.get("raw_latex", ""),
                "context_before": equation.get("context_before", ""),
                "context_after": equation.get("context_after", ""),
                "source_file": str(equations_path),
            }
        )
    return records


def collect_dataset_records(
    *,
    fulltext_dir: Path,
    domain: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect papers, chunks, and equations from full-text outputs."""
    papers: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []

    for paper_dir in discover_paper_dirs(fulltext_dir):
        paper = load_paper_record(paper_dir)
        if paper.get("parsing_status") in INCLUDED_PARSING_STATUSES:
            papers.append(paper)
            chunks.extend(
                load_rag_chunks(
                    paper=paper,
                    paper_dir=paper_dir,
                    source_fulltext_dir=fulltext_dir,
                    domain=domain,
                )
            )
            equations.extend(load_equation_records(paper=paper, paper_dir=paper_dir))

    return papers, chunks, equations


def build_rag_ready_dataset(
    *,
    fulltext_dir: Path,
    output_dir: Path,
    domain: str,
) -> dict[str, Any]:
    """Build all RAG-ready dataset v0 files."""
    fulltext_dir = Path(fulltext_dir)
    output_dir = Path(output_dir)
    if not fulltext_dir.exists():
        raise RagReadyDatasetError(f"full-text directory does not exist: {fulltext_dir}")
    if not domain.strip():
        raise RagReadyDatasetError("domain must be a non-empty string")

    output_dir.mkdir(parents=True, exist_ok=True)

    papers, chunks, equations = collect_dataset_records(
        fulltext_dir=fulltext_dir,
        domain=domain,
    )
    eval_questions = generate_eval_questions(papers=papers, domain=domain)
    validation = validate_chunks(chunks, papers=papers)
    quality_report = build_chunk_quality_report(
        papers=papers,
        chunks=chunks,
        validation=validation,
    )
    manifest = build_rag_manifest(
        source_fulltext_dir=fulltext_dir,
        output_dir=output_dir,
        domain=domain,
        papers=papers,
        chunks=chunks,
        equations=equations,
        eval_questions=eval_questions,
        validation=validation,
    )

    write_jsonl(chunks, output_dir / "chunks.jsonl")
    write_jsonl(papers, output_dir / "papers.jsonl")
    write_jsonl(equations, output_dir / "equations.jsonl")
    write_jsonl(eval_questions, output_dir / "eval_questions_v0.jsonl")
    write_json(quality_report, output_dir / "chunk_quality_report.json")
    (output_dir / "chunk_quality_report.md").write_text(
        format_chunk_quality_report_markdown(quality_report),
        encoding="utf-8",
    )
    write_json(manifest, output_dir / "rag_manifest_v0.json")
    return manifest
