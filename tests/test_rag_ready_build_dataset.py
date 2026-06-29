"""Tests for building a RAG-ready dataset from fake full-text outputs."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag_ready.build_dataset import build_rag_ready_dataset


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _make_fake_fulltext_tree(base: Path) -> Path:
    fulltext_dir = base / "fulltext_fake"
    paper_dir = fulltext_dir / "processed" / "papers" / "p1"
    chunks_path = paper_dir / "chunks.jsonl"
    equations_path = paper_dir / "equations.json"
    paper = {
        "paper_id": "p1",
        "arxiv_id": "1705.02355v2",
        "title": "Fast calorimeter simulation",
        "abstract": "A test abstract.",
        "parsing_status": "success",
        "source_type": "latex_source",
        "sections_count": 1,
        "equations_count": 1,
        "chunks_count": 1,
        "plain_text_word_count": 12,
        "files": {
            "chunks": str(chunks_path),
            "equations": str(equations_path),
            "paper": str(paper_dir / "paper.json"),
        },
    }
    chunk = {
        "chunk_id": "p1_chunk_0001",
        "paper_id": "p1",
        "arxiv_id": "1705.02355v2",
        "title": "Fast calorimeter simulation",
        "section_id": "sec_001",
        "section_heading": "Introduction",
        "source_type": "section",
        "text": "Fast calorimeter simulation can reduce detector simulation cost.",
        "word_count": 8,
    }
    equation = {
        "equation_id": "eq_001",
        "raw_latex": "\\begin{equation}E=mc^2\\end{equation}",
        "section_id": "sec_001",
        "context_before": "Before",
        "context_after": "After",
    }
    _write_json(paper_dir / "paper.json", paper)
    _write_jsonl(chunks_path, [chunk])
    _write_json(equations_path, [equation])
    return fulltext_dir


def test_build_rag_ready_dataset_from_tiny_fake_tree(tmp_path: Path) -> None:
    fulltext_dir = _make_fake_fulltext_tree(tmp_path)
    output_dir = tmp_path / "rag_ready"

    manifest = build_rag_ready_dataset(
        fulltext_dir=fulltext_dir,
        output_dir=output_dir,
        domain="HEP / FastCaloSimulation / calorimetry",
    )

    assert manifest["validation"]["status"] == "PASS"
    assert manifest["counts"]["papers"] == 1
    assert manifest["counts"]["chunks"] == 1
    assert manifest["counts"]["equations"] == 1
    assert manifest["counts"]["eval_questions"] >= 8
    assert (output_dir / "chunks.jsonl").exists()
    assert (output_dir / "papers.jsonl").exists()
    assert (output_dir / "equations.jsonl").exists()
    assert (output_dir / "eval_questions_v0.jsonl").exists()
    assert (output_dir / "rag_manifest_v0.json").exists()
    assert (output_dir / "chunk_quality_report.json").exists()
    assert (output_dir / "chunk_quality_report.md").exists()

    chunk = json.loads((output_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert chunk["metadata"]["domain"] == "HEP / FastCaloSimulation / calorimetry"
    assert chunk["metadata"]["source_type_original"] == "latex_source"
