"""Tests for RAG-ready chunk validation."""

from __future__ import annotations

from kamiknows.rag_ready.validate_chunks import validate_chunks


def _paper(paper_id: str = "p1") -> dict:
    return {
        "paper_id": paper_id,
        "arxiv_id": "1705.02355v2",
        "title": "A paper",
        "parsing_status": "success",
    }


def _chunk(chunk_id: str = "p1_chunk_0001", paper_id: str = "p1") -> dict:
    text = "This chunk describes a fast calorimeter simulation method."
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "arxiv_id": "1705.02355v2",
        "title": "A paper",
        "source_type": "section",
        "section_id": "sec_001",
        "section_heading": "Introduction",
        "text": text,
        "word_count": 8,
        "metadata": {
            "domain": "HEP / FastCaloSimulation / calorimetry",
            "source_pilot": "outputs/fulltext_fake",
            "paper_dir": "outputs/fulltext_fake/processed/papers/p1",
            "source_file": "outputs/fulltext_fake/processed/papers/p1/chunks.jsonl",
            "parsing_status": "success",
            "source_type_original": "latex_source",
        },
    }


def test_validate_chunks_passes_valid_chunk_schema() -> None:
    report = validate_chunks([_chunk()], papers=[_paper()])

    assert report["status"] == "PASS"
    assert report["orphan_chunks"] == 0
    assert report["empty_chunks"] == 0
    assert report["missing_required_fields"] == {}
    assert report["duplicate_chunk_ids"] == []


def test_validate_chunks_detects_duplicate_chunk_ids() -> None:
    report = validate_chunks([_chunk(), _chunk()], papers=[_paper()])

    assert report["status"] == "FAIL"
    assert report["duplicate_chunk_ids"] == ["p1_chunk_0001"]


def test_validate_chunks_detects_orphan_chunks() -> None:
    report = validate_chunks([_chunk(paper_id="missing")], papers=[_paper()])

    assert report["status"] == "FAIL"
    assert report["orphan_chunks"] == 1
