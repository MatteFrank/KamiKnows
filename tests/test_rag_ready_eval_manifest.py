"""Tests for RAG-ready eval scaffold and manifest generation."""

from __future__ import annotations

from pathlib import Path

from kamiknows.rag_ready.eval_questions import generate_eval_questions
from kamiknows.rag_ready.manifest import build_rag_manifest


def _papers() -> list[dict]:
    return [
        {
            "paper_id": f"paper_{index}",
            "arxiv_id": f"170{index}.0000{index}v1",
            "title": f"Paper {index}",
            "parsing_status": "success",
        }
        for index in range(1, 6)
    ]


def test_generate_eval_questions_has_required_distribution() -> None:
    questions = generate_eval_questions(
        papers=_papers(),
        domain="HEP / FastCaloSimulation / calorimetry",
    )

    types = [question["question_type"] for question in questions]
    assert len(questions) >= 8
    assert types.count("method") >= 2
    assert types.count("limitation") >= 2
    assert types.count("comparison") >= 2
    assert types.count("source_lookup") >= 1
    assert types.count("unanswerable") >= 1
    assert all(question["expected_source_arxiv_ids"] for question in questions)
    assert all(question["evaluation_criteria"] for question in questions)


def test_build_rag_manifest_records_counts_and_validation() -> None:
    validation = {
        "status": "PASS",
        "orphan_chunks": 0,
        "empty_chunks": 0,
        "missing_required_fields": {},
        "duplicate_chunk_ids": [],
        "papers_without_chunks": [],
        "warnings": [],
    }

    manifest = build_rag_manifest(
        source_fulltext_dir=Path("outputs/fulltext_fake"),
        output_dir=Path("outputs/rag_fake"),
        domain="HEP / FastCaloSimulation / calorimetry",
        papers=[{"paper_id": "p1"}],
        chunks=[{"chunk_id": "c1"}],
        equations=[{"equation_id": "e1"}],
        eval_questions=[{"question_id": "q1"}],
        validation=validation,
    )

    assert manifest["dataset_name"] == "rag_ready_fastcalo_v0"
    assert manifest["counts"] == {
        "papers": 1,
        "chunks": 1,
        "equations": 1,
        "eval_questions": 1,
    }
    assert manifest["validation"]["status"] == "PASS"
    assert "no embeddings" in manifest["scope"]
