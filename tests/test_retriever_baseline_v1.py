"""Offline tests for retrieval baseline v1 and citation hardening."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag.citation_validation import validate_citations
from kamiknows.rag.retriever_baseline_v1 import (
    build_baseline_records,
    build_eval_summary_v1,
    build_retrieval_diagnostics,
    run_retriever_baseline_v1,
)
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _chunk(chunk_id: str, arxiv_id: str, text: str, *, title: str = "Paper") -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": arxiv_id.replace(".", "_"),
        "arxiv_id": arxiv_id,
        "title": title,
        "source_type": "section",
        "section_id": "sec_001",
        "section_heading": "Method",
        "text": text,
        "word_count": len(text.split()),
        "metadata": {
            "domain": "HEP / FastCaloSimulation / calorimetry",
            "parsing_status": "success",
        },
    }


def _tiny_chunks() -> list[dict]:
    return [
        _chunk(
            "chunk_calo",
            "1705.00001v1",
            "CaloGAN uses a generative adversarial network for calorimeter shower simulation.",
            title="Fast calorimeter simulation",
        ),
        _chunk(
            "chunk_higgs",
            "1705.00002v1",
            "Higgs event selection reconstructs invariant mass from detector candidates.",
            title="Higgs analysis",
        ),
        _chunk(
            "chunk_limit",
            "1705.00003v1",
            "The validation caveat is that more detector studies are needed.",
            title="Validation limitations",
        ),
    ]


def _tiny_questions() -> list[dict]:
    return [
        {
            "question_id": "q_method",
            "question": "Which method performs calorimeter shower simulation?",
            "question_type": "method",
            "expected_source_arxiv_ids": ["1705.00001v1"],
            "expected_section_keywords": ["method", "simulation"],
        },
        {
            "question_id": "q_limit",
            "question": "What validation caveat is stated?",
            "question_type": "limitation",
            "expected_source_arxiv_ids": ["1705.00003v1"],
            "expected_section_keywords": ["validation", "caveat"],
        },
        {
            "question_id": "q_miss",
            "question": "Which accelerator lattice optics method is proposed?",
            "question_type": "method",
            "expected_source_arxiv_ids": ["1705.99999v1"],
            "expected_section_keywords": ["accelerator", "optics"],
        },
    ]


def _make_fake_rag_ready_dir(tmp_path: Path) -> Path:
    rag_dir = tmp_path / "rag_ready"
    chunks = _tiny_chunks()
    papers = [
        {
            "paper_id": chunk["paper_id"],
            "arxiv_id": chunk["arxiv_id"],
            "title": chunk["title"],
            "abstract": "A",
            "parsing_status": "success",
            "source_type": "latex_source",
            "sections_count": 1,
            "equations_count": 0,
            "chunks_count": 1,
            "plain_text_word_count": 10,
            "paper_dir": "fake",
            "files": {},
        }
        for chunk in chunks
    ]
    _write_jsonl(rag_dir / "chunks.jsonl", chunks)
    _write_jsonl(rag_dir / "papers.jsonl", papers)
    _write_jsonl(rag_dir / "equations.jsonl", [])
    _write_jsonl(rag_dir / "eval_questions_v0.jsonl", _tiny_questions())
    _write_json(
        rag_dir / "rag_manifest_v0.json",
        {
            "counts": {
                "papers": 3,
                "chunks": 3,
                "equations": 0,
                "eval_questions": 3,
            },
            "validation": {"status": "PASS"},
        },
    )
    return rag_dir


def test_tfidf_v1_deterministic_ranking_on_tiny_fixture() -> None:
    retriever = TfidfRetrieverV1(_tiny_chunks())

    first = retriever.retrieve(
        "Which method performs calorimeter shower simulation?",
        top_k=3,
        question_type="method",
    )
    second = retriever.retrieve(
        "Which method performs calorimeter shower simulation?",
        top_k=3,
        question_type="method",
    )

    assert [item.chunk["chunk_id"] for item in first] == [item.chunk["chunk_id"] for item in second]
    assert first[0].chunk["chunk_id"] == "chunk_calo"
    assert first[0].score > first[1].score


def test_hit_rate_calculation_for_top_k_values() -> None:
    records = build_baseline_records(
        chunks=_tiny_chunks(),
        eval_questions=_tiny_questions(),
        top_k_values=[1, 2, 3],
    )

    summary = build_eval_summary_v1(records=records, top_k_values=[1, 2, 3])

    assert summary["questions_total"] == 3
    assert summary["hit_rate_by_top_k"]["1"] == 2 / 3
    assert summary["hit_rate_by_top_k"]["2"] == 2 / 3
    assert summary["hit_rate_by_top_k"]["3"] == 2 / 3
    assert summary["miss_questions_by_top_k"]["3"] == ["q_miss"]


def test_retrieval_miss_diagnostics_for_missing_expected_source() -> None:
    chunks = _tiny_chunks()
    question = _tiny_questions()[-1]

    diagnostics = build_retrieval_diagnostics(
        chunks=chunks,
        question=question,
        retrieved_chunks=[],
        query_terms=["accelerator", "optics"],
    )

    assert diagnostics["matched_expected_source_chunks_available"] is False
    assert diagnostics["likely_reason"] == "metadata_problem"


def test_citation_validation_valid_citations() -> None:
    result = validate_citations(
        {"citations": [{"chunk_id": "chunk_calo"}]},
        ["chunk_calo", "chunk_higgs"],
    )

    assert result["status"] == "valid"
    assert result["valid"] is True


def test_citation_validation_missing_citations() -> None:
    result = validate_citations({"citations": []}, ["chunk_calo"])

    assert result["status"] == "missing_citations"
    assert result["valid"] is False


def test_citation_validation_citation_not_in_retrieved_chunks() -> None:
    result = validate_citations(
        {"citations": [{"chunk_id": "chunk_higgs"}]},
        ["chunk_calo"],
        known_chunk_ids=["chunk_calo", "chunk_higgs"],
    )

    assert result["status"] == "cites_unretrieved_chunk"
    assert result["unretrieved_citation_chunk_ids"] == ["chunk_higgs"]


def test_cli_orchestration_creates_required_outputs_on_tiny_fixture(tmp_path: Path) -> None:
    rag_dir = _make_fake_rag_ready_dir(tmp_path)
    output_dir = tmp_path / "retrieval_v1"

    result = run_retriever_baseline_v1(
        rag_ready_dir=rag_dir,
        output_dir=output_dir,
        top_k_values=[1, 2, 3],
    )

    assert result["eval_summary"]["questions_total"] == 3
    assert (output_dir / "retriever_baseline_v1.jsonl").exists()
    assert (output_dir / "retriever_eval_summary_v1.json").exists()
    assert (output_dir / "retriever_error_analysis_v1.md").exists()
    assert (output_dir / "retrieval_debug_report.md").exists()
    assert (output_dir / "retrieval_run_config.json").exists()
    assert (output_dir / "retrieval_manifest.json").exists()
