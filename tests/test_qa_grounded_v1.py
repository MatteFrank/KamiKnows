"""Offline tests for Phase 2 QA grounded v1."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag.qa_grounded_v1 import (
    build_default_model_registry,
    build_eval_set_v1,
    build_retrieved_qa_chunk,
    parse_model_json_response,
    retrieve_contexts_v1,
    run_qa_grounded_v1,
    validate_answer_citations,
)


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
        "section": "Method",
        "section_heading": "Method",
        "quality_flags": [],
        "text": text,
    }


def _tiny_chunks() -> list[dict]:
    return [
        _chunk(
            "chunk_calo",
            "1705.00001v1",
            "CaloGAN uses a generative adversarial network for calorimeter shower simulation.",
            title="CaloGAN",
        ),
        _chunk(
            "chunk_flow",
            "2106.00001v1",
            "CaloFlow uses normalizing flows with explicit likelihoods and stable training.",
            title="CaloFlow",
        ),
    ]


def _tiny_questions_v0() -> list[dict]:
    return [
        {
            "question_id": "eval_v0_q001",
            "question": "Which method performs calorimeter shower simulation?",
            "question_type": "method",
            "expected_source_arxiv_ids": ["1705.00001v1"],
            "expected_section_keywords": ["method", "simulation"],
            "notes": "tiny",
        },
        {
            "question_id": "eval_v0_q008",
            "question": "What exact production deployment should be used?",
            "question_type": "unanswerable",
            "expected_source_arxiv_ids": ["1705.00001v1"],
            "expected_section_keywords": ["production"],
            "notes": "tiny no-answer",
        },
    ]


def _make_fake_rag_ready_v1(tmp_path: Path) -> Path:
    rag_dir = tmp_path / "rag_ready_v1"
    _write_jsonl(rag_dir / "chunks_v1.jsonl", _tiny_chunks())
    _write_jsonl(rag_dir / "eval_questions_v0.jsonl", _tiny_questions_v0())
    _write_json(
        rag_dir / "rag_manifest_v1.json",
        {
            "manifest_version": "rag_manifest_v1",
            "papers_count": 2,
            "chunks_v1_count": 2,
            "equations_count": 0,
            "eval_questions_count": 2,
            "quality_summary": {"chunks_with_quality_flags": 0, "quality_flag_counts": {}},
        },
    )
    return rag_dir


def test_model_registry_contains_required_generators() -> None:
    registry = build_default_model_registry()
    names = {model["model_name"] for model in registry["models"]}

    assert {"qwen3:0.6b", "qwen3:4b", "mistral:7b"} <= names
    assert all(model["backend"] == "ollama" for model in registry["models"])
    assert all(model["enabled"] is True for model in registry["models"])


def test_eval_set_v1_preserves_v0_and_adds_required_types() -> None:
    eval_set = build_eval_set_v1(_tiny_questions_v0())
    question_types = {question["question_type"] for question in eval_set}
    behaviors = {question["question_id"]: question["expected_behavior"] for question in eval_set}

    assert [question["question_id"] for question in eval_set[:2]] == ["eval_v0_q001", "eval_v0_q008"]
    assert len(eval_set) == 7
    assert {"fact_lookup", "method", "limitation", "comparison", "no_answer"} <= question_types
    assert behaviors["eval_v0_q008"] == "insufficient_evidence"
    assert behaviors["eval_v1_q013"] == "insufficient_evidence"


def test_retrieval_records_have_deterministic_citation_labels() -> None:
    eval_set = build_eval_set_v1(_tiny_questions_v0())[:1]

    records = retrieve_contexts_v1(chunks=_tiny_chunks(), eval_set=eval_set, top_k=2)

    labels = [chunk["citation_label"] for chunk in records[0]["retrieved_chunks"]]
    assert labels == ["[C1]", "[C2]"]
    assert records[0]["retrieval_eval"]["expected_source_hit"] is True


def test_label_citation_validation_detects_valid_and_invented_labels() -> None:
    retrieved = [
        build_retrieved_qa_chunk(chunk=_tiny_chunks()[0], rank=1, score=1.0),
        build_retrieved_qa_chunk(chunk=_tiny_chunks()[1], rank=2, score=0.5),
    ]
    known_chunk_ids = ["chunk_calo", "chunk_flow"]
    parsed = parse_model_json_response(
        '{"answer_status":"answered","answer":"CaloGAN is used [C1].","citations":["C1"],"confidence":"high","limitations":"","unsupported_or_inferred":[]}'
    )

    valid = validate_answer_citations(
        parsed_answer=parsed,
        retrieved_chunks=retrieved,
        known_chunk_ids=known_chunk_ids,
    )
    invented = validate_answer_citations(
        parsed_answer={**parsed, "citations": ["C9"], "answer": "Unsupported [C9]."},
        retrieved_chunks=retrieved,
        known_chunk_ids=known_chunk_ids,
    )
    missing = validate_answer_citations(
        parsed_answer={**parsed, "citations": [], "answer": "No citation here."},
        retrieved_chunks=retrieved,
        known_chunk_ids=known_chunk_ids,
    )

    assert valid["status"] == "valid"
    assert invented["status"] == "invented_citation"
    assert invented["invented_citation_labels"] == ["C9"]
    assert missing["status"] == "answered_without_valid_citation"


def test_full_orchestration_writes_outputs_when_ollama_unavailable(tmp_path: Path, monkeypatch) -> None:
    rag_dir = _make_fake_rag_ready_v1(tmp_path)
    output_dir = tmp_path / "qa_grounded"
    report_path = tmp_path / "phase2_task04.md"

    def fake_tags(base_url: str, timeout_seconds: int = 5):
        return {
            "available": False,
            "model_names": [],
            "backend_error": "fake backend unavailable",
        }

    monkeypatch.setattr("kamiknows.rag.qa_grounded_v1.list_local_ollama_models", fake_tags)

    result = run_qa_grounded_v1(
        rag_ready_dir=rag_dir,
        output_dir=output_dir,
        report_path=report_path,
        model_names=["qwen3:0.6b"],
        top_k=2,
    )

    assert result["eval_summary"]["questions_total"] == 7
    assert result["eval_summary"]["models_available"] == []
    assert result["eval_summary"]["backend_errors"] == 7
    assert (output_dir / "qa_model_registry_v1.json").exists()
    assert (output_dir / "rag_eval_set_v1.jsonl").exists()
    assert (output_dir / "retrieved_contexts_v1.jsonl").exists()
    assert (output_dir / "rag_answers_qwen3_0_6b_v1.jsonl").exists()
    assert (output_dir / "citation_validation_summary_v1.json").exists()
    assert (output_dir / "rag_eval_summary_v1.json").exists()
    assert (output_dir / "rag_model_comparison_v1.md").exists()
    assert (output_dir / "rag_error_analysis_v1.md").exists()
    assert report_path.exists()
