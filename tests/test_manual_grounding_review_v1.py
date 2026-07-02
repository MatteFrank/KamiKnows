"""Offline tests for Phase 2 Task 05 manual grounding review artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from kamiknows.rag.manual_grounding_review_v1 import run_manual_grounding_review_v1
from kamiknows.rag.qa_grounded_v1 import build_eval_set_v1


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _tiny_inputs(tmp_path: Path) -> dict[str, Path]:
    qa_dir = tmp_path / "qa"
    chunks_file = tmp_path / "chunks_v1.jsonl"
    eval_set_file = qa_dir / "rag_eval_set_v1.jsonl"
    contexts_file = qa_dir / "retrieved_contexts_v1.jsonl"
    answers_file = qa_dir / "rag_answers_tiny_model_v1.jsonl"
    audit_file = qa_dir / "citation_audit_v1.jsonl"
    citation_summary_file = qa_dir / "citation_validation_summary_v1.json"
    eval_summary_file = qa_dir / "rag_eval_summary_v1.json"

    eval_set = [
        {
            "question_id": "eval_tiny_q001",
            "question": "Which method is used?",
            "question_type": "method",
            "expected_behavior": "answerable",
            "expected_arxiv_ids": ["1111.00001v1"],
            "expected_paper_ids": ["1111_00001v1"],
        },
        {
            "question_id": "eval_tiny_q002",
            "question": "Compare two selected papers.",
            "question_type": "comparison",
            "expected_behavior": "answerable",
            "expected_arxiv_ids": ["1111.00001v1", "2222.00002v1"],
            "expected_paper_ids": ["1111_00001v1", "2222_00002v1"],
        },
    ]
    contexts = [
        {
            "question_id": "eval_tiny_q001",
            "retrieval_eval": {
                "has_expected_source": True,
                "expected_source_hit": True,
                "expected_source_hit_rank": 1,
            },
            "retrieved_chunks": [
                {
                    "citation_id": "C1",
                    "citation_label": "[C1]",
                    "rank": 1,
                    "chunk_id": "chunk_1",
                    "arxiv_id": "1111.00001v1",
                    "quality_flags": [],
                    "text": "The method is a GAN.",
                }
            ],
        },
        {
            "question_id": "eval_tiny_q002",
            "retrieval_eval": {
                "has_expected_source": True,
                "expected_source_hit": False,
                "expected_source_hit_rank": None,
            },
            "retrieved_chunks": [
                {
                    "citation_id": "C1",
                    "citation_label": "[C1]",
                    "rank": 1,
                    "chunk_id": "chunk_3",
                    "arxiv_id": "3333.00003v1",
                    "quality_flags": ["very_long_text"],
                    "text": "Unrelated context.",
                }
            ],
        },
    ]
    answers = [
        {
            "question_id": "eval_tiny_q001",
            "question": "Which method is used?",
            "question_type": "method",
            "expected_behavior": "answerable",
            "model_name": "tiny:model",
            "model_slug": "tiny_model",
            "answer_status": "answered",
            "answer": "A GAN is used [C1].",
            "evidence": "C1",
            "citations": ["C1"],
            "limitations": "",
            "retrieval_eval": contexts[0]["retrieval_eval"],
            "retrieved_chunks": contexts[0]["retrieved_chunks"],
            "citation_validation": {"status": "valid", "valid": True},
            "warnings": [],
        },
        {
            "question_id": "eval_tiny_q002",
            "question": "Compare two selected papers.",
            "question_type": "comparison",
            "expected_behavior": "answerable",
            "model_name": "tiny:model",
            "model_slug": "tiny_model",
            "answer_status": "answered",
            "answer": "Only unrelated context was retrieved [C1].",
            "evidence": "C1",
            "citations": ["C1"],
            "limitations": "",
            "retrieval_eval": contexts[1]["retrieval_eval"],
            "retrieved_chunks": contexts[1]["retrieved_chunks"],
            "citation_validation": {"status": "valid", "valid": True},
            "warnings": [],
        },
    ]
    audit = [
        {
            "question_id": answer["question_id"],
            "model_name": "tiny:model",
            "citation_validation_status": "valid",
            "citation_validation_valid": True,
            "warnings": [],
        }
        for answer in answers
    ]

    _write_jsonl(chunks_file, [])
    _write_jsonl(eval_set_file, eval_set)
    _write_jsonl(contexts_file, contexts)
    _write_jsonl(answers_file, answers)
    _write_jsonl(audit_file, audit)
    _write_json(
        citation_summary_file,
        {
            "invalid_citation_count": 0,
            "invented_citation_count": 0,
            "answered_without_valid_citation_count": 0,
            "status_counts": {"valid": 2},
        },
    )
    _write_json(
        eval_summary_file,
        {
            "expected_source_hit_rate": 0.5,
            "expected_source_hits": 1,
            "expected_source_questions": 2,
            "retrieval_miss_question_ids": ["eval_tiny_q002"],
        },
    )
    return {
        "qa_dir": qa_dir,
        "chunks_file": chunks_file,
        "eval_set_file": eval_set_file,
        "contexts_file": contexts_file,
        "answers_file": answers_file,
        "audit_file": audit_file,
        "citation_summary_file": citation_summary_file,
        "eval_summary_file": eval_summary_file,
    }


def test_manual_grounding_review_outputs_are_pending_and_traceable(tmp_path: Path) -> None:
    paths = _tiny_inputs(tmp_path)
    output_dir = tmp_path / "manual_review"
    report_path = tmp_path / "phase2_task05.md"

    result = run_manual_grounding_review_v1(
        qa_output_dir=paths["qa_dir"],
        output_dir=output_dir,
        report_path=report_path,
        answer_files=[paths["answers_file"]],
        eval_set_file=paths["eval_set_file"],
        retrieved_contexts_file=paths["contexts_file"],
        citation_audit_file=paths["audit_file"],
        citation_validation_summary_file=paths["citation_summary_file"],
        rag_eval_summary_file=paths["eval_summary_file"],
        chunks_file=paths["chunks_file"],
    )

    queue_path = output_dir / "manual_review_queue_v1.jsonl"
    sheet_path = output_dir / "manual_review_sheet_v1.csv"
    summary_path = output_dir / "manual_review_summary_v1.json"
    audit_summary_path = output_dir / "eval_question_audit_summary_v1.json"

    assert queue_path.exists()
    assert sheet_path.exists()
    assert summary_path.exists()
    assert audit_summary_path.exists()
    assert report_path.exists()

    queue = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert len(queue) == 2
    assert {record["manual_review_status"] for record in queue} == {"pending"}
    assert queue[1]["manual_grounding"] == "pending"
    assert "retrieval_miss" in queue[1]["suggested_failure_modes"]
    assert "multi_source_retrieval_incomplete" in queue[1]["suggested_failure_modes"]

    with sheet_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["manual_citation_support"] == "pending"
    assert rows[1]["retrieval_expected_source_hit"] == "false"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_review_records"] == 2
    assert summary["pending_count"] == 2
    assert summary["citation_validation_clean"] is True
    assert summary["retrieval_miss_questions"] == ["eval_tiny_q002"]
    assert result["manual_review_summary"] == summary


def test_build_eval_set_can_load_static_v1_questions_from_file(tmp_path: Path) -> None:
    static_file = tmp_path / "static_questions.jsonl"
    _write_jsonl(
        static_file,
        [
            {
                "question_id": "eval_v1_custom",
                "question": "What is the static question?",
                "question_type": "fact_lookup",
                "expected_behavior": "answerable",
                "expected_arxiv_ids": ["1111.00001v1"],
            }
        ],
    )

    eval_set = build_eval_set_v1(
        [],
        static_v1_questions_enabled=True,
        static_v1_questions_path=static_file,
    )

    assert [record["question_id"] for record in eval_set] == ["eval_v1_custom"]
    assert eval_set[0]["source"] == "eval_questions_v1_extension"
