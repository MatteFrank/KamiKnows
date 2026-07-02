"""Offline tests for Phase 2 Task 06 light reranking v1."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag.reranker_v1 import (
    LightRerankerV1,
    build_baseline_retrieval_records,
    build_eval_set_hardened_markdown,
    build_question_deltas,
    build_reranked_retrieval_records,
    build_reranking_eval_summary,
    evaluate_retrieval,
)
from scripts.run_light_reranking_v1 import parse_args, resolve_run_settings


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _chunk(chunk_id: str, arxiv_id: str, text: str, *, title: str = "Paper", section: str = "Method", flags=None) -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": arxiv_id.replace(".", "_"),
        "arxiv_id": arxiv_id,
        "title": title,
        "section": section,
        "section_heading": section,
        "source_type": "section",
        "quality_flags": list(flags or []),
        "text": text,
    }


def _tiny_chunks() -> list[dict]:
    return [
        _chunk(
            "chunk_expected",
            "1111.00001v1",
            "CaloGAN uses a generative adversarial network for fast calorimeter shower simulation.",
            title="CaloGAN fast calorimeter simulation",
            section="Method",
        ),
        _chunk(
            "chunk_other",
            "2222.00002v1",
            "Normalizing flows provide likelihood evaluation for calorimeter showers.",
            title="CaloFlow",
            section="Introduction",
            flags=["very_long_text"],
        ),
        _chunk(
            "chunk_empty",
            "3333.00003v1",
            "",
            title="Empty",
            section="Discussion",
        ),
    ]


def _tiny_question() -> dict:
    return {
        "question_id": "q_method",
        "question": "Which method performs fast calorimeter shower simulation?",
        "question_type": "method",
        "expected_behavior": "answerable",
        "expected_arxiv_ids": ["1111.00001v1"],
        "expected_paper_ids": ["1111_00001v1"],
        "expected_section_keywords": ["method", "simulation"],
        "required_all_expected_sources": False,
    }


def _candidate(rank: int, chunk: dict, score: float) -> dict:
    return {
        "rank": rank,
        "score": score,
        "chunk_id": chunk["chunk_id"],
        "paper_id": chunk["paper_id"],
        "arxiv_id": chunk["arxiv_id"],
        "title": chunk["title"],
        "section": chunk["section"],
        "section_heading": chunk["section_heading"],
        "source_type": chunk["source_type"],
        "quality_flags": chunk["quality_flags"],
        "text_preview": chunk["text"][:120],
        "text": chunk["text"],
    }


def test_light_reranker_does_not_use_expected_arxiv_ids() -> None:
    reranker = LightRerankerV1()
    candidates = [_candidate(1, _tiny_chunks()[0], 0.4), _candidate(2, _tiny_chunks()[1], 0.3)]
    base_question = _tiny_question()
    changed_expected = {**base_question, "expected_arxiv_ids": ["9999.99999v9"]}

    first = reranker.rerank(base_question, candidates, final_top_k=2)
    second = reranker.rerank(changed_expected, candidates, final_top_k=2)

    assert [(r["chunk_id"], r["final_score"]) for r in first] == [
        (r["chunk_id"], r["final_score"]) for r in second
    ]


def test_reranker_score_breakdown_and_sorted_ranks() -> None:
    reranker = LightRerankerV1()
    candidates = [_candidate(1, _tiny_chunks()[1], 0.3), _candidate(2, _tiny_chunks()[0], 0.25)]

    reranked = reranker.rerank(_tiny_question(), candidates, final_top_k=2)

    assert "score_breakdown" in reranked[0]
    assert reranked[0]["final_score"] >= reranked[1]["final_score"]
    assert [record["reranked_rank"] for record in reranked] == [1, 2]


def test_citation_labels_are_reassigned_after_reranking() -> None:
    reranker = LightRerankerV1()
    candidates = [_candidate(1, _tiny_chunks()[1], 0.3), _candidate(2, _tiny_chunks()[0], 0.25)]

    reranked = reranker.rerank(_tiny_question(), candidates, final_top_k=2)

    assert [record["citation_label"] for record in reranked] == ["[C1]", "[C2]"]
    assert reranked[0]["citation_label"] == "[C1]"
    assert reranked[0]["original_rank"] in {1, 2}


def test_summary_calculates_hit_at_5_hit_at_1_and_mrr() -> None:
    question = _tiny_question()
    baseline_records = [
        {
            "question_id": question["question_id"],
            "question": question["question"],
            "question_type": question["question_type"],
            "expected_behavior": question["expected_behavior"],
            "expected_arxiv_ids": question["expected_arxiv_ids"],
            "expected_paper_ids": question["expected_paper_ids"],
            "required_all_expected_sources": False,
            "candidate_top_n": 2,
            "final_top_k": 2,
            "baseline_top_k_results": [_candidate(1, _tiny_chunks()[1], 0.3), _candidate(2, _tiny_chunks()[0], 0.2)],
            "candidate_top_n_results": [_candidate(1, _tiny_chunks()[1], 0.3), _candidate(2, _tiny_chunks()[0], 0.2)],
            "baseline_retrieval_eval": {"expected_source_hit": True, "expected_source_hit_rank": 2},
            "candidate_pool_eval": {"expected_source_hit": True, "expected_source_hit_rank": 2},
        }
    ]
    reranked_records = [
        {
            "question_id": question["question_id"],
            "question": question["question"],
            "question_type": question["question_type"],
            "expected_behavior": question["expected_behavior"],
            "expected_arxiv_ids": question["expected_arxiv_ids"],
            "expected_paper_ids": question["expected_paper_ids"],
            "required_all_expected_sources": False,
            "reranked_top_k_results": [_candidate(1, _tiny_chunks()[0], 0.2), _candidate(2, _tiny_chunks()[1], 0.3)],
            "reranked_retrieval_eval": {"expected_source_hit": True, "expected_source_hit_rank": 1},
        }
    ]

    deltas = build_question_deltas(
        baseline_records=baseline_records,
        reranked_records=reranked_records,
        final_top_k=2,
    )
    summary = build_reranking_eval_summary(question_deltas=deltas, candidate_top_n=2, final_top_k=2)

    assert summary["baseline_hit_at_5"] == 1.0
    assert summary["reranked_hit_at_5"] == 1.0
    assert summary["baseline_hit_at_1"] == 0.0
    assert summary["reranked_hit_at_1"] == 1.0
    assert summary["baseline_mrr"] == 0.5
    assert summary["reranked_mrr"] == 1.0
    assert summary["improved_questions"] == ["q_method"]


def test_required_all_expected_sources_evaluated_correctly() -> None:
    question = {
        "expected_arxiv_ids": ["1111.00001v1", "2222.00002v1"],
        "expected_paper_ids": ["1111_00001v1", "2222_00002v1"],
        "required_all_expected_sources": True,
    }

    incomplete = evaluate_retrieval(question=question, retrieved_chunks=[_candidate(1, _tiny_chunks()[0], 0.4)], top_k=1)
    complete = evaluate_retrieval(
        question=question,
        retrieved_chunks=[_candidate(1, _tiny_chunks()[0], 0.4), _candidate(2, _tiny_chunks()[1], 0.3)],
        top_k=2,
    )

    assert incomplete["expected_source_hit"] is False
    assert incomplete["missing_expected_source_ids"] == ["2222.00002v1"]
    assert complete["expected_source_hit"] is True
    assert complete["expected_source_hit_rank"] == 2


def test_candidate_pool_contains_expected_source_calculated() -> None:
    baseline = build_baseline_retrieval_records(
        chunks=_tiny_chunks(),
        eval_questions=[_tiny_question()],
        candidate_top_n=3,
        final_top_k=1,
    )[0]

    assert baseline["candidate_pool_eval"]["expected_source_hit"] is True


def test_eval_set_markdown_generated_from_jsonl() -> None:
    markdown = build_eval_set_hardened_markdown([_tiny_question()])

    assert "# RAG Eval Set v1 Hardened" in markdown
    assert "## Question table" in markdown
    assert "Which method performs fast calorimeter shower simulation?" in markdown


def test_cli_reads_config_and_supports_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "reranking_config.json"
    _write_json(
        config_path,
        {
            "input": {
                "rag_ready_dir": "config/rag",
                "eval_questions_file": "config/eval.jsonl",
            },
            "output": {
                "output_dir": "config/output",
                "report_path": "config/report.md",
            },
            "retrieval": {
                "candidate_top_n": 20,
                "final_top_k": 5,
            },
            "reranking": {
                "score_weights": {"base_retrieval_score": 0.5},
            },
        },
    )
    args = parse_args(
        [
            "--config",
            str(config_path),
            "--rag-ready-dir",
            "cli/rag",
            "--eval-questions-file",
            "cli/eval.jsonl",
            "--output-dir",
            "cli/output",
            "--candidate-top-n",
            "10",
            "--final-top-k",
            "4",
        ]
    )

    settings = resolve_run_settings(args)

    assert settings["rag_ready_dir"] == Path("cli/rag")
    assert settings["eval_questions_file"] == Path("cli/eval.jsonl")
    assert settings["output_dir"] == Path("cli/output")
    assert settings["report_path"] == Path("config/report.md")
    assert settings["candidate_top_n"] == 10
    assert settings["final_top_k"] == 4
    assert settings["score_weights"] == {"base_retrieval_score": 0.5}
