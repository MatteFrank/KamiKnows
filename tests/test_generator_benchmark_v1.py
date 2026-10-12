"""Offline tests for Phase 2 Task 07 RAG generator benchmark v1."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag.generator_benchmark_v1 import (
    build_abstention_analysis,
    build_benchmark_citation_validation_summary,
    build_benchmark_matrix,
    build_citation_failure_audit,
    build_prompt,
    build_recommended_config,
    classify_citation_failure,
    extract_used_citation_labels_from_record,
    filter_benchmark_configs,
    filter_eval_questions,
    load_benchmark_contexts,
    run_rag_generator_benchmark_v1,
    validate_context_label_mapping,
)
from kamiknows.rag.qa_grounded_v1 import parse_model_json_response, validate_answer_citations
from scripts.run_rag_generator_benchmark_v1 import build_dry_run_matrix, parse_args, resolve_run_settings


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _chunk(chunk_id: str, arxiv_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": arxiv_id.replace(".", "_"),
        "arxiv_id": arxiv_id,
        "title": "Tiny Calorimeter Paper",
        "section": "Method",
        "section_heading": "Method",
        "source_type": "section",
        "quality_flags": [],
        "text": text,
    }


def _tiny_chunks() -> list[dict]:
    return [
        _chunk("chunk_expected", "1111.00001v1", "CaloGAN uses a GAN for calorimeter shower simulation."),
        _chunk("chunk_other", "2222.00002v1", "Normalizing flows estimate likelihoods."),
    ]


def _tiny_questions() -> list[dict]:
    return [
        {
            "question_id": "q001",
            "question": "Which method performs calorimeter shower simulation?",
            "question_type": "method",
            "expected_behavior": "answerable",
            "expected_arxiv_ids": ["1111.00001v1"],
            "expected_paper_ids": ["1111_00001v1"],
            "expected_section_keywords": ["method", "simulation"],
            "required_all_expected_sources": False,
        },
        {
            "question_id": "q002",
            "question": "What exact production deployment date is stated?",
            "question_type": "no_answer",
            "expected_behavior": "insufficient_evidence",
            "expected_arxiv_ids": [],
            "expected_paper_ids": [],
            "required_all_expected_sources": False,
        },
    ]


def _tiny_context() -> dict:
    return {
        "question_id": "q001",
        "question": "Which method performs calorimeter shower simulation?",
        "question_type": "method",
        "expected_behavior": "answerable",
        "required_all_expected_sources": False,
        "context_source": "tfidf_baseline_top5",
        "top_k": 1,
        "retrieval_eval": {"has_expected_source": True, "expected_source_hit": True},
        "retrieved_chunks": [
            {
                "citation_id": "C1",
                "citation_label": "[C1]",
                "rank": 1,
                "chunk_id": "chunk_expected",
                "paper_id": "1111_00001v1",
                "arxiv_id": "1111.00001v1",
                "title": "Tiny Calorimeter Paper",
                "section": "Method",
                "source_type": "section",
                "quality_flags": [],
                "text": "CaloGAN uses a GAN for calorimeter shower simulation.",
            }
        ],
    }


def test_benchmark_matrix_generation() -> None:
    matrix = build_benchmark_matrix(
        context_sources=["tfidf_baseline_top5", "light_reranked_top5"],
        prompt_profiles=[{"name": "grounded_json_v1"}, {"name": "strict_abstention_v2"}],
        model_names=["m1", "m2"],
        parameter_sets=[{"name": "deterministic_512"}],
        backend="ollama",
        top_k=5,
    )

    assert len(matrix["configs"]) == 8
    assert matrix["configs"][0]["config_id"].startswith("cfg_001")


def test_context_loading_tfidf_fallback() -> None:
    contexts = load_benchmark_contexts(
        chunks=_tiny_chunks(),
        eval_questions=_tiny_questions(),
        reranking_output_dir=None,
        include_tfidf_baseline=True,
        include_light_reranked=True,
        fallback_to_tfidf_if_reranking_missing=True,
        top_k=1,
    )

    assert set(contexts) == {"tfidf_baseline_top5", "light_reranked_top5"}
    assert contexts["tfidf_baseline_top5"][0]["retrieved_chunks"][0]["citation_label"] == "[C1]"
    assert contexts["light_reranked_top5"][0]["context_warning"] == "reranking_missing_tfidf_fallback"


def test_context_loading_from_p2t6_outputs(tmp_path: Path) -> None:
    rerank_dir = tmp_path / "reranking"
    _write_jsonl(
        rerank_dir / "baseline_retrieval_results_v1.jsonl",
        [
            {
                **_tiny_questions()[0],
                "baseline_top_k_results": _tiny_context()["retrieved_chunks"],
                "baseline_retrieval_eval": {"has_expected_source": True, "expected_source_hit": True},
            }
        ],
    )
    _write_jsonl(
        rerank_dir / "reranked_retrieval_results_v1.jsonl",
        [
            {
                **_tiny_questions()[0],
                "reranked_top_k_results": _tiny_context()["retrieved_chunks"],
                "reranked_retrieval_eval": {"has_expected_source": True, "expected_source_hit": True},
            }
        ],
    )

    contexts = load_benchmark_contexts(
        chunks=_tiny_chunks(),
        eval_questions=_tiny_questions(),
        reranking_output_dir=rerank_dir,
        include_tfidf_baseline=True,
        include_light_reranked=True,
        fallback_to_tfidf_if_reranking_missing=True,
        top_k=1,
    )

    assert len(contexts["tfidf_baseline_top5"]) == 1
    assert len(contexts["light_reranked_top5"]) == 1
    assert contexts["light_reranked_top5"][0]["retrieved_chunks"][0]["citation_label"] == "[C1]"


def test_prompt_builders() -> None:
    grounded = build_prompt(prompt_profile="grounded_json_v1", context_record=_tiny_context())
    strict = build_prompt(prompt_profile="strict_abstention_v2", context_record=_tiny_context())

    assert "Return only valid JSON" in grounded
    assert "required_all_expected_sources" in strict
    assert "expected_arxiv_ids" not in strict
    assert 'For answer_status="answered", citations must contain at least one valid citation_id' in grounded
    assert 'For answer_status="answered", citations must contain at least one valid citation_id' in strict


def test_json_parsing_fallback_and_citation_validation() -> None:
    parsed = parse_model_json_response("GAN answer [C1]")
    validation = validate_answer_citations(
        parsed_answer=parsed,
        retrieved_chunks=_tiny_context()["retrieved_chunks"],
        known_chunk_ids=["chunk_expected"],
    )

    assert "json_parse_error" in parsed["parse_warnings"]
    assert validation["status"] == "valid"


def test_citation_extraction_ignores_embedded_uuid_fragments() -> None:
    parsed = parse_model_json_response('{"id": "5d62ecf2-b0af-4c98-a9c8-beba3c7190e4"}')

    assert parsed["citations"] == []


def test_citation_failure_classifier_citations_in_text_but_not_array() -> None:
    record = {
        **_tiny_context(),
        "benchmark_config_id": "cfg_test",
        "model_name": "tiny:model",
        "prompt_profile": "grounded_json_v1",
        "context_source": "tfidf_baseline_top5",
        "answer_status": "answered",
        "answer": "CaloGAN uses a GAN [C1].",
        "evidence": "",
        "limitations": "",
        "citations": [],
        "unsupported_or_inferred": [],
        "parse_warnings": [],
        "citation_validation": {
            "valid": True,
            "valid_used_citation_labels": ["C1"],
            "invented_citation_labels": [],
            "answered_without_valid_citation": False,
        },
    }

    assert extract_used_citation_labels_from_record(record)["all"] == ["C1"]
    assert classify_citation_failure(record) == "citations_in_text_but_not_array"


def test_citation_failure_audit_answered_without_valid_and_invented() -> None:
    no_citation_record = {
        **_tiny_context(),
        "benchmark_config_id": "cfg_no_cite",
        "model_name": "tiny:model",
        "prompt_profile": "grounded_json_v1",
        "context_source": "tfidf_baseline_top5",
        "answer_status": "answered",
        "answer": "CaloGAN uses a GAN.",
        "evidence": "",
        "limitations": "",
        "citations": [],
        "unsupported_or_inferred": [],
        "raw_model_response": "",
        "parse_warnings": [],
        "citation_validation": {
            "status": "answered_without_valid_citation",
            "valid": False,
            "valid_used_citation_labels": [],
            "invented_citation_labels": [],
            "answered_without_valid_citation": True,
        },
    }
    invented_record = {
        **no_citation_record,
        "benchmark_config_id": "cfg_invented",
        "citations": ["C9"],
        "citation_validation": {
            "status": "invented_citation",
            "valid": False,
            "valid_used_citation_labels": [],
            "invented_citation_labels": ["C9"],
            "answered_without_valid_citation": False,
        },
    }

    audit = build_citation_failure_audit([no_citation_record, invented_record])

    assert audit[0]["suspected_failure_cause"] == "model_returned_empty_citations_array"
    assert audit[0]["answered_without_valid_citation"] is True
    assert audit[1]["suspected_failure_cause"] == "citation_label_not_in_context"
    assert audit[1]["invented_citation_labels"] == ["C9"]


def test_context_label_mapping_for_reranked_contexts() -> None:
    record = {
        **_tiny_context(),
        "context_source": "light_reranked_top5",
        "retrieved_chunks": [
            {**_tiny_context()["retrieved_chunks"][0], "rank": 1, "citation_id": "C1", "citation_label": "[C1]"},
            {
                **_tiny_context()["retrieved_chunks"][0],
                "rank": 2,
                "citation_id": "C2",
                "citation_label": "[C2]",
                "chunk_id": "chunk_other",
            },
        ],
    }

    summary = validate_context_label_mapping([record])

    assert summary["status"] == "valid"
    assert summary["issue_count"] == 0


def test_reduced_selection_helpers() -> None:
    questions = filter_eval_questions(_tiny_questions(), question_ids=["q002"], limit_questions=1)
    configs = filter_benchmark_configs(
        [{"config_id": "cfg_001"}, {"config_id": "cfg_002"}, {"config_id": "cfg_003"}],
        config_ids=["cfg_001", "cfg_003"],
        limit_configs=1,
    )

    assert [question["question_id"] for question in questions] == ["q002"]
    assert [config["config_id"] for config in configs] == ["cfg_001"]


def test_abstention_analysis_and_recommended_config() -> None:
    records = [
        {
            "benchmark_config_id": "cfg_good",
            "model_name": "m1",
            "prompt_profile": "strict_abstention_v2",
            "context_source": "tfidf_baseline_top5",
            "expected_behavior": "insufficient_evidence",
            "answer_status": "insufficient_evidence",
            "citation_validation": {"valid": True, "invented_citation_labels": []},
        },
        {
            "benchmark_config_id": "cfg_bad",
            "model_name": "m2",
            "prompt_profile": "grounded_json_v1",
            "context_source": "tfidf_baseline_top5",
            "expected_behavior": "answerable",
            "answer_status": "answered",
            "citation_validation": {
                "valid": False,
                "invented_citation_labels": ["C9"],
                "answered_without_valid_citation": False,
            },
        },
    ]

    abstention = build_abstention_analysis(records)
    citation_summary = build_benchmark_citation_validation_summary(records)
    recommended = build_recommended_config(
        [
            {
                "config_id": "cfg_good",
                "invented_citation_count": 0,
                "answered_without_valid_citation_count": 0,
                "backend_error_count": 0,
                "json_parse_error_count": 0,
                "failed_abstention_possible_overclaim_count": 0,
                "answerable_over_abstention_count": 0,
                "invalid_citation_count": 0,
                "citation_validation_clean": True,
                "retrieval_expected_source_hit_rate": 1.0,
                "answered_count": 1,
            },
            {
                "config_id": "cfg_bad",
                "invented_citation_count": 1,
                "answered_without_valid_citation_count": 0,
                "backend_error_count": 0,
                "json_parse_error_count": 0,
                "failed_abstention_possible_overclaim_count": 0,
                "answerable_over_abstention_count": 0,
                "invalid_citation_count": 1,
                "citation_validation_clean": False,
                "retrieval_expected_source_hit_rate": 1.0,
                "answered_count": 1,
            },
        ]
    )

    assert abstention["correct_abstention_count"] == 1
    assert citation_summary["invented_citation_count"] == 1
    assert recommended["recommended_for_manual_review"] == "cfg_good"
    assert recommended["rejected_configs"][0]["config_id"] == "cfg_bad"


def test_run_outputs_manifest_with_backend_unavailable(tmp_path: Path, monkeypatch) -> None:
    rag_dir = tmp_path / "rag_ready"
    _write_jsonl(rag_dir / "chunks_v1.jsonl", _tiny_chunks())
    _write_json(rag_dir / "rag_manifest_v1.json", {"papers_count": 2, "chunks_v1_count": 2, "equations_count": 0})
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(eval_file, _tiny_questions()[:1])

    monkeypatch.setattr(
        "kamiknows.rag.generator_benchmark_v1.list_local_ollama_models",
        lambda base_url: {"available": False, "model_names": [], "backend_error": "fake unavailable"},
    )

    result = run_rag_generator_benchmark_v1(
        rag_ready_dir=rag_dir,
        eval_questions_file=eval_file,
        reranking_output_dir=None,
        output_dir=tmp_path / "benchmark",
        report_path=tmp_path / "report.md",
        prompt_profiles=[{"name": "grounded_json_v1", "description": "test"}],
        model_names=["tiny:model"],
        parameter_sets=[{"name": "deterministic_512", "temperature": 0.0, "top_p": 0.9, "num_predict": 32, "timeout_seconds": 1}],
        include_light_reranked=False,
    )

    assert result["summary"]["answer_records_total"] == 1
    assert result["answer_records"][0]["answer_status"] == "backend_error"
    assert (tmp_path / "benchmark" / "benchmark_manifest_v1.json").exists()


def test_cli_config_and_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_json(
        config_path,
        {
            "input": {
                "rag_ready_dir": "config/rag",
                "eval_questions_file": "config/eval.jsonl",
                "reranking_output_dir": "config/rerank",
            },
            "output": {
                "output_dir": "config/output",
                "report_path": "config/report.md",
            },
            "contexts": {"top_k": 5, "include_tfidf_baseline": True, "include_light_reranked": True},
            "prompt_profiles": [{"name": "grounded_json_v1", "description": "test"}],
            "answer_generation": {
                "backend": "ollama",
                "base_url": "http://config",
                "models": ["config:model"],
                "parameter_sets": [{"name": "deterministic_512"}],
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
            "--models",
            "cli:model",
            "--prompt-profiles",
            "strict_abstention_v2",
            "--context-sources",
            "tfidf_baseline_top5",
            "--limit-questions",
            "2",
            "--limit-configs",
            "1",
            "--question-ids",
            "q001",
            "q002",
            "--config-ids",
            "cfg_001__tfidf_baseline_top5__strict_abstention_v2__cli_model__deterministic_512",
            "--dry-run-matrix",
            "--base-url",
            "http://cli",
            "--timeout-seconds",
            "3",
        ]
    )

    settings = resolve_run_settings(args)

    assert settings["rag_ready_dir"] == Path("cli/rag")
    assert settings["eval_questions_file"] == Path("cli/eval.jsonl")
    assert settings["model_names"] == ["cli:model"]
    assert settings["prompt_profiles"] == [{"name": "strict_abstention_v2", "description": ""}]
    assert settings["context_sources"] == ["tfidf_baseline_top5"]
    assert settings["limit_questions"] == 2
    assert settings["limit_configs"] == 1
    assert settings["question_ids"] == ["q001", "q002"]
    assert settings["config_ids"] == ["cfg_001__tfidf_baseline_top5__strict_abstention_v2__cli_model__deterministic_512"]
    assert settings["dry_run_matrix"] is True
    assert settings["base_url"] == "http://cli"
    assert settings["timeout_seconds_override"] == 3

    matrix = build_dry_run_matrix(settings)
    assert matrix["selection"]["dry_run_matrix"] is True
    assert len(matrix["configs"]) == 1
