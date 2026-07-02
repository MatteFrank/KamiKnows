"""Offline tests for Phase 2 QA grounded v1."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag.qa_grounded_v1 import (
    DEFAULT_MODEL_NAMES,
    DEFAULT_HARDENED_EVAL_QUESTIONS_PATH,
    build_default_model_registry,
    build_eval_set_v1,
    build_retrieved_qa_chunk,
    load_eval_questions_from_file,
    parse_model_json_response,
    retrieve_contexts_v1,
    run_qa_grounded_v1,
    validate_answer_citations,
)
from scripts.run_qa_grounded_v1 import load_qa_grounded_config, parse_args, resolve_run_settings


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


def _make_fake_rag_ready_v1_without_eval_questions(tmp_path: Path) -> Path:
    rag_dir = tmp_path / "rag_ready_v1_no_eval"
    _write_jsonl(rag_dir / "chunks_v1.jsonl", _tiny_chunks())
    _write_json(
        rag_dir / "rag_manifest_v1.json",
        {
            "manifest_version": "rag_manifest_v1",
            "papers_count": 2,
            "chunks_v1_count": 2,
            "equations_count": 0,
            "eval_questions_count": 0,
            "quality_summary": {"chunks_with_quality_flags": 0, "quality_flag_counts": {}},
        },
    )
    return rag_dir


def _tiny_eval_questions_file(tmp_path: Path) -> Path:
    path = tmp_path / "eval_questions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "question_id": "eval_file_q001",
                "question": "Which method performs calorimeter shower simulation?",
                "question_type": "method",
                "expected_behavior": "answerable",
                "expected_arxiv_ids": ["1705.00001v1"],
                "expected_source_arxiv_ids": ["1705.00001v1"],
                "expected_paper_ids": ["1705_00001v1"],
                "expected_section_keywords": ["method", "simulation"],
                "required_all_expected_sources": False,
                "source": "test_eval_file",
            }
        ],
    )
    return path


def test_model_registry_contains_required_generators() -> None:
    registry = build_default_model_registry()
    names = {model["model_name"] for model in registry["models"]}

    assert DEFAULT_MODEL_NAMES == ("qwen3:0.6b", "qwen3:4b-instruct", "mistral:7b")
    assert {"qwen3:0.6b", "qwen3:4b-instruct", "mistral:7b"} <= names
    assert "qwen3:4b" not in names
    assert all(model["backend"] == "ollama" for model in registry["models"])
    assert all(model["enabled"] is True for model in registry["models"])


def test_eval_set_v1_preserves_v0_without_implicit_static_questions() -> None:
    eval_set = build_eval_set_v1(_tiny_questions_v0())
    behaviors = {question["question_id"]: question["expected_behavior"] for question in eval_set}

    assert [question["question_id"] for question in eval_set] == ["eval_v0_q001", "eval_v0_q008"]
    assert len(eval_set) == 2
    assert behaviors["eval_v0_q008"] == "insufficient_evidence"


def test_hardened_eval_set_file_has_expected_questions() -> None:
    records = load_eval_questions_from_file(DEFAULT_HARDENED_EVAL_QUESTIONS_PATH)
    question_ids = [record["question_id"] for record in records]

    assert len(records) == 13
    assert len(question_ids) == len(set(question_ids))
    assert question_ids[:8] == [f"eval_v0_q{i:03d}" for i in range(1, 9)]
    assert question_ids[8:] == [f"eval_v1_q{i:03d}" for i in range(9, 14)]
    by_id = {record["question_id"]: record for record in records}
    assert by_id["eval_v0_q008"]["expected_behavior"] == "insufficient_evidence"
    assert by_id["eval_v1_q013"]["expected_behavior"] == "insufficient_evidence"
    assert by_id["eval_v1_q012"]["required_all_expected_sources"] is True


def test_no_operational_questions_are_hardcoded_in_qa_module() -> None:
    source = Path("kamiknows/rag/qa_grounded_v1.py").read_text(encoding="utf-8")

    assert "What named model is introduced" not in source
    assert "How does the CaloGAN setup encourage" not in source
    assert "What exact ATLAS Run 4 production deployment date" not in source
    assert "first selected FastCalo paper" not in source


def test_eval_set_v1_uses_eval_questions_file_only(tmp_path: Path) -> None:
    eval_file = _tiny_eval_questions_file(tmp_path)

    eval_set = build_eval_set_v1(
        _tiny_questions_v0(),
        eval_questions_file=eval_file,
        static_v1_questions_enabled=False,
    )

    assert [question["question_id"] for question in eval_set] == ["eval_file_q001"]
    assert eval_set[0]["source"] == "test_eval_file"


def test_legacy_static_questions_require_explicit_file(tmp_path: Path) -> None:
    static_file = _tiny_eval_questions_file(tmp_path)
    eval_set = build_eval_set_v1(
        _tiny_questions_v0(),
        static_v1_questions_enabled=True,
        static_v1_questions_path=static_file,
        llm_question_generation_enabled=False,
    )

    assert len(eval_set) == 3
    assert {question["source"] for question in eval_set} == {
        "eval_questions_v0",
        "test_eval_file",
    }


def test_legacy_static_questions_fail_without_explicit_file() -> None:
    try:
        build_eval_set_v1(_tiny_questions_v0(), static_v1_questions_enabled=True)
    except ValueError as exc:
        assert "static_v1_questions_path is required" in str(exc)
    else:
        raise AssertionError("legacy static questions must require an explicit data file")


def test_load_eval_questions_from_file_fails_if_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    try:
        load_eval_questions_from_file(missing)
    except Exception as exc:
        assert "eval questions file does not exist" in str(exc)
    else:
        raise AssertionError("missing eval-question files must fail")


def test_llm_question_generation_true_is_not_implemented() -> None:
    try:
        build_eval_set_v1(_tiny_questions_v0(), llm_question_generation_enabled=True)
    except ValueError as exc:
        assert "LLM question generation is not implemented" in str(exc)
    else:
        raise AssertionError("LLM question generation must remain disabled in P2T4")


def test_from_file_mode_does_not_call_deprecated_helper(tmp_path: Path, monkeypatch) -> None:
    eval_file = _tiny_eval_questions_file(tmp_path)

    def fail_if_called():
        raise AssertionError("_additional_eval_questions_v1 must not be called")

    monkeypatch.setattr("kamiknows.rag.qa_grounded_v1._additional_eval_questions_v1", fail_if_called)

    eval_set = build_eval_set_v1([], eval_questions_file=eval_file)

    assert [record["question_id"] for record in eval_set] == ["eval_file_q001"]


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


def test_retrieval_required_all_expected_sources_for_comparison() -> None:
    eval_set = [
        {
            "question_id": "eval_file_q_compare",
            "question": "Compare CaloGAN and CaloFlow.",
            "question_type": "comparison",
            "expected_behavior": "answerable",
            "expected_arxiv_ids": ["1705.00001v1", "2106.00001v1"],
            "expected_paper_ids": ["1705_00001v1", "2106_00001v1"],
            "expected_section_keywords": ["CaloGAN"],
            "required_all_expected_sources": True,
        }
    ]

    records = retrieve_contexts_v1(chunks=_tiny_chunks()[:1], eval_set=eval_set, top_k=1)

    retrieval_eval = records[0]["retrieval_eval"]
    assert retrieval_eval["expected_source_hit"] is False
    assert retrieval_eval["required_all_expected_sources"] is True
    assert retrieval_eval["missing_expected_source_ids"] == ["2106.00001v1"]


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

    assert result["eval_summary"]["questions_total"] == 2
    assert result["eval_summary"]["models_available"] == []
    assert result["eval_summary"]["backend_errors"] == 2
    assert (output_dir / "qa_model_registry_v1.json").exists()
    assert (output_dir / "rag_eval_set_v1.jsonl").exists()
    assert (output_dir / "retrieved_contexts_v1.jsonl").exists()
    assert (output_dir / "rag_answers_qwen3_0_6b_v1.jsonl").exists()
    assert (output_dir / "citation_validation_summary_v1.json").exists()
    assert (output_dir / "rag_eval_summary_v1.json").exists()
    assert (output_dir / "rag_model_comparison_v1.md").exists()
    assert (output_dir / "rag_error_analysis_v1.md").exists()
    assert report_path.exists()


def test_full_orchestration_uses_eval_questions_file_when_provided(tmp_path: Path, monkeypatch) -> None:
    rag_dir = _make_fake_rag_ready_v1_without_eval_questions(tmp_path)
    eval_file = _tiny_eval_questions_file(tmp_path)
    output_dir = tmp_path / "qa_grounded_from_file"
    report_path = tmp_path / "phase2_task04_from_file.md"

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
        eval_set_mode="from_file",
        eval_questions_file=eval_file,
    )

    assert result["eval_summary"]["questions_total"] == 1
    assert result["dataset"]["counts"]["eval_questions_v0"] == 0
    assert [record["question_id"] for record in result["eval_set"]] == ["eval_file_q001"]
    run_config = json.loads((output_dir / "qa_run_config_v1.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "qa_manifest_v1.json").read_text(encoding="utf-8"))
    assert run_config["eval_set"]["mode"] == "from_file"
    assert run_config["eval_set"]["questions_file"] == str(eval_file)
    assert manifest["eval_set"]["questions_file"] == str(eval_file)


def test_load_canonical_config_json() -> None:
    config = load_qa_grounded_config(Path("configs/phase2_p2t4_qa_grounded_v1.json"))

    assert config["input"]["rag_ready_dir"] == "outputs/rag_ready_fastcalo_v1"
    assert config["output"]["output_dir"] == "outputs/rag_v1_fastcalosim_qa_grounded_config_v1"
    assert config["retrieval"]["top_k"] == 5
    assert config["answer_generation"]["models"] == [
        "qwen3:0.6b",
        "qwen3:4b-instruct",
        "mistral:7b",
    ]
    assert config["eval_set"]["mode"] == "from_file"
    assert config["eval_set"]["questions_file"] == "data/eval/rag_eval_set_v1_hardened.jsonl"
    assert config["eval_set"]["static_v1_questions_enabled"] is False
    assert config["eval_set"]["llm_question_generation_enabled"] is False
    assert config["scope"]["no_training"] is True
    assert config["scope"]["no_discovery"] is True
    assert config["scope"]["no_corpus_scaling"] is True


def test_cli_overrides_config_values(tmp_path: Path) -> None:
    config_path = tmp_path / "qa_config.json"
    _write_json(
        config_path,
        {
            "input": {"rag_ready_dir": "from_config/rag"},
            "output": {
                "output_dir": "from_config/output",
                "report_path": "from_config/report.md",
            },
            "retrieval": {"top_k": 3},
            "answer_generation": {
                "base_url": "http://config.example",
                "models": ["config:model"],
                "temperature": 0.2,
                "top_p": 0.5,
                "num_predict": 128,
                "timeout_seconds": 30,
            },
            "eval_set": {
                "mode": "config_mode",
                "questions_file": "from_config/questions.jsonl",
                "static_v1_questions_enabled": False,
                "llm_question_generation_enabled": False,
            },
        },
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--rag-ready-dir",
            "cli/rag",
            "--output-dir",
            "cli/output",
            "--report-path",
            "cli/report.md",
            "--top-k",
            "9",
            "--base-url",
            "http://cli.example",
            "--models",
            "cli:model:a",
            "cli:model:b",
            "--temperature",
            "0.0",
            "--top-p",
            "0.95",
            "--num-predict",
            "256",
            "--timeout-seconds",
            "45",
            "--eval-questions-file",
            "cli/questions.jsonl",
        ]
    )
    settings = resolve_run_settings(args)

    assert settings["rag_ready_dir"] == Path("cli/rag")
    assert settings["output_dir"] == Path("cli/output")
    assert settings["report_path"] == Path("cli/report.md")
    assert settings["top_k"] == 9
    assert settings["base_url"] == "http://cli.example"
    assert settings["model_names"] == ["cli:model:a", "cli:model:b"]
    assert settings["temperature"] == 0.0
    assert settings["top_p"] == 0.95
    assert settings["num_predict"] == 256
    assert settings["timeout_seconds"] == 45
    assert settings["eval_set_mode"] == "config_mode"
    assert settings["eval_questions_file"] == Path("cli/questions.jsonl")
    assert settings["static_v1_questions_enabled"] is False


def test_legacy_cli_without_config_still_resolves_required_paths() -> None:
    args = parse_args(
        [
            "--rag-ready-dir",
            "outputs/rag_ready_fastcalo_v1",
            "--output-dir",
            "outputs/tmp_qa",
        ]
    )
    settings = resolve_run_settings(args)

    assert settings["rag_ready_dir"] == Path("outputs/rag_ready_fastcalo_v1")
    assert settings["output_dir"] == Path("outputs/tmp_qa")
    assert settings["top_k"] == 5
    assert settings["base_url"] == "http://localhost:11434"
    assert settings["model_names"] is None
    assert settings["eval_questions_file"] is None
    assert settings["static_v1_questions_enabled"] is False
