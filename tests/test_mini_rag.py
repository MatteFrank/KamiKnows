"""Offline tests for mini-RAG v0."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag.evaluation import evaluate_expected_source_hit
from kamiknows.rag.load_dataset import load_rag_ready_dataset
from kamiknows.rag.outputs import run_mini_rag, write_jsonl
from kamiknows.rag.prompting import build_grounded_qa_prompt
from kamiknows.rag.retriever import TfidfRetriever, tokenize


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
        "title": f"Paper {arxiv_id}",
        "source_type": "section",
        "section_id": "sec_001",
        "section_heading": "Method",
        "text": text,
        "word_count": len(text.split()),
        "metadata": {
            "domain": "HEP / FastCaloSimulation / calorimetry",
            "source_pilot": "outputs/fulltext_fake",
            "paper_dir": f"outputs/fulltext_fake/processed/papers/{arxiv_id}",
            "source_file": f"outputs/fulltext_fake/processed/papers/{arxiv_id}/chunks.jsonl",
            "parsing_status": "success",
            "source_type_original": "latex_source",
        },
    }


def _make_fake_rag_ready_dir(tmp_path: Path) -> Path:
    rag_dir = tmp_path / "rag_ready"
    chunks = [
        _chunk(
            "chunk_calo",
            "1705.00001v1",
            "Fast calorimeter simulation uses surrogate shower models for detector response.",
        ),
        _chunk(
            "chunk_higgs",
            "1705.00002v1",
            "Higgs boson analysis uses event selection and invariant mass reconstruction.",
        ),
    ]
    papers = [
        {
            "paper_id": "1705_00001v1",
            "arxiv_id": "1705.00001v1",
            "title": "Fast calorimeter simulation",
            "abstract": "A",
            "parsing_status": "success",
            "source_type": "latex_source",
            "sections_count": 1,
            "equations_count": 0,
            "chunks_count": 1,
            "plain_text_word_count": 9,
            "paper_dir": "fake/p1",
            "files": {},
        },
        {
            "paper_id": "1705_00002v1",
            "arxiv_id": "1705.00002v1",
            "title": "Higgs analysis",
            "abstract": "B",
            "parsing_status": "success",
            "source_type": "latex_source",
            "sections_count": 1,
            "equations_count": 0,
            "chunks_count": 1,
            "plain_text_word_count": 8,
            "paper_dir": "fake/p2",
            "files": {},
        },
    ]
    eval_questions = [
        {
            "question_id": "q1",
            "question": "What method is used for fast calorimeter simulation?",
            "question_type": "method",
            "expected_source_arxiv_ids": ["1705.00001v1"],
            "expected_section_keywords": ["method", "simulation"],
            "evaluation_criteria": "Must retrieve the calorimeter source.",
            "notes": "",
        }
    ]
    _write_jsonl(rag_dir / "chunks.jsonl", chunks)
    _write_jsonl(rag_dir / "papers.jsonl", papers)
    _write_jsonl(rag_dir / "equations.jsonl", [])
    _write_jsonl(rag_dir / "eval_questions_v0.jsonl", eval_questions)
    _write_json(
        rag_dir / "rag_manifest_v0.json",
        {
            "counts": {
                "papers": 2,
                "chunks": 2,
                "equations": 0,
                "eval_questions": 1,
            },
            "validation": {"status": "PASS"},
        },
    )
    return rag_dir


def test_load_rag_ready_dataset_from_tiny_fixture(tmp_path: Path) -> None:
    rag_dir = _make_fake_rag_ready_dir(tmp_path)

    dataset = load_rag_ready_dataset(rag_dir)

    assert dataset["counts"] == {
        "papers": 2,
        "chunks": 2,
        "equations": 0,
        "eval_questions": 1,
    }


def test_tokenizer_and_tfidf_retriever_are_deterministic() -> None:
    assert tokenize("Fast-Calo simulation, fast!") == ["fast", "calo", "simulation", "fast"]
    chunks = [
        _chunk("chunk_calo", "1705.00001v1", "calorimeter shower simulation surrogate"),
        _chunk("chunk_higgs", "1705.00002v1", "higgs invariant mass reconstruction"),
    ]

    retriever = TfidfRetriever(chunks)
    results = retriever.retrieve("calorimeter simulation method", top_k=2)

    assert results[0].chunk["chunk_id"] == "chunk_calo"
    assert results[0].score > results[1].score


def test_expected_source_hit_calculation() -> None:
    retrieved = [
        {"rank": 1, "arxiv_id": "1705.00002v1"},
        {"rank": 2, "arxiv_id": "1705.00001v1"},
    ]

    result = evaluate_expected_source_hit(retrieved, ["1705.00001v1"])

    assert result["expected_source_hit"] is True
    assert result["expected_source_hit_rank"] == 2


def test_grounded_prompt_contains_context_and_citation_instructions() -> None:
    prompt = build_grounded_qa_prompt(
        question="What is the method?",
        retrieved_chunks=[
            {
                "rank": 1,
                "chunk_id": "chunk_calo",
                "arxiv_id": "1705.00001v1",
                "title": "Fast calorimeter simulation",
                "section_heading": "Method",
                "text": "The method uses surrogate shower models.",
            }
        ],
    )

    assert "Answer the question using ONLY the provided context" in prompt
    assert "Cite chunk_id and arxiv_id" in prompt
    assert "chunk_calo" in prompt
    assert "1705.00001v1" in prompt


def test_output_writing_for_contexts_and_answers(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"question_id": "q1"}, {"question_id": "q2"}])

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["question_id"] == "q1"


def test_retrieval_only_run_creates_required_outputs(tmp_path: Path) -> None:
    rag_dir = _make_fake_rag_ready_dir(tmp_path)
    output_dir = tmp_path / "rag_run"

    result = run_mini_rag(
        rag_ready_dir=rag_dir,
        output_dir=output_dir,
        backend="ollama",
        model="qwen3:0.6b",
        top_k=1,
        retrieval_only=True,
    )

    assert result["eval_summary"]["questions_total"] == 1
    assert result["eval_summary"]["generation"]["retrieval_only"] is True
    assert result["answers"][0]["generation_status"] == "skipped_retrieval_only"
    assert (output_dir / "rag_run_config.json").exists()
    assert (output_dir / "retrieved_contexts.jsonl").exists()
    assert (output_dir / "rag_answers.jsonl").exists()
    assert (output_dir / "rag_eval_summary.json").exists()
    assert (output_dir / "rag_error_analysis.md").exists()
    assert (output_dir / "rag_manual_review_checklist.md").exists()
    assert (output_dir / "rag_manifest.json").exists()


def test_manifest_generation_in_retrieval_only_run(tmp_path: Path) -> None:
    rag_dir = _make_fake_rag_ready_dir(tmp_path)
    output_dir = tmp_path / "rag_run"

    result = run_mini_rag(
        rag_ready_dir=rag_dir,
        output_dir=output_dir,
        backend="ollama",
        model="qwen3:0.6b",
        top_k=1,
        retrieval_only=True,
    )

    manifest = result["manifest"]
    assert manifest["dataset_counts"] == {
        "papers": 2,
        "chunks": 2,
        "equations": 0,
        "eval_questions": 1,
    }
    assert manifest["retriever"]["name"] == "tfidf_local_v0"
    assert manifest["generator"]["enabled"] is False
    assert manifest["validation"]["status"] == "PASS"
