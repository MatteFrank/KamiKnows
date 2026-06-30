"""Offline tests for Phase 2 embedding model selection v1."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag.embedding_registry import (
    apply_embedding_prefix,
    get_embedding_model_config,
    get_embedding_registry,
)
from kamiknows.rag.embedding_selection import (
    EmbeddingEncoder,
    compute_embedding_metrics,
    evaluate_embedding_retrieval,
    hash_embedding,
    retrieve_embedding_chunks,
    run_embedding_model_selection_v1,
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
        "section_id": "sec_001",
        "section_heading": "Method",
        "text": text,
        "word_count": len(text.split()),
        "metadata": {"parsing_status": "success"},
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
            "chunk_limit",
            "1705.00002v1",
            "Validation caveats describe detector studies and uncertainty checks.",
            title="Validation limitations",
        ),
        _chunk(
            "chunk_higgs",
            "1705.00003v1",
            "Higgs event selection reconstructs invariant mass candidates.",
            title="Higgs analysis",
        ),
    ]


def _tiny_questions() -> list[dict]:
    return [
        {
            "question_id": "q_calo",
            "question": "Which method performs calorimeter shower simulation?",
            "question_type": "method",
            "expected_source_arxiv_ids": ["1705.00001v1"],
        },
        {
            "question_id": "q_limit",
            "question": "What validation caveat is discussed?",
            "question_type": "limitation",
            "expected_source_arxiv_ids": ["1705.00002v1"],
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
                "eval_questions": 2,
            },
            "validation": {"status": "PASS"},
        },
    )
    return rag_dir


def test_embedding_registry_contains_required_models() -> None:
    registry = get_embedding_registry()
    names = {model["model_name"] for model in registry}

    assert "sentence-transformers/all-MiniLM-L6-v2" in names
    assert "intfloat/e5-small-v2" in names
    assert "BAAI/bge-m3" in names
    assert all("normalize_embeddings" in model for model in registry)


def test_e5_prefix_handling() -> None:
    config = get_embedding_model_config("e5-small-v2")

    assert apply_embedding_prefix(config, "hello", text_kind="query") == "query: hello"
    assert apply_embedding_prefix(config, "hello", text_kind="passage") == "passage: hello"


def test_hash_embedding_shape_is_consistent() -> None:
    vector = hash_embedding(
        "calorimeter shower simulation",
        dimension=16,
        model_name="test-model",
    )

    assert len(vector) == 16
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-9


def test_embedding_encoder_hash_backend_shape() -> None:
    config = get_embedding_model_config("all-MiniLM-L6-v2")
    encoder = EmbeddingEncoder(config, backend="hash")

    vectors = encoder.encode(["calorimeter simulation", "validation caveat"], text_kind="query")

    assert len(vectors) == 2
    assert len(vectors[0]) == config["embedding_dimension"]
    assert encoder.backend == "hash"


def test_embedding_retrieval_top_k_and_metrics() -> None:
    config = get_embedding_model_config("all-MiniLM-L6-v2")
    encoder = EmbeddingEncoder(config, backend="hash")
    chunks = _tiny_chunks()
    chunk_vectors = encoder.encode([chunk["text"] for chunk in chunks], text_kind="passage")
    query_vector = encoder.encode(["calorimeter shower simulation"], text_kind="query")[0]

    retrieved = retrieve_embedding_chunks(
        query_vector=query_vector,
        chunk_vectors=chunk_vectors,
        chunks=chunks,
        top_k=2,
    )
    retrieval_eval = evaluate_embedding_retrieval(
        retrieved_chunks=retrieved,
        question={"expected_source_arxiv_ids": ["1705.00001v1"]},
    )
    metrics = compute_embedding_metrics(
        [
            {
                "question_id": "q_calo",
                "retrieval_eval": retrieval_eval,
            }
        ]
    )

    assert len(retrieved) == 2
    assert retrieved[0]["citation_label"] == "[C1]"
    assert retrieved[1]["citation_label"] == "[C2]"
    assert retrieval_eval["expected_source_hit"] is True
    assert metrics["expected_source_hit_rate_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_embedding_selection_outputs_and_citation_labels(tmp_path: Path) -> None:
    rag_dir = _make_fake_rag_ready_dir(tmp_path)
    output_dir = tmp_path / "embedding_selection"

    result = run_embedding_model_selection_v1(
        rag_ready_dir=rag_dir,
        output_dir=output_dir,
        model_names_or_keys=["all-MiniLM-L6-v2", "e5-small-v2", "bge-m3"],
        top_k=2,
        encoder_backend="hash",
    )

    assert result["eval_summary"]["models_total"] == 3
    assert (output_dir / "embedding_model_registry_v1.json").exists()
    assert (output_dir / "embedding_benchmark_results_v1.jsonl").exists()
    assert (output_dir / "embedding_eval_summary_v1.json").exists()
    assert (output_dir / "citation_validation_summary_v1.json").exists()
    assert (output_dir / "embedding_model_comparison_v1.md").exists()
    assert result["citation_summary"]["all_citations_reference_retrieved_chunks"] is True

    first_record = result["benchmark_records"][0]["question_records"][0]
    labels = [chunk["citation_label"] for chunk in first_record["retrieved_chunks"]]
    assert labels == ["[C1]", "[C2]"]
    assert first_record["citation_validation"]["status"] == "valid"
