"""Offline tests for RAG-ready chunk metadata v1."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.rag.load_dataset import read_jsonl
from kamiknows.rag_ready.chunk_metadata_v1 import (
    build_chunks_v1,
    build_rag_ready_chunk_metadata_v1,
    build_retrieval_smoke_summary,
    normalize_section,
    quality_flags_for_v0_chunk,
    stable_chunk_id_v1,
)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _paper(arxiv_id: str, title: str) -> dict:
    paper_id = arxiv_id.replace(".", "_")
    return {
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": "A",
        "parsing_status": "success",
        "source_type": "latex_source",
        "sections_count": 1,
        "equations_count": 0,
        "chunks_count": 1,
        "plain_text_word_count": 100,
        "paper_dir": f"fake/{paper_id}",
        "files": {"chunks": f"fake/{paper_id}/chunks.jsonl"},
    }


def _chunk(chunk_id: str, arxiv_id: str, text: str, **overrides) -> dict:
    paper_id = arxiv_id.replace(".", "_")
    chunk = {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "source_type": "section",
        "section_id": "sec_001",
        "section_heading": "Method Details",
        "text": text,
        "metadata": {
            "domain": "HEP / FastCaloSimulation / calorimetry",
            "source_file": f"fake/{paper_id}/chunks.jsonl",
        },
    }
    chunk.update(overrides)
    return chunk


def _tiny_rag_ready_dir(tmp_path: Path) -> Path:
    rag_dir = tmp_path / "rag_ready_v0"
    papers = [
        _paper("1705.00001v1", "Fast calorimeter simulation"),
        _paper("1705.00002v1", "Validation caveats"),
    ]
    chunks = [
        _chunk(
            "old_calo",
            "1705.00001v1",
            "CaloGAN uses a generative adversarial network for calorimeter shower simulation.",
        ),
        _chunk(
            "old_limit",
            "1705.00002v1",
            "Validation caveats describe detector studies and uncertainty checks.",
            title="",
        ),
    ]
    orphan = _chunk("old_orphan", "9999.00000v1", "Short")
    orphan.update(
        {
            "paper_id": "missing_paper",
            "arxiv_id": "",
            "title": "",
            "section_heading": "",
            "source_type": "mystery",
            "metadata": {},
        }
    )
    chunks.append(orphan)
    eval_questions = [
        {
            "question_id": "q_calo",
            "question": "Which method performs calorimeter shower simulation?",
            "question_type": "method",
            "expected_source_arxiv_ids": ["1705.00001v1"],
            "expected_section_keywords": ["method", "simulation"],
        }
    ]
    _write_jsonl(rag_dir / "chunks.jsonl", chunks)
    _write_jsonl(rag_dir / "papers.jsonl", papers)
    _write_jsonl(rag_dir / "equations.jsonl", [])
    _write_jsonl(rag_dir / "eval_questions_v0.jsonl", eval_questions)
    _write_json(rag_dir / "rag_manifest_v0.json", {"validation": {"status": "PASS"}})
    return rag_dir


def test_load_chunks_v0_from_fixture(tmp_path: Path) -> None:
    rag_dir = _tiny_rag_ready_dir(tmp_path)

    chunks = read_jsonl(rag_dir / "chunks.jsonl")

    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == "old_calo"


def test_stable_chunk_id_and_section_normalization() -> None:
    assert stable_chunk_id_v1("1705_00001v1", 7) == "1705_00001v1::chunk_v1::0007"
    assert normalize_section(" Method Details / Results ") == "method_details_results"


def test_metadata_propagation_and_minimum_schema() -> None:
    papers = [_paper("1705.00002v1", "Validation caveats")]
    chunks = [
        _chunk(
            "old_limit",
            "1705.00002v1",
            "Validation caveats describe detector studies and uncertainty checks.",
            title="",
            metadata={},
        )
    ]

    chunks_v1, chunk_map = build_chunks_v1(chunks=chunks, papers=papers, equations=[])

    chunk = chunks_v1[0]
    assert chunk["chunk_id"] == "1705_00002v1::chunk_v1::0001"
    assert chunk_map == {"old_limit": "1705_00002v1::chunk_v1::0001"}
    assert chunk["title"] == "Validation caveats"
    assert chunk["metadata"]["version"] == "v1"
    assert chunk["metadata"]["source_file"] == "fake/1705_00002v1/chunks.jsonl"
    assert set(["chunk_id", "paper_id", "text", "source_type", "metadata", "quality_flags"]).issubset(chunk)
    assert chunk["source_refs"]["parent_chunk_id"] == "old_limit"


def test_quality_flags_and_orphan_chunk_detection() -> None:
    flags = quality_flags_for_v0_chunk(
        chunk={
            "chunk_id": "bad",
            "paper_id": "missing_paper",
            "arxiv_id": "",
            "title": "",
            "section_heading": "",
            "source_type": "mystery",
            "text": "Short",
        },
        paper_ids={"known_paper"},
        normalized_text_counts={"short": 1},
    )

    assert "missing_arxiv_id" in flags
    assert "missing_title" in flags
    assert "missing_section" in flags
    assert "unknown_source_type" in flags
    assert "very_short_text" in flags
    assert "orphan_chunk" in flags


def test_build_outputs_manifest_map_and_smoke_summary(tmp_path: Path) -> None:
    rag_dir = _tiny_rag_ready_dir(tmp_path)
    output_dir = tmp_path / "rag_ready_v1"

    result = build_rag_ready_chunk_metadata_v1(input_dir=rag_dir, output_dir=output_dir)

    assert len(result["chunks_v1"]) == 3
    assert result["manifest"]["chunks_v0_count"] == 3
    assert result["manifest"]["chunks_v1_count"] == 3
    assert result["manifest"]["schema_version"] == "chunk_schema_v1"
    assert result["chunk_id_map"]["old_calo"] == "1705_00001v1::chunk_v1::0001"
    assert result["retrieval_smoke_summary"]["status"] == "PASS"
    assert (output_dir / "chunks_v1.jsonl").exists()
    assert (output_dir / "chunk_audit_v0_summary.json").exists()
    assert (output_dir / "chunk_audit_v0_details.jsonl").exists()
    assert (output_dir / "chunk_id_map_v0_to_v1.json").exists()
    assert (output_dir / "rag_manifest_v1.json").exists()
    assert (output_dir / "eval_questions_v0.jsonl").exists()
    assert (output_dir / "retrieval_smoke_summary_v1.json").exists()


def test_retrieval_smoke_test_is_modular() -> None:
    chunks_v1, _ = build_chunks_v1(
        chunks=[
            _chunk(
                "old_calo",
                "1705.00001v1",
                "CaloGAN uses a generative adversarial network for calorimeter shower simulation.",
            )
        ],
        papers=[_paper("1705.00001v1", "Fast calorimeter simulation")],
        equations=[],
    )

    summary = build_retrieval_smoke_summary(
        chunks_v1=chunks_v1,
        eval_questions=[
            {
                "question_id": "q_calo",
                "question": "calorimeter shower simulation",
                "question_type": "method",
                "expected_source_arxiv_ids": ["1705.00001v1"],
            }
        ],
        top_k=1,
    )

    assert summary["expected_source_hit_rate"] == 1.0
    assert summary["miss_question_ids"] == []
