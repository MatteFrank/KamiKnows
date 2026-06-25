"""Tests for scripts/run_arxiv_extraction.py without live arXiv calls."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.run_metadata import DEFAULT_PROMPT_VERSION
from kamiknows.storage.jsonl import read_jsonl_records
from scripts import run_arxiv_extraction

SAMPLE_METADATA = {
    "arxiv_id": "2301.00001v1",
    "title": "Fast calorimeter simulation for high energy physics",
    "authors": ["Ada Example", "Bruno Example"],
    "abstract": (
        "We present a lightweight method for fast calorimeter response simulation "
        "in high energy physics. The method uses parameterized shower shapes."
    ),
    "categories": ["hep-ex", "physics.ins-det"],
    "published": "2023-01-01T00:00:00Z",
    "url": "https://arxiv.org/abs/2301.00001v1",
}


def test_run_arxiv_extraction_by_id_with_fake_backend(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "outputs" / "arxiv_extractions.jsonl"

    def fake_fetch(arxiv_id: str) -> dict:
        assert arxiv_id == "2301.00001"
        return SAMPLE_METADATA

    monkeypatch.setattr(run_arxiv_extraction, "fetch_arxiv_metadata_by_id", fake_fetch)

    exit_code = run_arxiv_extraction.main(
        ["--id", "2301.00001", "--backend", "fake", "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Selected arXiv paper: 2301.00001v1" in captured.out
    assert "Backend: fake" in captured.out

    records = read_jsonl_records(output_path)
    assert len(records) == 1
    record = records[0]
    assert record["source"]["arxiv_id"] == "2301.00001v1"
    assert record["source"]["url"] == "https://arxiv.org/abs/2301.00001v1"
    assert record["run"]["backend"] == "fake"
    assert record["run"]["model"] == "fake"
    assert record["run"]["prompt_version"] == DEFAULT_PROMPT_VERSION
    assert record["run"]["run_id"]
    assert record["run"]["created_at"].endswith("Z")
    assert record["extraction"]["title"] == SAMPLE_METADATA["title"]
    assert record["extraction"]["confidence"] == "medium"


def test_run_arxiv_extraction_by_query_selects_first_result(
    monkeypatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "outputs" / "query_extractions.jsonl"

    def fake_search(query: str, max_results: int) -> list[dict]:
        assert query == "cat:hep-ex AND calorimeter"
        assert max_results == 2
        second = {**SAMPLE_METADATA, "arxiv_id": "2301.00002v1"}
        return [SAMPLE_METADATA, second]

    monkeypatch.setattr(run_arxiv_extraction, "search_arxiv_metadata", fake_search)

    exit_code = run_arxiv_extraction.main(
        [
            "--query",
            "cat:hep-ex AND calorimeter",
            "--max-results",
            "2",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    records = read_jsonl_records(output_path)
    assert len(records) == 1
    assert records[0]["source"]["arxiv_id"] == "2301.00001v1"


def test_run_arxiv_extraction_from_offline_metadata_file(
    tmp_path: Path, capsys
) -> None:
    metadata_path = tmp_path / "arxiv_metadata_example.json"
    output_path = tmp_path / "outputs" / "offline_extractions.jsonl"
    metadata_path.write_text(json.dumps(SAMPLE_METADATA), encoding="utf-8")

    exit_code = run_arxiv_extraction.main(
        [
            "--metadata-file",
            str(metadata_path),
            "--backend",
            "fake",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Selected arXiv paper: 2301.00001v1" in captured.out
    assert "Backend: fake" in captured.out

    records = read_jsonl_records(output_path)
    assert len(records) == 1
    assert records[0]["source"]["title"] == SAMPLE_METADATA["title"]
    assert records[0]["run"]["backend"] == "fake"
    assert records[0]["run"]["model"] == "fake"
    assert records[0]["run"]["prompt_version"] == DEFAULT_PROMPT_VERSION


def test_load_arxiv_metadata_file_rejects_missing_required_fields(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "bad_metadata.json"
    metadata_path.write_text(json.dumps({"title": "Incomplete"}), encoding="utf-8")

    exit_code = run_arxiv_extraction.main(
        ["--metadata-file", str(metadata_path), "--output", str(tmp_path / "out.jsonl")]
    )

    assert exit_code == 1
