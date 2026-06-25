"""Tests for scripts/run_batch_arxiv_extraction.py."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.run_metadata import DEFAULT_PROMPT_VERSION
from kamiknows.models.fake import FakeExtractionModel
from kamiknows.storage.jsonl import read_jsonl_records
from scripts import run_batch_arxiv_extraction

SAMPLE_METADATA_1 = {
    "arxiv_id": "0000.00001v1",
    "title": "Fast calorimeter simulation for high energy physics",
    "authors": ["Ada Example"],
    "abstract": "We present a small batch tutorial abstract for detector simulation.",
    "categories": ["hep-ex"],
    "published": "2026-01-01T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00001v1",
}

SAMPLE_METADATA_2 = {
    "arxiv_id": "0000.00002v1",
    "title": "Graph-based reconstruction study for particle detector events",
    "authors": ["Bruno Example"],
    "abstract": "We present a small batch tutorial abstract for event reconstruction.",
    "categories": ["hep-ex", "cs.LG"],
    "published": "2026-01-02T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00002v1",
}


def _write_metadata_batch(path: Path) -> None:
    path.write_text(json.dumps([SAMPLE_METADATA_1, SAMPLE_METADATA_2]), encoding="utf-8")


def test_run_batch_arxiv_extraction_writes_multiple_records(
    tmp_path: Path, capsys
) -> None:
    metadata_path = tmp_path / "batch.json"
    output_path = tmp_path / "outputs" / "batch.jsonl"
    _write_metadata_batch(metadata_path)

    exit_code = run_batch_arxiv_extraction.main(
        ["--metadata-list", str(metadata_path), "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Processed records: 2" in captured.out
    assert "Backend: fake" in captured.out

    records = read_jsonl_records(output_path)
    assert len(records) == 2
    assert records[0]["source"]["arxiv_id"] == "0000.00001v1"
    assert records[0]["source"]["abstract"] == SAMPLE_METADATA_1["abstract"]
    assert records[1]["source"]["arxiv_id"] == "0000.00002v1"
    assert records[0]["run"]["backend"] == "fake"
    assert records[0]["run"]["model"] == "fake"
    assert records[0]["run"]["prompt_version"] == DEFAULT_PROMPT_VERSION
    assert records[0]["extraction"]["title"] == SAMPLE_METADATA_1["title"]
    assert records[1]["extraction"]["title"] == SAMPLE_METADATA_2["title"]


def test_run_batch_arxiv_extraction_respects_limit(tmp_path: Path) -> None:
    metadata_path = tmp_path / "batch.json"
    output_path = tmp_path / "outputs" / "batch.jsonl"
    _write_metadata_batch(metadata_path)

    exit_code = run_batch_arxiv_extraction.main(
        [
            "--metadata-list",
            str(metadata_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    records = read_jsonl_records(output_path)
    assert len(records) == 1
    assert records[0]["source"]["arxiv_id"] == "0000.00001v1"


def test_run_batch_arxiv_extraction_rejects_bad_limit(
    tmp_path: Path, capsys
) -> None:
    metadata_path = tmp_path / "batch.json"
    output_path = tmp_path / "outputs" / "batch.jsonl"
    _write_metadata_batch(metadata_path)

    exit_code = run_batch_arxiv_extraction.main(
        [
            "--metadata-list",
            str(metadata_path),
            "--output",
            str(output_path),
            "--limit",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--limit must be >= 1" in captured.err
    assert not output_path.exists()


def test_run_batch_arxiv_extraction_rejects_non_list_file(
    tmp_path: Path, capsys
) -> None:
    metadata_path = tmp_path / "bad_batch.json"
    output_path = tmp_path / "outputs" / "batch.jsonl"
    metadata_path.write_text(json.dumps(SAMPLE_METADATA_1), encoding="utf-8")

    exit_code = run_batch_arxiv_extraction.main(
        ["--metadata-list", str(metadata_path), "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "must contain a JSON list" in captured.err
    assert not output_path.exists()


def test_run_batch_arxiv_extraction_remote_query_processes_all_results(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "outputs" / "remote_query.jsonl"

    def fake_search(query: str, *, max_results: int, timeout_seconds: int) -> list[dict]:
        assert query == "cat:hep-ex AND calorimeter"
        assert max_results == 2
        assert timeout_seconds == 7
        return [SAMPLE_METADATA_1, SAMPLE_METADATA_2]

    monkeypatch.setattr(run_batch_arxiv_extraction, "search_arxiv_metadata", fake_search)

    exit_code = run_batch_arxiv_extraction.main(
        [
            "--query",
            "cat:hep-ex AND calorimeter",
            "--max-results",
            "2",
            "--timeout-seconds",
            "7",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Batch source: remote arXiv query: cat:hep-ex AND calorimeter" in captured.out
    assert "Processed records: 2" in captured.out

    records = read_jsonl_records(output_path)
    assert len(records) == 2
    assert records[0]["source"]["arxiv_id"] == "0000.00001v1"
    assert records[1]["source"]["arxiv_id"] == "0000.00002v1"


def test_run_batch_arxiv_extraction_remote_ids_processes_each_id(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "outputs" / "remote_ids.jsonl"
    seen_ids: list[str] = []

    def fake_fetch(arxiv_id: str, *, timeout_seconds: int) -> dict:
        assert timeout_seconds == 9
        seen_ids.append(arxiv_id)
        if arxiv_id == "0000.00001v1":
            return SAMPLE_METADATA_1
        if arxiv_id == "0000.00002v1":
            return SAMPLE_METADATA_2
        raise AssertionError(f"unexpected id: {arxiv_id}")

    monkeypatch.setattr(run_batch_arxiv_extraction, "fetch_arxiv_metadata_by_id", fake_fetch)

    exit_code = run_batch_arxiv_extraction.main(
        [
            "--ids",
            "0000.00001v1",
            "0000.00002v1",
            "--timeout-seconds",
            "9",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen_ids == ["0000.00001v1", "0000.00002v1"]
    assert "Batch source: remote arXiv IDs: 2" in captured.out

    records = read_jsonl_records(output_path)
    assert len(records) == 2
    assert records[0]["run"]["backend"] == "fake"
    assert records[1]["source"]["arxiv_id"] == "0000.00002v1"


def test_run_batch_arxiv_extraction_remote_query_respects_limit_after_fetch(
    monkeypatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "outputs" / "remote_query_limit.jsonl"

    def fake_search(query: str, *, max_results: int, timeout_seconds: int) -> list[dict]:
        return [SAMPLE_METADATA_1, SAMPLE_METADATA_2]

    monkeypatch.setattr(run_batch_arxiv_extraction, "search_arxiv_metadata", fake_search)

    exit_code = run_batch_arxiv_extraction.main(
        [
            "--query",
            "cat:hep-ex AND calorimeter",
            "--max-results",
            "2",
            "--limit",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    records = read_jsonl_records(output_path)
    assert len(records) == 1
    assert records[0]["source"]["arxiv_id"] == "0000.00001v1"


def test_run_batch_arxiv_extraction_accepts_ollama_backend_when_model_is_stubbed(
    monkeypatch, tmp_path: Path
) -> None:
    metadata_path = tmp_path / "batch.json"
    output_path = tmp_path / "outputs" / "batch_ollama.jsonl"
    _write_metadata_batch(metadata_path)
    calls: list[tuple[str, str, str, str]] = []

    def fake_build_model_for_record(
        *, backend: str, title: str, model_name: str, base_url: str
    ) -> FakeExtractionModel:
        calls.append((backend, title, model_name, base_url))
        return FakeExtractionModel(title=title)

    monkeypatch.setattr(
        run_batch_arxiv_extraction,
        "build_model_for_record",
        fake_build_model_for_record,
    )

    exit_code = run_batch_arxiv_extraction.main(
        [
            "--metadata-list",
            str(metadata_path),
            "--backend",
            "ollama",
            "--model",
            "qwen3:0.6b",
            "--base-url",
            "http://localhost:11434",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0][0] == "ollama"
    assert calls[0][2] == "qwen3:0.6b"
    records = read_jsonl_records(output_path)
    assert records[0]["run"]["backend"] == "ollama"
    assert records[0]["run"]["model"] == "qwen3:0.6b"


def test_run_batch_arxiv_extraction_rejects_bad_max_results(
    tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "outputs" / "bad_max_results.jsonl"

    exit_code = run_batch_arxiv_extraction.main(
        [
            "--query",
            "cat:hep-ex AND calorimeter",
            "--max-results",
            "0",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--max-results must be >= 1" in captured.err
    assert not output_path.exists()


def test_run_batch_arxiv_extraction_remote_ids_file_processes_each_id(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text(
        "# selected ids\n0000.00001v1\nhttps://arxiv.org/abs/0000.00002v1\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "outputs" / "remote_ids_file.jsonl"
    seen_ids: list[str] = []

    def fake_fetch(arxiv_id: str, *, timeout_seconds: int) -> dict:
        seen_ids.append(arxiv_id)
        if arxiv_id == "0000.00001v1":
            return SAMPLE_METADATA_1
        if arxiv_id == "https://arxiv.org/abs/0000.00002v1":
            return SAMPLE_METADATA_2
        raise AssertionError(f"unexpected id: {arxiv_id}")

    monkeypatch.setattr(run_batch_arxiv_extraction, "fetch_arxiv_metadata_by_id", fake_fetch)

    exit_code = run_batch_arxiv_extraction.main(
        [
            "--ids-file",
            str(ids_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen_ids == ["0000.00001v1", "https://arxiv.org/abs/0000.00002v1"]
    assert "remote arXiv IDs file" in captured.out

    records = read_jsonl_records(output_path)
    assert len(records) == 2
    assert records[1]["source"]["arxiv_id"] == "0000.00002v1"
