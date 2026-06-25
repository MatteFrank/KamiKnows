"""Tests for scripts/smoke_test_offline.py."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.run_metadata import DEFAULT_PROMPT_VERSION
from kamiknows.storage.jsonl import read_jsonl_records
from scripts import smoke_test_offline

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


def write_metadata(path: Path) -> None:
    path.write_text(json.dumps(SAMPLE_METADATA), encoding="utf-8")


def test_smoke_test_offline_creates_traceable_jsonl(
    tmp_path: Path, capsys
) -> None:
    metadata_path = tmp_path / "arxiv_metadata_example.json"
    output_path = tmp_path / "outputs" / "smoke_arxiv_extractions.jsonl"
    write_metadata(metadata_path)

    exit_code = smoke_test_offline.main(
        ["--metadata-file", str(metadata_path), "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "SMOKE TEST PASSED" in captured.out
    assert f"Output file: {output_path}" in captured.out

    records = read_jsonl_records(output_path)
    assert len(records) == 1
    record = records[0]
    assert record["source"]["arxiv_id"] == SAMPLE_METADATA["arxiv_id"]
    assert record["extraction"]["title"] == SAMPLE_METADATA["title"]
    assert record["run"]["backend"] == "fake"
    assert record["run"]["model"] == "fake"
    assert record["run"]["prompt_version"] == DEFAULT_PROMPT_VERSION
    assert record["run"]["run_id"]
    assert record["run"]["created_at"].endswith("Z")


def test_smoke_test_offline_replaces_output_by_default(tmp_path: Path) -> None:
    metadata_path = tmp_path / "arxiv_metadata_example.json"
    output_path = tmp_path / "outputs" / "smoke_arxiv_extractions.jsonl"
    write_metadata(metadata_path)
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    exit_code = smoke_test_offline.main(
        ["--metadata-file", str(metadata_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    records = read_jsonl_records(output_path)
    assert len(records) == 1
    assert "old" not in records[0]


def test_smoke_test_offline_append_keeps_existing_records(tmp_path: Path) -> None:
    metadata_path = tmp_path / "arxiv_metadata_example.json"
    output_path = tmp_path / "outputs" / "smoke_arxiv_extractions.jsonl"
    write_metadata(metadata_path)
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    exit_code = smoke_test_offline.main(
        [
            "--metadata-file",
            str(metadata_path),
            "--output",
            str(output_path),
            "--append",
        ]
    )

    assert exit_code == 0
    records = read_jsonl_records(output_path)
    assert len(records) == 2
    assert records[0] == {"old": True}
    assert records[1]["run"]["backend"] == "fake"
    assert records[1]["run"]["model"] == "fake"
    assert records[1]["run"]["prompt_version"] == DEFAULT_PROMPT_VERSION


def test_smoke_test_offline_missing_metadata_returns_error(tmp_path: Path) -> None:
    output_path = tmp_path / "outputs" / "smoke_arxiv_extractions.jsonl"

    exit_code = smoke_test_offline.main(
        ["--metadata-file", str(tmp_path / "missing.json"), "--output", str(output_path)]
    )

    assert exit_code == 1
    assert not output_path.exists()
