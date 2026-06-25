"""Tests for kamiknows.dataset_manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kamiknows.dataset_manifest import (
    DatasetManifestError,
    build_dataset_manifest,
    build_file_entry,
    read_dataset_manifest,
    write_dataset_manifest,
)


def test_build_file_entry_for_jsonl_counts_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

    entry = build_file_entry(path, role="jsonl_extractions")

    assert entry["exists"] is True
    assert entry["role"] == "jsonl_extractions"
    assert entry["record_count"] == 2
    assert entry["json_top_level_type"] == "jsonl"
    assert len(entry["sha256"]) == 64


def test_build_file_entry_for_json_identifies_top_level_type(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps([{"a": 1}]), encoding="utf-8")

    entry = build_file_entry(path, role="metadata")

    assert entry["json_top_level_type"] == "list"
    assert entry["record_count"] is None


def test_build_dataset_manifest_and_round_trip(tmp_path: Path) -> None:
    jsonl = tmp_path / "records.jsonl"
    summary = tmp_path / "summary.json"
    output = tmp_path / "dataset_manifest.json"
    jsonl.write_text('{"record": 1}\n', encoding="utf-8")
    summary.write_text('{"evaluation_status": "PASS"}\n', encoding="utf-8")

    manifest = build_dataset_manifest(
        name="test_manifest",
        files=[(jsonl, "jsonl_extractions"), (summary, "summary")],
        backend="fake",
        model="fake",
        prompt_version="abstract_to_json_v0",
        prompt_template_sha256="a" * 64,
        extraction_schema_version="extraction_schema_v0",
        base_dir=tmp_path,
    )
    write_dataset_manifest(manifest, output)
    loaded = read_dataset_manifest(output)

    assert loaded["manifest_version"] == "dataset_manifest_v0"
    assert loaded["status"] == "PASS"
    assert loaded["run_context"]["backend"] == "fake"
    assert len(loaded["files"]) == 2
    assert loaded["files"][0]["relative_path"] == "records.jsonl"


def test_build_dataset_manifest_marks_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    manifest = build_dataset_manifest(
        name="missing_manifest",
        files=[(missing, "jsonl_extractions")],
    )

    assert manifest["status"] == "WARN"
    assert manifest["missing_paths"] == [str(missing)]


def test_build_dataset_manifest_rejects_empty_file_list() -> None:
    with pytest.raises(DatasetManifestError):
        build_dataset_manifest(name="bad", files=[])
