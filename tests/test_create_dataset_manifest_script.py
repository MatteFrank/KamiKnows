"""Tests for scripts/create_dataset_manifest.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import create_dataset_manifest


def test_create_dataset_manifest_from_explicit_files(tmp_path: Path, capsys) -> None:
    metadata = tmp_path / "metadata.json"
    jsonl = tmp_path / "records.jsonl"
    summary = tmp_path / "summary.json"
    output = tmp_path / "dataset_manifest.json"

    metadata.write_text('[{"arxiv_id": "0000.00000v1"}]\n', encoding="utf-8")
    jsonl.write_text('{"record": 1}\n', encoding="utf-8")
    summary.write_text('{"evaluation_status": "PASS"}\n', encoding="utf-8")

    exit_code = create_dataset_manifest.main(
        [
            "--metadata-file",
            str(metadata),
            "--jsonl-file",
            str(jsonl),
            "--summary-file",
            str(summary),
            "--backend",
            "fake",
            "--model",
            "fake",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "KamiKnows dataset manifest written" in captured.out
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["run_context"]["backend"] == "fake"
    assert len(manifest["files"]) == 3


def test_create_dataset_manifest_from_mini_benchmark_dir(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "metadata_query_a.json").write_text("[]\n", encoding="utf-8")
    (benchmark_dir / "query_a_fake_fake.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (benchmark_dir / "query_a_fake_fake_summary.json").write_text(
        '{"evaluation_status": "PASS"}\n', encoding="utf-8"
    )
    (benchmark_dir / "mini_benchmark_report.json").write_text(
        json.dumps(
            {
                "backend": "fake",
                "model": "fake",
                "prompt": {
                    "prompt_version": "abstract_to_json_v0",
                    "prompt_template_sha256": "b" * 64,
                    "extraction_schema_version": "extraction_schema_v0",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = benchmark_dir / "dataset_manifest.json"

    exit_code = create_dataset_manifest.main(
        [
            "--from-mini-benchmark-dir",
            str(benchmark_dir),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    roles = [entry["role"] for entry in manifest["files"]]
    assert "metadata" in roles
    assert "jsonl_extractions" in roles
    assert "summary" in roles
    assert "benchmark_report" in roles


def test_create_dataset_manifest_registers_manual_review_files(tmp_path: Path) -> None:
    checklist = tmp_path / "query_a_manual_quality_checklist.md"
    review_summary = tmp_path / "query_a_manual_review_summary.json"
    output = tmp_path / "dataset_manifest.json"

    checklist.write_text("# Manual checklist\n", encoding="utf-8")
    review_summary.write_text(
        '{"review_summary_version": "manual_review_summary_v0", "status": "PASS"}\n',
        encoding="utf-8",
    )

    exit_code = create_dataset_manifest.main(
        [
            "--checklist-file",
            str(checklist),
            "--manual-review-summary-file",
            str(review_summary),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    roles = [entry["role"] for entry in manifest["files"]]
    assert "manual_quality_checklist" in roles
    assert "manual_review_summary" in roles


def test_create_dataset_manifest_from_mini_benchmark_dir_collects_manual_review_files(
    tmp_path: Path,
) -> None:
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "metadata_query_a.json").write_text("[]\n", encoding="utf-8")
    (benchmark_dir / "query_a_fake_fake.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (benchmark_dir / "query_a_fake_fake_summary.json").write_text(
        '{"evaluation_status": "PASS"}\n', encoding="utf-8"
    )
    (benchmark_dir / "query_a_manual_quality_checklist.md").write_text(
        "# Checklist\n", encoding="utf-8"
    )
    (benchmark_dir / "query_a_manual_review_summary.json").write_text(
        '{"review_summary_version": "manual_review_summary_v0", "status": "PASS"}\n',
        encoding="utf-8",
    )
    output = benchmark_dir / "dataset_manifest.json"

    exit_code = create_dataset_manifest.main(
        [
            "--from-mini-benchmark-dir",
            str(benchmark_dir),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    roles = [entry["role"] for entry in manifest["files"]]
    assert "summary" in roles
    assert "manual_quality_checklist" in roles
    assert "manual_review_summary" in roles
