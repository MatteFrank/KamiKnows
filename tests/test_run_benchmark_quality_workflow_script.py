"""Tests for scripts/run_benchmark_quality_workflow.py."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.ingestion.arxiv_metadata import write_arxiv_metadata_list_file
from scripts import run_benchmark_quality_workflow

SAMPLE_A = {
    "arxiv_id": "0000.00101v1",
    "title": "Calorimeter workflow record",
    "authors": ["Ada Example"],
    "abstract": "We present a calorimeter extraction workflow example.",
    "categories": ["hep-ex"],
    "published": "2026-01-01T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00101v1",
}

SAMPLE_B = {
    "arxiv_id": "0000.00201v1",
    "title": "Higgs workflow record",
    "authors": ["Bruno Example"],
    "abstract": "We present a Higgs extraction workflow example.",
    "categories": ["hep-ex"],
    "published": "2026-01-02T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00201v1",
}


def test_run_benchmark_quality_workflow_from_local_metadata(tmp_path: Path, capsys) -> None:
    metadata_a = tmp_path / "metadata_a.json"
    metadata_b = tmp_path / "metadata_b.json"
    output_dir = tmp_path / "workflow"
    write_arxiv_metadata_list_file([SAMPLE_A], metadata_a)
    write_arxiv_metadata_list_file([SAMPLE_B], metadata_b)

    exit_code = run_benchmark_quality_workflow.main(
        [
            "--metadata-a",
            str(metadata_a),
            "--metadata-b",
            str(metadata_b),
            "--backend",
            "fake",
            "--output-dir",
            str(output_dir),
            "--review-limit",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "KamiKnows benchmark quality workflow completed" in captured.out
    assert "Formal status: PASS" in captured.out

    workflow_report_path = output_dir / "benchmark_quality_workflow_report.json"
    report = json.loads(workflow_report_path.read_text(encoding="utf-8"))
    assert report["workflow_type"] == "mini_benchmark_plus_manual_quality_checklist"
    assert report["formal_status"] == "PASS"
    assert report["review_limit"] == 1

    checklist_a = output_dir / "query_a_manual_quality_checklist.md"
    checklist_b = output_dir / "query_b_manual_quality_checklist.md"
    assert checklist_a.exists()
    assert checklist_b.exists()

    manifest_path = output_dir / "dataset_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = [entry["role"] for entry in manifest["files"]]
    assert "manual_quality_checklist" in roles
    assert "workflow_report" in roles

    text_a = checklist_a.read_text(encoding="utf-8")
    assert "Calorimeter workflow record" in text_a
    assert "main_claim" in text_a
    assert "Review outcome" in text_a


def test_build_mini_benchmark_argv_includes_metadata_paths(tmp_path: Path) -> None:
    args = run_benchmark_quality_workflow.parse_args(
        [
            "--metadata-a",
            str(tmp_path / "a.json"),
            "--metadata-b",
            str(tmp_path / "b.json"),
            "--backend",
            "fake",
            "--output-dir",
            str(tmp_path / "out"),
            "--no-manifest",
        ]
    )

    argv = run_benchmark_quality_workflow.build_mini_benchmark_argv(args)
    assert "--metadata-a" in argv
    assert str(tmp_path / "a.json") in argv
    assert "--metadata-b" in argv
    assert str(tmp_path / "b.json") in argv
    assert "--no-manifest" in argv


def test_run_benchmark_quality_workflow_rejects_bad_review_limit(tmp_path: Path, capsys) -> None:
    exit_code = run_benchmark_quality_workflow.main(
        ["--backend", "fake", "--review-limit", "0", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--review-limit must be >= 1" in captured.err
