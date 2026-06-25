"""Tests for the KamiKnows quality gate."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.quality.quality_gate import (
    discover_manual_review_summary_paths,
    evaluate_quality_gate,
    evaluate_quality_gate_from_files,
)
from scripts.run_quality_gate import main as quality_gate_main


PASS_MANIFEST = {
    "manifest_version": "dataset_manifest_v0",
    "name": "test_manifest",
    "status": "PASS",
    "files": [
        {
            "path": "manual_review_summary.json",
            "role": "manual_review_summary",
            "exists": True,
        }
    ],
    "missing_paths": [],
}

PASS_REVIEW_SUMMARY = {
    "review_summary_version": "manual_review_summary_v0",
    "source_path": "manual_quality_checklist.md",
    "total_records": 2,
    "status": "PASS",
    "outcome_counts": {
        "pass": 2,
        "revise": 0,
        "reject": 0,
        "unclear": 0,
    },
    "fully_checked_records": 2,
}

REVISE_REVIEW_SUMMARY = {
    **PASS_REVIEW_SUMMARY,
    "status": "REVIEW_REVISIONS_NEEDED",
    "outcome_counts": {
        "pass": 1,
        "revise": 1,
        "reject": 0,
        "unclear": 0,
    },
    "fully_checked_records": 1,
}

REJECT_REVIEW_SUMMARY = {
    **PASS_REVIEW_SUMMARY,
    "status": "REVIEW_REJECTED_RECORDS",
    "outcome_counts": {
        "pass": 1,
        "revise": 0,
        "reject": 1,
        "unclear": 0,
    },
}


def test_quality_gate_accepts_when_manifest_and_reviews_pass() -> None:
    report = evaluate_quality_gate(PASS_MANIFEST, [PASS_REVIEW_SUMMARY])

    assert report["decision"] == "ACCEPT"
    assert report["manual_review"]["total_reviewed_records"] == 2
    assert report["manual_review"]["outcome_counts"]["pass"] == 2


def test_quality_gate_revises_when_manual_review_missing() -> None:
    report = evaluate_quality_gate(PASS_MANIFEST, [])

    assert report["decision"] == "REVISE"
    assert "manual review is required" in report["reasons"][0]


def test_quality_gate_can_allow_unreviewed_manifest_only_accept() -> None:
    report = evaluate_quality_gate(
        PASS_MANIFEST,
        [],
        require_manual_review=False,
    )

    assert report["decision"] == "ACCEPT"


def test_quality_gate_revises_when_review_needs_revision() -> None:
    report = evaluate_quality_gate(PASS_MANIFEST, [REVISE_REVIEW_SUMMARY])

    assert report["decision"] == "REVISE"
    assert any("REVIEW_REVISIONS_NEEDED" in reason for reason in report["reasons"])


def test_quality_gate_rejects_when_review_has_rejected_records() -> None:
    report = evaluate_quality_gate(PASS_MANIFEST, [REJECT_REVIEW_SUMMARY])

    assert report["decision"] == "REJECT"
    assert any("rejected records" in reason for reason in report["reasons"])


def test_quality_gate_rejects_when_manifest_has_missing_paths() -> None:
    manifest = {
        **PASS_MANIFEST,
        "status": "WARN",
        "missing_paths": ["missing.jsonl"],
    }

    report = evaluate_quality_gate(manifest, [PASS_REVIEW_SUMMARY])

    assert report["decision"] == "REJECT"
    assert any("missing files" in reason for reason in report["reasons"])


def test_discover_manual_review_summary_paths_from_manifest() -> None:
    paths = discover_manual_review_summary_paths(PASS_MANIFEST)

    assert paths == [Path("manual_review_summary.json")]


def test_evaluate_quality_gate_from_files_discovers_review_summaries(tmp_path: Path) -> None:
    review_path = tmp_path / "manual_review_summary.json"
    manifest_path = tmp_path / "dataset_manifest.json"
    review_path.write_text(json.dumps(PASS_REVIEW_SUMMARY), encoding="utf-8")
    manifest = {
        **PASS_MANIFEST,
        "files": [
            {
                "path": str(review_path),
                "role": "manual_review_summary",
                "exists": True,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_quality_gate_from_files(manifest_path)

    assert report["decision"] == "ACCEPT"
    assert report["input_files"]["manual_review_summaries"] == [str(review_path)]


def test_run_quality_gate_cli_writes_report(tmp_path: Path) -> None:
    review_path = tmp_path / "manual_review_summary.json"
    manifest_path = tmp_path / "dataset_manifest.json"
    output_path = tmp_path / "quality_gate_report.json"
    review_path.write_text(json.dumps(PASS_REVIEW_SUMMARY), encoding="utf-8")
    manifest = {
        **PASS_MANIFEST,
        "files": [
            {
                "path": str(review_path),
                "role": "manual_review_summary",
                "exists": True,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = quality_gate_main(
        ["--manifest", str(manifest_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["decision"] == "ACCEPT"


def test_run_quality_gate_cli_strict_mode_fails_on_revise(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset_manifest.json"
    output_path = tmp_path / "quality_gate_report.json"
    manifest_path.write_text(json.dumps(PASS_MANIFEST), encoding="utf-8")

    exit_code = quality_gate_main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--fail-on-non-accept",
        ]
    )

    assert exit_code == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["decision"] == "REVISE"
