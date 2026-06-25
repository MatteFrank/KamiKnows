"""Tests for post-pilot analysis helpers and CLI."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from kamiknows.pilot.post_pilot_analysis import (
    build_post_pilot_analysis,
    build_post_pilot_analysis_from_manifest,
)
from scripts.summarize_hep_pilot_run import main as summarize_hep_pilot_main


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _valid_record() -> dict:
    return {
        "source": {
            "arxiv_id": "0000.00001v1",
            "title": "Pilot title",
            "authors": ["Ada Example"],
            "categories": ["hep-ex"],
            "published": "2026-01-01T00:00:00Z",
            "url": "https://arxiv.org/abs/0000.00001v1",
        },
        "extraction": {
            "title": "Pilot title",
            "field": "High Energy Physics",
            "main_claim": "A controlled claim.",
            "method": "A controlled method.",
            "limitations": "Limited validation.",
            "confidence": "medium",
        },
        "run": {
            "run_id": "run-1",
            "created_at": "2026-01-01T00:00:00Z",
            "backend": "ollama",
            "model": "qwen3:0.6b",
            "prompt_version": "abstract_to_json_v0",
            "prompt_template_sha256": "abc",
            "extraction_schema_version": "extraction_schema_v0",
        },
    }


def test_post_pilot_analysis_recommends_ready_when_gate_accepts(tmp_path: Path) -> None:
    metadata = _write_json(tmp_path / "pilot_metadata.json", {"items": []})
    jsonl = _write_jsonl(tmp_path / "pilot.jsonl", [_valid_record()])
    summary = _write_json(
        tmp_path / "pilot_summary.json",
        {"evaluation_status": "PASS", "total_records": 1},
    )
    review = _write_json(
        tmp_path / "pilot_manual_review_summary.json",
        {
            "status": "PASS",
            "total_records": 1,
            "fully_checked_records": 1,
            "outcome_counts": {"pass": 1, "revise": 0, "reject": 0, "unclear": 0},
        },
    )
    manifest = build_dataset_manifest(
        name="pilot",
        files=[
            (metadata, "metadata"),
            (jsonl, "jsonl_extractions"),
            (summary, "summary"),
            (review, "manual_review_summary"),
        ],
        backend="ollama",
        model="qwen3:0.6b",
    )
    gate = {
        "decision": "ACCEPT",
        "reasons": ["manifest and manual review summaries passed the quality gate"],
    }

    report = build_post_pilot_analysis(
        manifest=manifest,
        quality_gate_report=gate,
    )

    assert report["recommendation"] == "READY_FOR_NEXT_PILOT_CYCLE"
    assert report["formal_summary"]["evaluation_status_counts"] == {"PASS": 1}
    assert report["manual_review"]["outcome_counts"]["pass"] == 1
    assert report["artifacts"]["jsonl_record_count_from_manifest"] == 1


def test_post_pilot_analysis_recommends_run_gate_when_missing(tmp_path: Path) -> None:
    metadata = _write_json(tmp_path / "pilot_metadata.json", {"items": []})
    jsonl = _write_jsonl(tmp_path / "pilot.jsonl", [_valid_record()])
    manifest = build_dataset_manifest(
        name="pilot",
        files=[(metadata, "metadata"), (jsonl, "jsonl_extractions")],
    )

    report = build_post_pilot_analysis(manifest=manifest)

    assert report["recommendation"] == "RUN_QUALITY_GATE"


def test_post_pilot_analysis_from_manifest_auto_reads_gate_next_to_manifest(tmp_path: Path) -> None:
    jsonl = _write_jsonl(tmp_path / "pilot.jsonl", [_valid_record()])
    review = _write_json(
        tmp_path / "pilot_manual_review_summary.json",
        {
            "status": "PASS",
            "total_records": 1,
            "fully_checked_records": 1,
            "outcome_counts": {"pass": 1, "revise": 0, "reject": 0, "unclear": 0},
        },
    )
    manifest = build_dataset_manifest(
        name="pilot",
        files=[(jsonl, "jsonl_extractions"), (review, "manual_review_summary")],
    )
    manifest_path = write_dataset_manifest(manifest, tmp_path / "dataset_manifest.json")
    _write_json(tmp_path / "quality_gate_report.json", {"decision": "REVISE", "reasons": ["needs edits"]})

    report = build_post_pilot_analysis_from_manifest(manifest_path)

    assert report["quality_gate"]["decision"] == "REVISE"
    assert report["recommendation"] == "REVISE_BEFORE_SCALING"


def test_summarize_hep_pilot_run_cli_writes_report(tmp_path: Path, capsys) -> None:
    jsonl = _write_jsonl(tmp_path / "pilot.jsonl", [_valid_record()])
    manifest = build_dataset_manifest(
        name="pilot",
        files=[(jsonl, "jsonl_extractions")],
    )
    manifest_path = write_dataset_manifest(manifest, tmp_path / "dataset_manifest.json")
    _write_json(tmp_path / "quality_gate_report.json", {"decision": "ACCEPT", "reasons": ["ok"]})
    output = tmp_path / "post_pilot_analysis.json"

    exit_code = summarize_hep_pilot_main(
        ["--manifest", str(manifest_path), "--output", str(output)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "KamiKnows HEP post-pilot analysis" in captured.out
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["recommendation"] == "READY_FOR_NEXT_PILOT_CYCLE"
