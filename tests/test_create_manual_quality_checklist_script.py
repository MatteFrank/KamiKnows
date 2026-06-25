"""Tests for scripts/create_manual_quality_checklist.py."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.storage.jsonl import append_jsonl_record
from scripts import create_manual_quality_checklist


def _record_without_abstract() -> dict:
    return {
        "source": {
            "arxiv_id": "0000.00001v1",
            "title": "Fast calorimeter simulation for high energy physics",
            "authors": ["Ada Example"],
            "categories": ["hep-ex"],
            "published": "2026-01-01T00:00:00Z",
            "url": "https://arxiv.org/abs/0000.00001v1",
        },
        "extraction": {
            "title": "Fast calorimeter simulation for high energy physics",
            "field": "High Energy Physics / detector simulation",
            "main_claim": "A parameterized method can support fast simulation.",
            "method": "Parameterized simulation.",
            "limitations": "Validation limits remain.",
            "confidence": "medium",
        },
        "run": {
            "run_id": "test-run",
            "created_at": "2026-01-01T00:00:00Z",
            "backend": "ollama",
            "model": "qwen3:0.6b",
            "prompt_version": "abstract_to_json_v0",
            "prompt_template_sha256": "0" * 64,
            "extraction_schema_version": "extraction_schema_v0",
        },
    }


def test_create_manual_quality_checklist_cli_writes_markdown(
    tmp_path: Path, capsys
) -> None:
    jsonl_path = tmp_path / "records.jsonl"
    metadata_path = tmp_path / "metadata.json"
    output_path = tmp_path / "checklist.md"

    append_jsonl_record(_record_without_abstract(), jsonl_path)
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "arxiv_id": "0000.00001v1",
                    "title": "Fast calorimeter simulation for high energy physics",
                    "authors": ["Ada Example"],
                    "abstract": "Recovered abstract for manual review.",
                    "categories": ["hep-ex"],
                    "published": "2026-01-01T00:00:00Z",
                    "url": "https://arxiv.org/abs/0000.00001v1",
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = create_manual_quality_checklist.main(
        [
            str(jsonl_path),
            "--metadata-list",
            str(metadata_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Manual quality checklist created" in captured.out
    text = output_path.read_text(encoding="utf-8")
    assert "Recovered abstract for manual review." in text
    assert "Review outcome" in text


def test_create_manual_quality_checklist_cli_rejects_bad_limit(
    tmp_path: Path, capsys
) -> None:
    jsonl_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "checklist.md"
    append_jsonl_record(_record_without_abstract(), jsonl_path)

    exit_code = create_manual_quality_checklist.main(
        [str(jsonl_path), "--limit", "0", "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "limit must be >= 1" in captured.err
    assert not output_path.exists()
