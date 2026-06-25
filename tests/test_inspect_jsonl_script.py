"""Tests for scripts/inspect_jsonl.py."""

from __future__ import annotations

from pathlib import Path

from kamiknows.storage.jsonl import append_jsonl_record
from scripts import inspect_jsonl


def _traceable_record(arxiv_id: str, title: str, backend: str = "fake") -> dict:
    return {
        "source": {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": ["Ada Example"],
            "categories": ["hep-ex"],
            "published": "2026-01-01T00:00:00Z",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        },
        "extraction": {
            "title": title,
            "field": "High Energy Physics / detector simulation",
            "main_claim": "A small tutorial claim is extracted from the abstract.",
            "method": "Fake deterministic extraction.",
            "limitations": "Not scientifically meaningful.",
            "confidence": "medium",
        },
        "run": {
            "run_id": "test-run-id",
            "created_at": "2026-01-01T00:00:00Z",
            "backend": backend,
            "model": backend,
            "prompt_version": "abstract_to_json_v0",
            "prompt_template_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "extraction_schema_version": "extraction_schema_v0",
        },
    }


def test_inspect_jsonl_prints_traceable_summary(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "records.jsonl"
    append_jsonl_record(
        _traceable_record("0000.00001v1", "First HEP tutorial paper"), output_path
    )

    exit_code = inspect_jsonl.main([str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "JSONL file:" in captured.out
    assert "Records: 1" in captured.out
    assert "First HEP tutorial paper" in captured.out
    assert "source: 0000.00001v1" in captured.out
    assert "backend/model: fake / fake" in captured.out
    assert "main_claim:" in captured.out


def test_inspect_jsonl_limit_hides_extra_records(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "records.jsonl"
    append_jsonl_record(_traceable_record("0000.00001v1", "First paper"), output_path)
    append_jsonl_record(_traceable_record("0000.00002v1", "Second paper"), output_path)

    exit_code = inspect_jsonl.main([str(output_path), "--limit", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Records: 2" in captured.out
    assert "Showing: 1" in captured.out
    assert "First paper" in captured.out
    assert "Second paper" not in captured.out
    assert "1 more record(s) not shown" in captured.out


def test_inspect_jsonl_rejects_non_positive_limit(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "records.jsonl"
    append_jsonl_record(_traceable_record("0000.00001v1", "First paper"), output_path)

    exit_code = inspect_jsonl.main([str(output_path), "--limit", "0"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--limit must be >= 1" in captured.err
