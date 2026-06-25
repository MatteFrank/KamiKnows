"""Tests for scripts/summarize_jsonl.py."""

from __future__ import annotations

from pathlib import Path

from kamiknows.storage.jsonl import append_jsonl_record
from scripts import summarize_jsonl


def _traceable_record(
    *,
    arxiv_id: str = "0000.00001v1",
    confidence: str = "medium",
    backend: str = "fake",
    model: str = "fake",
) -> dict[str, object]:
    return {
        "source": {
            "arxiv_id": arxiv_id,
            "title": "Fast calorimeter simulation for high energy physics",
            "authors": ["Ada Example"],
            "categories": ["hep-ex"],
            "published": "2026-01-01T00:00:00Z",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        },
        "extraction": {
            "title": "Fast calorimeter simulation for high energy physics",
            "field": "High Energy Physics / detector simulation",
            "main_claim": "A parameterized method can reduce simulation time.",
            "method": "Parameterized shower shapes.",
            "limitations": "Tutorial-only example.",
            "confidence": confidence,
        },
        "run": {
            "run_id": "abc123",
            "created_at": "2026-01-01T00:00:00Z",
            "backend": backend,
            "model": model,
            "prompt_version": "abstract_to_json_v0",
            "prompt_template_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "extraction_schema_version": "extraction_schema_v0",
        },
    }


def test_summarize_records_counts_confidence_backend_and_model() -> None:
    records = [
        _traceable_record(confidence="medium", backend="fake", model="fake"),
        _traceable_record(
            arxiv_id="0000.00002v1",
            confidence="low",
            backend="ollama",
            model="qwen3:0.6b",
        ),
    ]

    summary = summarize_jsonl.summarize_records(records)

    assert summary["total_records"] == 2
    assert summary["records_with_source"] == 2
    assert summary["confidence_counts"] == {"low": 1, "medium": 1}
    assert summary["backend_counts"] == {"fake": 1, "ollama": 1}
    assert summary["model_counts"] == {"fake": 1, "qwen3:0.6b": 1}
    assert summary["prompt_template_sha256_counts"] == {"0" * 64: 2}
    assert summary["extraction_schema_version_counts"] == {"extraction_schema_v0": 2}
    assert summary["missing_fields"] == {}
    assert summary["invalid_confidence_records"] == []


def test_summarize_records_reports_missing_source_when_required() -> None:
    record = _traceable_record()
    record.pop("source")

    summary = summarize_jsonl.summarize_records([record], require_source=True)

    assert summary["records_with_source"] == 0
    assert summary["missing_fields"]["source.arxiv_id"] == [1]
    assert summary["missing_fields"]["source.title"] == [1]


def test_summarize_records_can_allow_simple_records() -> None:
    record = _traceable_record()
    record.pop("source")

    summary = summarize_jsonl.summarize_records([record], require_source=False)

    assert summary["records_with_source"] == 0
    assert summary["missing_fields"] == {}


def test_summarize_records_reports_invalid_confidence_label() -> None:
    record = _traceable_record(confidence="certain")

    summary = summarize_jsonl.summarize_records([record])

    assert summary["confidence_counts"] == {"certain": 1}
    assert summary["invalid_confidence_records"] == [1]


def test_summarize_jsonl_main_prints_pass_for_valid_file(
    tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "records.jsonl"
    append_jsonl_record(_traceable_record(), output_path)

    exit_code = summarize_jsonl.main([str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Records: 1" in captured.out
    assert "Confidence labels:" in captured.out
    assert "medium: 1" in captured.out
    assert "Backends:" in captured.out
    assert "fake: 1" in captured.out
    assert "Missing fields: none" in captured.out
    assert "Evaluation status: PASS" in captured.out


def test_summarize_jsonl_main_can_fail_on_warnings(
    tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "records.jsonl"
    append_jsonl_record(_traceable_record(confidence="certain"), output_path)

    exit_code = summarize_jsonl.main([str(output_path), "--fail-on-warnings"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Invalid confidence labels: 1 record(s) [1]" in captured.out
    assert "Evaluation status: WARN" in captured.out


def test_summarize_records_includes_evaluation_status() -> None:
    pass_summary = summarize_jsonl.summarize_records([_traceable_record()])
    warn_summary = summarize_jsonl.summarize_records(
        [_traceable_record(confidence="certain")]
    )

    assert pass_summary["evaluation_status"] == "PASS"
    assert warn_summary["evaluation_status"] == "WARN"


def test_write_summary_json_creates_machine_readable_file(tmp_path: Path) -> None:
    summary = summarize_jsonl.summarize_records([_traceable_record()])
    output_path = tmp_path / "reports" / "summary.json"

    written_path = summarize_jsonl.write_summary_json(summary, output_path)

    assert written_path == output_path
    assert output_path.exists()
    assert '"evaluation_status": "PASS"' in output_path.read_text(encoding="utf-8")


def test_summarize_jsonl_main_writes_json_output(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "records.jsonl"
    json_output_path = tmp_path / "summary" / "batch_summary.json"
    append_jsonl_record(_traceable_record(), input_path)

    exit_code = summarize_jsonl.main(
        [str(input_path), "--json-output", str(json_output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Summary JSON written to:" in captured.out
    assert json_output_path.exists()
    assert '"total_records": 1' in json_output_path.read_text(encoding="utf-8")
    assert '"evaluation_status": "PASS"' in json_output_path.read_text(
        encoding="utf-8"
    )
