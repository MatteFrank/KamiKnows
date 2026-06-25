"""Tests for parsing completed manual quality review checklists."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.quality.manual_review import (
    ManualReviewError,
    parse_manual_quality_checklist,
    summarize_manual_quality_checklist_file,
    summarize_manual_review_records,
)
from scripts.summarize_manual_quality_review import main as summarize_review_main


COMPLETED_CHECKLIST = """# KamiKnows manual quality checklist

## Record 1: Example calorimeter paper

- arXiv ID: `0000.00001v1`
- URL: https://arxiv.org/abs/0000.00001v1
- backend/model: `ollama` / `qwen3:0.6b`
- prompt version: `abstract_to_json_v0`
- extraction confidence: `medium`

### Abstract used for review

Example abstract.

### Extracted fields to check

- `main_claim`: A supported claim.
- `method`: A supported method.
- `limitations`: A supported limitation.

### Manual checklist

- [x] `main_claim` is directly supported by the abstract.
- [x] `method` is directly supported by the abstract.
- [x] `limitations` are directly supported by the abstract.
- [x] No unsupported scientific claim was introduced.
- [x] The confidence label is plausible for this extraction.
- [x] Notes: faithful abstract-level extraction.

Review outcome: pass

## Record 2: Example Higgs paper

- arXiv ID: `0000.00002v1`
- URL: https://arxiv.org/abs/0000.00002v1
- backend/model: `ollama` / `qwen3:0.6b`
- prompt version: `abstract_to_json_v0`
- extraction confidence: `high`

### Abstract used for review

Example abstract.

### Extracted fields to check

- `main_claim`: An over-strong claim.
- `method`: A supported method.
- `limitations`: Missing limitation.

### Manual checklist

- [ ] `main_claim` is directly supported by the abstract.
- [x] `method` is directly supported by the abstract.
- [ ] `limitations` are directly supported by the abstract.
- [ ] No unsupported scientific claim was introduced.
- [ ] The confidence label is plausible for this extraction.
- [x] Notes: main claim is too strong.

Review outcome: revise
"""


def test_parse_manual_quality_checklist_extracts_records() -> None:
    records = parse_manual_quality_checklist(COMPLETED_CHECKLIST)

    assert len(records) == 2
    assert records[0].title == "Example calorimeter paper"
    assert records[0].arxiv_id == "0000.00001v1"
    assert records[0].checks["main_claim_supported"] is True
    assert records[0].outcome == "pass"
    assert records[0].notes == "faithful abstract-level extraction."

    assert records[1].checks["main_claim_supported"] is False
    assert records[1].checks["method_supported"] is True
    assert records[1].outcome == "revise"


def test_summarize_manual_review_records_status_revisions_needed() -> None:
    records = parse_manual_quality_checklist(COMPLETED_CHECKLIST)
    summary = summarize_manual_review_records(records, source_path="checklist.md")

    assert summary["total_records"] == 2
    assert summary["status"] == "REVIEW_REVISIONS_NEEDED"
    assert summary["outcome_counts"]["pass"] == 1
    assert summary["outcome_counts"]["revise"] == 1
    assert summary["check_pass_counts"]["method_supported"] == 2
    assert summary["fully_checked_records"] == 1


def test_summarize_manual_quality_checklist_file(tmp_path: Path) -> None:
    checklist_path = tmp_path / "manual_quality_checklist.md"
    checklist_path.write_text(COMPLETED_CHECKLIST, encoding="utf-8")

    summary = summarize_manual_quality_checklist_file(checklist_path)

    assert summary["source_path"] == str(checklist_path)
    assert summary["records"][0]["outcome"] == "pass"


def test_parse_empty_checklist_raises_error() -> None:
    try:
        parse_manual_quality_checklist("")
    except ManualReviewError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ManualReviewError")


def test_summarize_manual_quality_review_cli_writes_json(tmp_path: Path) -> None:
    checklist_path = tmp_path / "manual_quality_checklist.md"
    output_path = tmp_path / "manual_quality_summary.json"
    checklist_path.write_text(COMPLETED_CHECKLIST, encoding="utf-8")

    exit_code = summarize_review_main(
        [str(checklist_path), "--json-output", str(output_path)]
    )

    assert exit_code == 0
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["status"] == "REVIEW_REVISIONS_NEEDED"
    assert summary["outcome_counts"]["revise"] == 1


def test_summarize_manual_quality_review_cli_strict_mode_fails(tmp_path: Path) -> None:
    checklist_path = tmp_path / "manual_quality_checklist.md"
    output_path = tmp_path / "manual_quality_summary.json"
    checklist_path.write_text(COMPLETED_CHECKLIST, encoding="utf-8")

    exit_code = summarize_review_main(
        [
            str(checklist_path),
            "--json-output",
            str(output_path),
            "--fail-on-non-pass",
        ]
    )

    assert exit_code == 1
    assert output_path.exists()
