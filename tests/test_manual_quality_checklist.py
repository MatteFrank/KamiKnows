"""Tests for manual quality checklist helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from kamiknows.quality.manual_checklist import (
    ManualChecklistError,
    build_abstract_index,
    build_manual_quality_checklist,
)


def _record(include_abstract: bool = True) -> dict:
    source = {
        "arxiv_id": "0000.00001v1",
        "title": "Fast calorimeter simulation for high energy physics",
        "authors": ["Ada Example"],
        "categories": ["hep-ex"],
        "published": "2026-01-01T00:00:00Z",
        "url": "https://arxiv.org/abs/0000.00001v1",
    }
    if include_abstract:
        source["abstract"] = "We present a parameterized calorimeter simulation method and discuss validation limits."

    return {
        "source": source,
        "extraction": {
            "title": "Fast calorimeter simulation for high energy physics",
            "field": "High Energy Physics / detector simulation",
            "main_claim": "A parameterized method can support fast calorimeter simulation.",
            "method": "Parameterized simulation calibrated on examples.",
            "limitations": "Validation limits remain.",
            "confidence": "medium",
        },
        "run": {
            "backend": "ollama",
            "model": "qwen3:0.6b",
            "prompt_version": "abstract_to_json_v0",
        },
    }


def test_build_manual_quality_checklist_includes_review_items() -> None:
    markdown = build_manual_quality_checklist(
        records=[_record()],
        source_path=Path("outputs/test.jsonl"),
        limit=1,
    )

    assert "# KamiKnows manual quality checklist" in markdown
    assert "Records selected: 1 / 1" in markdown
    assert "Fast calorimeter simulation" in markdown
    assert "Abstract used for review" in markdown
    assert "`main_claim`" in markdown
    assert "`method`" in markdown
    assert "`limitations`" in markdown
    assert "Review outcome: `pass | revise | reject | unclear`" in markdown


def test_build_manual_quality_checklist_can_use_metadata_abstract_index() -> None:
    record = _record(include_abstract=False)
    abstract_index = build_abstract_index(
        [
            {
                "arxiv_id": "0000.00001v1",
                "abstract": "Recovered abstract from metadata file.",
            }
        ]
    )

    markdown = build_manual_quality_checklist(
        records=[record],
        source_path="records.jsonl",
        limit=1,
        abstract_index=abstract_index,
    )

    assert "Recovered abstract from metadata file." in markdown
    assert "abstract missing" not in markdown


def test_build_manual_quality_checklist_rejects_empty_records() -> None:
    with pytest.raises(ManualChecklistError, match="no records"):
        build_manual_quality_checklist(records=[], source_path="empty.jsonl")


def test_build_manual_quality_checklist_rejects_bad_limit() -> None:
    with pytest.raises(ManualChecklistError, match="limit must be >= 1"):
        build_manual_quality_checklist(records=[_record()], source_path="x.jsonl", limit=0)
