"""Tests for minimal JSONL storage helpers."""

from __future__ import annotations

import pytest

from kamiknows.extraction.abstract_to_json import ExtractionError
from kamiknows.storage.jsonl import (
    JsonlStorageError,
    append_extraction_jsonl,
    append_jsonl_record,
    read_jsonl_records,
)


VALID_EXTRACTION = {
    "title": "Fast calorimeter simulation for HEP",
    "field": "High Energy Physics / detector simulation",
    "main_claim": "A parameterized method can reduce simulation time within validated regions.",
    "method": "Parameterized shower shapes calibrated on reference samples.",
    "limitations": "Requires further validation on full detector geometries.",
    "confidence": "medium",
}


def test_append_jsonl_record_writes_one_line(tmp_path) -> None:
    output_path = tmp_path / "extractions.jsonl"

    saved_path = append_jsonl_record({"a": 1, "b": "test"}, output_path)

    assert saved_path == output_path
    assert output_path.read_text(encoding="utf-8").count("\n") == 1
    assert read_jsonl_records(output_path) == [{"a": 1, "b": "test"}]


def test_append_extraction_jsonl_validates_before_writing(tmp_path) -> None:
    output_path = tmp_path / "outputs" / "extractions.jsonl"

    append_extraction_jsonl(VALID_EXTRACTION, output_path)
    records = read_jsonl_records(output_path)

    assert len(records) == 1
    assert records[0]["title"] == "Fast calorimeter simulation for HEP"
    assert records[0]["confidence"] == "medium"


def test_append_extraction_jsonl_rejects_invalid_extraction(tmp_path) -> None:
    output_path = tmp_path / "outputs" / "extractions.jsonl"
    invalid = dict(VALID_EXTRACTION)
    invalid["confidence"] = "certain"

    with pytest.raises(ExtractionError):
        append_extraction_jsonl(invalid, output_path)

    assert not output_path.exists()


def test_append_jsonl_record_rejects_non_serializable_record(tmp_path) -> None:
    output_path = tmp_path / "bad.jsonl"

    with pytest.raises(JsonlStorageError):
        append_jsonl_record({"bad": object()}, output_path)
