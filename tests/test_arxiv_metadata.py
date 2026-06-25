"""Tests for arXiv-style metadata validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kamiknows.ingestion.arxiv_metadata import (
    ArxivMetadataError,
    load_arxiv_id_list_file,
    load_arxiv_metadata_file,
    load_arxiv_metadata_list_file,
    validate_arxiv_metadata,
)

VALID_METADATA = {
    "arxiv_id": "2301.00001v1",
    "title": "Fast calorimeter simulation for high energy physics",
    "authors": ["Ada Example", "Bruno Example"],
    "abstract": "We present a small tutorial metadata record for KamiKnows.",
    "categories": ["hep-ex", "physics.ins-det"],
    "published": "2023-01-01T00:00:00Z",
    "url": "https://arxiv.org/abs/2301.00001v1",
}


def test_validate_arxiv_metadata_accepts_valid_record() -> None:
    record = {**VALID_METADATA, "note": "extra fields are allowed"}

    validated = validate_arxiv_metadata(record)

    assert validated is record
    assert validated["title"] == VALID_METADATA["title"]
    assert validated["note"] == "extra fields are allowed"


@pytest.mark.parametrize(
    "missing_field",
    ["arxiv_id", "title", "authors", "abstract", "categories", "published", "url"],
)
def test_validate_arxiv_metadata_rejects_missing_required_field(
    missing_field: str,
) -> None:
    record = dict(VALID_METADATA)
    record.pop(missing_field)

    with pytest.raises(ArxivMetadataError, match=missing_field):
        validate_arxiv_metadata(record)


@pytest.mark.parametrize("field", ["title", "abstract", "arxiv_id", "url"])
def test_validate_arxiv_metadata_rejects_empty_core_string(field: str) -> None:
    record = dict(VALID_METADATA)
    record[field] = "   "

    with pytest.raises(ArxivMetadataError, match=field):
        validate_arxiv_metadata(record)


def test_validate_arxiv_metadata_rejects_wrong_list_type() -> None:
    record = dict(VALID_METADATA)
    record["authors"] = "Ada Example"

    with pytest.raises(ArxivMetadataError, match="authors"):
        validate_arxiv_metadata(record)


def test_validate_arxiv_metadata_rejects_non_string_list_items() -> None:
    record = dict(VALID_METADATA)
    record["categories"] = ["hep-ex", 123]

    with pytest.raises(ArxivMetadataError, match="categories"):
        validate_arxiv_metadata(record)


def test_load_arxiv_metadata_file_loads_and_validates(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(VALID_METADATA), encoding="utf-8")

    record = load_arxiv_metadata_file(metadata_path)

    assert record["arxiv_id"] == "2301.00001v1"
    assert record["authors"] == ["Ada Example", "Bruno Example"]


def test_load_arxiv_metadata_file_rejects_json_list(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata_list.json"
    metadata_path.write_text(json.dumps([VALID_METADATA]), encoding="utf-8")

    with pytest.raises(ArxivMetadataError, match="object/dict"):
        load_arxiv_metadata_file(metadata_path)


def test_load_arxiv_metadata_list_file_accepts_valid_list(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata_batch.json"
    records = [VALID_METADATA, {**VALID_METADATA, "arxiv_id": "2301.00002v1"}]
    metadata_path.write_text(json.dumps(records), encoding="utf-8")

    loaded = load_arxiv_metadata_list_file(metadata_path)

    assert len(loaded) == 2
    assert loaded[0]["arxiv_id"] == "2301.00001v1"
    assert loaded[1]["arxiv_id"] == "2301.00002v1"


def test_load_arxiv_metadata_list_file_rejects_non_list(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata_batch.json"
    metadata_path.write_text(json.dumps(VALID_METADATA), encoding="utf-8")

    with pytest.raises(ArxivMetadataError, match="must contain a JSON list"):
        load_arxiv_metadata_list_file(metadata_path)


def test_load_arxiv_metadata_list_file_reports_bad_index(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata_batch.json"
    bad = {**VALID_METADATA, "title": ""}
    metadata_path.write_text(json.dumps([VALID_METADATA, bad]), encoding="utf-8")

    with pytest.raises(ArxivMetadataError, match="index 2"):
        load_arxiv_metadata_list_file(metadata_path)


def test_load_arxiv_id_list_file_accepts_comments_and_urls(tmp_path: Path) -> None:
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text(
        """
# selected pilot IDs
2301.00001v1
arXiv:2301.00002v2  # trailing note
https://arxiv.org/abs/2301.00003v1
https://arxiv.org/pdf/2301.00004v1.pdf

""".strip()
        + "\n",
        encoding="utf-8",
    )

    ids = load_arxiv_id_list_file(ids_path)

    assert ids == [
        "2301.00001v1",
        "arXiv:2301.00002v2",
        "https://arxiv.org/abs/2301.00003v1",
        "https://arxiv.org/pdf/2301.00004v1.pdf",
    ]


def test_load_arxiv_id_list_file_rejects_empty_file(tmp_path: Path) -> None:
    ids_path = tmp_path / "empty_ids.txt"
    ids_path.write_text("# no ids here\n\n", encoding="utf-8")

    try:
        load_arxiv_id_list_file(ids_path)
    except ArxivMetadataError as exc:
        assert "contains no IDs" in str(exc)
    else:
        raise AssertionError("expected ArxivMetadataError")
