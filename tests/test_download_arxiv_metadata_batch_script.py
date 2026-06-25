"""Tests for scripts/download_arxiv_metadata_batch.py."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.ingestion.arxiv_metadata import (
    load_arxiv_metadata_list_file,
    write_arxiv_metadata_list_file,
)
from scripts import download_arxiv_metadata_batch

SAMPLE_METADATA_1 = {
    "arxiv_id": "0000.00001v1",
    "title": "Fast calorimeter simulation for high energy physics",
    "authors": ["Ada Example"],
    "abstract": "We present a tutorial abstract for detector simulation.",
    "categories": ["hep-ex"],
    "published": "2026-01-01T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00001v1",
}

SAMPLE_METADATA_2 = {
    "arxiv_id": "0000.00002v1",
    "title": "Higgs analysis tutorial metadata record",
    "authors": ["Bruno Example"],
    "abstract": "We present a tutorial abstract for a Higgs analysis.",
    "categories": ["hep-ex"],
    "published": "2026-01-02T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00002v1",
}


def test_write_arxiv_metadata_list_file_round_trips(tmp_path: Path) -> None:
    output_path = tmp_path / "metadata" / "records.json"

    written = write_arxiv_metadata_list_file([SAMPLE_METADATA_1, SAMPLE_METADATA_2], output_path)
    loaded = load_arxiv_metadata_list_file(written)

    assert written == output_path
    assert len(loaded) == 2
    assert loaded[0]["arxiv_id"] == "0000.00001v1"


def test_download_arxiv_metadata_batch_query_writes_json(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "downloaded.json"

    def fake_search(query: str, *, max_results: int, timeout_seconds: int) -> list[dict]:
        assert query == "cat:hep-ex AND calorimeter"
        assert max_results == 2
        assert timeout_seconds == 7
        return [SAMPLE_METADATA_1, SAMPLE_METADATA_2]

    monkeypatch.setattr(download_arxiv_metadata_batch, "search_arxiv_metadata", fake_search)

    exit_code = download_arxiv_metadata_batch.main(
        [
            "--query",
            "cat:hep-ex AND calorimeter",
            "--max-results",
            "2",
            "--timeout-seconds",
            "7",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Metadata download completed" in captured.out
    assert "Records: 2" in captured.out

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[1]["arxiv_id"] == "0000.00002v1"


def test_download_arxiv_metadata_batch_ids_writes_json(
    monkeypatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "ids.json"
    seen: list[str] = []

    def fake_fetch(arxiv_id: str, *, timeout_seconds: int) -> dict:
        seen.append(arxiv_id)
        if arxiv_id == "0000.00001v1":
            return SAMPLE_METADATA_1
        if arxiv_id == "0000.00002v1":
            return SAMPLE_METADATA_2
        raise AssertionError(arxiv_id)

    monkeypatch.setattr(download_arxiv_metadata_batch, "fetch_arxiv_metadata_by_id", fake_fetch)

    exit_code = download_arxiv_metadata_batch.main(
        [
            "--ids",
            "0000.00001v1",
            "0000.00002v1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert seen == ["0000.00001v1", "0000.00002v1"]
    loaded = load_arxiv_metadata_list_file(output_path)
    assert len(loaded) == 2


def test_download_arxiv_metadata_batch_rejects_bad_max_results(
    tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "bad.json"

    exit_code = download_arxiv_metadata_batch.main(
        [
            "--query",
            "cat:hep-ex AND calorimeter",
            "--max-results",
            "0",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--max-results must be >= 1" in captured.err
    assert not output_path.exists()


def test_download_arxiv_metadata_batch_ids_file_writes_json(
    monkeypatch, tmp_path: Path
) -> None:
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text(
        "# selected ids\n0000.00001v1\narXiv:0000.00002v1\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "ids_file.json"
    seen: list[str] = []

    def fake_fetch(arxiv_id: str, *, timeout_seconds: int) -> dict:
        seen.append(arxiv_id)
        if arxiv_id == "0000.00001v1":
            return SAMPLE_METADATA_1
        if arxiv_id == "arXiv:0000.00002v1":
            return SAMPLE_METADATA_2
        raise AssertionError(arxiv_id)

    monkeypatch.setattr(download_arxiv_metadata_batch, "fetch_arxiv_metadata_by_id", fake_fetch)

    exit_code = download_arxiv_metadata_batch.main(
        [
            "--ids-file",
            str(ids_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert seen == ["0000.00001v1", "arXiv:0000.00002v1"]
    loaded = load_arxiv_metadata_list_file(output_path)
    assert len(loaded) == 2
