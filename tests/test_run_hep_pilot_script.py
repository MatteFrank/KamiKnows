"""Tests for scripts/run_hep_pilot.py."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.ingestion.arxiv_metadata import write_arxiv_metadata_list_file
from scripts import run_hep_pilot

SAMPLE_RECORDS = [
    {
        "arxiv_id": f"0000.0100{i}v1",
        "title": f"HEP pilot tutorial record {i}",
        "authors": ["Ada Example"],
        "abstract": f"We present a controlled HEP pilot abstract about calorimeter response {i}.",
        "categories": ["hep-ex"],
        "published": "2026-01-01T00:00:00Z",
        "url": f"https://arxiv.org/abs/0000.0100{i}v1",
    }
    for i in range(1, 4)
]


def test_validate_pilot_sample_size_rejects_small_without_override() -> None:
    try:
        run_hep_pilot.validate_pilot_sample_size(3)
    except run_hep_pilot.HepPilotError as exc:
        assert "below the minimum" in str(exc)
    else:
        raise AssertionError("expected HepPilotError")


def test_validate_pilot_sample_size_accepts_10_to_20() -> None:
    run_hep_pilot.validate_pilot_sample_size(10)
    run_hep_pilot.validate_pilot_sample_size(20)


def test_run_hep_pilot_from_local_metadata_fake_small_sample(tmp_path: Path, capsys) -> None:
    metadata_path = tmp_path / "metadata.json"
    output_dir = tmp_path / "hep_pilot"
    write_arxiv_metadata_list_file(SAMPLE_RECORDS, metadata_path)

    exit_code = run_hep_pilot.main(
        [
            "--metadata-list",
            str(metadata_path),
            "--backend",
            "fake",
            "--allow-small-sample",
            "--review-limit",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "KamiKnows HEP pilot completed" in captured.out
    assert "Formal status: PASS" in captured.out

    report_path = output_dir / "pilot_report.json"
    jsonl_path = output_dir / "pilot_fake_fake.jsonl"
    summary_path = output_dir / "pilot_fake_fake_summary.json"
    checklist_path = output_dir / "pilot_manual_quality_checklist.md"
    manifest_path = output_dir / "dataset_manifest.json"

    assert report_path.exists()
    assert jsonl_path.exists()
    assert summary_path.exists()
    assert checklist_path.exists()
    assert manifest_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pilot_type"] == "hep_10_20_paper_controlled_query_pilot"
    assert report["formal_status"] == "PASS"
    assert report["counts"]["input_records"] == 3
    assert report["counts"]["output_records"] == 3
    assert report["metadata"]["source_type"] == "local_metadata_file"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = {entry["role"] for entry in manifest["files"]}
    assert "metadata" in roles
    assert "jsonl_extractions" in roles
    assert "summary" in roles
    assert "manual_quality_checklist" in roles
    assert "pilot_report" in roles

    checklist_text = checklist_path.read_text(encoding="utf-8")
    assert "HEP pilot tutorial record 1" in checklist_text
    assert "Review outcome" in checklist_text


def test_run_hep_pilot_can_download_remote_query(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "hep_pilot_remote"
    seen: dict[str, object] = {}

    def fake_search(query: str, *, max_results: int, timeout_seconds: int) -> list[dict]:
        seen["query"] = query
        seen["max_results"] = max_results
        seen["timeout_seconds"] = timeout_seconds
        return SAMPLE_RECORDS

    monkeypatch.setattr(run_hep_pilot, "search_arxiv_metadata", fake_search)

    exit_code = run_hep_pilot.main(
        [
            "--query",
            "cat:hep-ex AND calorimeter",
            "--max-results",
            "10",
            "--backend",
            "fake",
            "--allow-small-sample",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert seen["query"] == "cat:hep-ex AND calorimeter"
    assert seen["max_results"] == 10
    assert (output_dir / "pilot_metadata.json").exists()


def test_run_hep_pilot_rejects_too_many_records(tmp_path: Path, capsys) -> None:
    records = []
    for i in range(21):
        record = dict(SAMPLE_RECORDS[0])
        record["arxiv_id"] = f"0000.02{i:03d}v1"
        record["title"] = f"Too many record {i}"
        records.append(record)
    metadata_path = tmp_path / "metadata_many.json"
    write_arxiv_metadata_list_file(records, metadata_path)

    exit_code = run_hep_pilot.main(
        [
            "--metadata-list",
            str(metadata_path),
            "--backend",
            "fake",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "above the maximum" in captured.err


def test_run_hep_pilot_can_use_ids_file(monkeypatch, tmp_path: Path) -> None:
    ids_path = tmp_path / "pilot_ids.txt"
    ids_path.write_text(
        "# controlled pilot IDs\n0000.01001v1\nhttps://arxiv.org/abs/0000.01002v1\n0000.01003v1\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "hep_pilot_ids_file"
    seen_ids: list[str] = []

    def fake_fetch(arxiv_id: str, *, timeout_seconds: int) -> dict:
        seen_ids.append(arxiv_id)
        if arxiv_id == "0000.01001v1":
            return SAMPLE_RECORDS[0]
        if arxiv_id == "https://arxiv.org/abs/0000.01002v1":
            return SAMPLE_RECORDS[1]
        if arxiv_id == "0000.01003v1":
            return SAMPLE_RECORDS[2]
        raise AssertionError(f"unexpected id: {arxiv_id}")

    monkeypatch.setattr(run_hep_pilot, "fetch_arxiv_metadata_by_id", fake_fetch)

    exit_code = run_hep_pilot.main(
        [
            "--ids-file",
            str(ids_path),
            "--backend",
            "fake",
            "--allow-small-sample",
            "--review-limit",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert seen_ids == [
        "0000.01001v1",
        "https://arxiv.org/abs/0000.01002v1",
        "0000.01003v1",
    ]
    report = json.loads((output_dir / "pilot_report.json").read_text(encoding="utf-8"))
    assert report["metadata"]["source_type"] == "remote_arxiv_ids_file"
    assert (output_dir / "pilot_metadata.json").exists()
