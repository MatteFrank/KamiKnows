"""Tests for scripts/run_model_mini_benchmark.py."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.ingestion.arxiv_metadata import write_arxiv_metadata_list_file
from kamiknows.storage.jsonl import read_jsonl_records
from scripts import run_model_mini_benchmark

SAMPLE_METADATA_A1 = {
    "arxiv_id": "0000.00011v1",
    "title": "Fast calorimeter simulation benchmark record",
    "authors": ["Ada Example"],
    "abstract": "We present a tutorial abstract about calorimeter simulation.",
    "categories": ["hep-ex", "physics.ins-det"],
    "published": "2026-01-01T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00011v1",
}

SAMPLE_METADATA_A2 = {
    "arxiv_id": "0000.00012v1",
    "title": "Detector response benchmark record",
    "authors": ["Bruno Example"],
    "abstract": "We present a tutorial abstract about detector response.",
    "categories": ["hep-ex"],
    "published": "2026-01-02T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00012v1",
}

SAMPLE_METADATA_B1 = {
    "arxiv_id": "0000.00021v1",
    "title": "Higgs analysis benchmark record",
    "authors": ["Chiara Example"],
    "abstract": "We present a tutorial abstract about Higgs analysis.",
    "categories": ["hep-ex"],
    "published": "2026-01-03T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00021v1",
}

SAMPLE_METADATA_B2 = {
    "arxiv_id": "0000.00022v1",
    "title": "LHC measurement benchmark record",
    "authors": ["Diego Example"],
    "abstract": "We present a tutorial abstract about an LHC measurement.",
    "categories": ["hep-ex"],
    "published": "2026-01-04T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00022v1",
}


def _write_group(path: Path, records: list[dict]) -> None:
    write_arxiv_metadata_list_file(records, path)


def test_run_model_mini_benchmark_from_local_metadata(tmp_path: Path, capsys) -> None:
    metadata_a = tmp_path / "metadata_a.json"
    metadata_b = tmp_path / "metadata_b.json"
    output_dir = tmp_path / "benchmark"
    _write_group(metadata_a, [SAMPLE_METADATA_A1, SAMPLE_METADATA_A2])
    _write_group(metadata_b, [SAMPLE_METADATA_B1, SAMPLE_METADATA_B2])

    exit_code = run_model_mini_benchmark.main(
        [
            "--metadata-a",
            str(metadata_a),
            "--metadata-b",
            str(metadata_b),
            "--backend",
            "fake",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "KamiKnows model mini benchmark completed" in captured.out
    assert "Formal status: PASS" in captured.out

    report_path = output_dir / "mini_benchmark_report.json"
    manifest_path = output_dir / "dataset_manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["benchmark_type"] == "formal_two_query_model_mini_benchmark"
    assert report["formal_status"] == "PASS"
    assert report["prompt"]["prompt_version"] == "abstract_to_json_v0"
    assert len(report["prompt"]["prompt_template_sha256"]) == 64
    assert report["prompt"]["extraction_schema_version"] == "extraction_schema_v0"
    assert report["groups"]["query_a"]["result"]["output_records"] == 2
    assert report["groups"]["query_b"]["result"]["output_records"] == 2
    assert report["dataset_manifest_path"] == str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["run_context"]["backend"] == "fake"

    records_a = read_jsonl_records(output_dir / "query_a_fake_fake.jsonl")
    records_b = read_jsonl_records(output_dir / "query_b_fake_fake.jsonl")
    assert len(records_a) == 2
    assert len(records_b) == 2
    assert records_a[0]["run"]["backend"] == "fake"


def test_run_model_mini_benchmark_can_download_two_queries(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "benchmark_remote"
    seen_queries: list[str] = []

    def fake_search(query: str, *, max_results: int, timeout_seconds: int) -> list[dict]:
        seen_queries.append(query)
        assert max_results == 2
        if query == "cat:hep-ex AND calorimeter":
            return [SAMPLE_METADATA_A1, SAMPLE_METADATA_A2]
        if query == "cat:hep-ex AND Higgs":
            return [SAMPLE_METADATA_B1, SAMPLE_METADATA_B2]
        raise AssertionError(query)

    monkeypatch.setattr(run_model_mini_benchmark, "search_arxiv_metadata", fake_search)

    exit_code = run_model_mini_benchmark.main(
        [
            "--query-a",
            "cat:hep-ex AND calorimeter",
            "--query-b",
            "cat:hep-ex AND Higgs",
            "--max-results",
            "2",
            "--backend",
            "fake",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert seen_queries == ["cat:hep-ex AND calorimeter", "cat:hep-ex AND Higgs"]
    assert (output_dir / "metadata_query_a.json").exists()
    assert (output_dir / "metadata_query_b.json").exists()
    report = json.loads((output_dir / "mini_benchmark_report.json").read_text())
    assert report["groups"]["query_a"]["metadata"]["source_type"] == "remote_arxiv_query"


def test_run_model_mini_benchmark_rejects_bad_max_results(
    tmp_path: Path, capsys
) -> None:
    exit_code = run_model_mini_benchmark.main(
        ["--max-results", "0", "--backend", "fake", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--max-results must be >= 1" in captured.err
