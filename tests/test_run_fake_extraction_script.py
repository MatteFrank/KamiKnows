"""CLI smoke tests for scripts/run_fake_extraction.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from kamiknows.run_metadata import DEFAULT_PROMPT_VERSION
from kamiknows.storage.jsonl import read_jsonl_records


def test_run_fake_extraction_script_writes_one_jsonl_record(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "outputs" / "extractions.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_fake_extraction.py"),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Backend: fake" in completed.stdout
    assert "Validated extraction JSON" in completed.stdout
    assert "Saved one JSONL record" in completed.stdout

    records = read_jsonl_records(output_path)
    assert len(records) == 1
    record = records[0]
    assert record["extraction"]["confidence"] == "medium"
    assert record["extraction"]["title"] == "Fast calorimeter simulation for HEP"
    assert record["run"]["backend"] == "fake"
    assert record["run"]["model"] == "fake"
    assert record["run"]["prompt_version"] == DEFAULT_PROMPT_VERSION
    assert record["run"]["run_id"]
    assert record["run"]["created_at"].endswith("Z")


def test_run_fake_extraction_script_accepts_explicit_fake_backend(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "outputs" / "custom.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_fake_extraction.py"),
            "--backend",
            "fake",
            "--title",
            "Custom HEP abstract",
            "--abstract",
            "We test a small extraction example for a controlled tutorial.",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Backend: fake" in completed.stdout

    records = read_jsonl_records(output_path)
    assert len(records) == 1
    assert records[0]["extraction"]["title"] == "Custom HEP abstract"
    assert records[0]["run"]["backend"] == "fake"
