"""Run a deterministic offline end-to-end smoke test for KamiKnows Fase 0.

This script is intentionally small and does not require network or Ollama.
It verifies the current minimal path:

offline arXiv-style metadata JSON -> validation -> fake ModelPlugin ->
abstract_to_json -> traceable JSONL save -> read-back sanity checks

Usage from the repository root:

    python scripts/smoke_test_offline.py

Optional paths:

    python scripts/smoke_test_offline.py \
      --metadata-file data/examples/arxiv_metadata_example.json \
      --output outputs/smoke_arxiv_extractions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/smoke_test_offline.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.run_metadata import RunMetadataError, validate_run_metadata
from kamiknows.storage.jsonl import JsonlStorageError, read_jsonl_records
from scripts import run_arxiv_extraction

DEFAULT_METADATA_FILE = Path("data/examples/arxiv_metadata_example.json")
DEFAULT_OUTPUT_FILE = Path("outputs/smoke_arxiv_extractions.jsonl")


class SmokeTestError(RuntimeError):
    """Raised when the offline smoke test output is not structurally valid."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic offline KamiKnows Fase 0 smoke test."
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA_FILE,
        help=(
            "Offline arXiv-style metadata JSON file. "
            "Default: data/examples/arxiv_metadata_example.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Destination JSONL file. Default: outputs/smoke_arxiv_extractions.jsonl",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output JSONL instead of replacing it first.",
    )
    return parser.parse_args(argv)


def validate_smoke_record(record: dict[str, Any]) -> None:
    """Check only the minimal traceable record shape produced by the smoke test."""
    for key in ("source", "extraction", "run"):
        if key not in record:
            raise SmokeTestError(f"missing top-level key in smoke record: {key}")

    source = record["source"]
    extraction = record["extraction"]
    run = record["run"]

    if not isinstance(source, dict):
        raise SmokeTestError("record['source'] must be a dictionary")
    if not isinstance(extraction, dict):
        raise SmokeTestError("record['extraction'] must be a dictionary")
    if not isinstance(run, dict):
        raise SmokeTestError("record['run'] must be a dictionary")

    for key in ("arxiv_id", "title", "authors", "categories", "published", "url"):
        if key not in source:
            raise SmokeTestError(f"missing source key: {key}")

    for key in ("title", "field", "main_claim", "method", "limitations", "confidence"):
        if key not in extraction:
            raise SmokeTestError(f"missing extraction key: {key}")

    try:
        validated_run = validate_run_metadata(run)
    except RunMetadataError as exc:
        raise SmokeTestError(f"invalid run metadata: {exc}") from exc

    if validated_run["backend"] != "fake":
        raise SmokeTestError("offline smoke test must use the fake backend")
    if validated_run["model"] != "fake":
        raise SmokeTestError("offline smoke test must record model='fake'")


def run_smoke_test(metadata_file: Path, output: Path, append: bool = False) -> Path:
    """Run the offline end-to-end demo and validate the produced JSONL."""
    if not metadata_file.exists():
        raise SmokeTestError(f"metadata file does not exist: {metadata_file}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not append:
        output.unlink()

    exit_code = run_arxiv_extraction.main(
        [
            "--metadata-file",
            str(metadata_file),
            "--backend",
            "fake",
            "--output",
            str(output),
        ]
    )
    if exit_code != 0:
        raise SmokeTestError(f"run_arxiv_extraction failed with exit code {exit_code}")

    records = read_jsonl_records(output)
    if not records:
        raise SmokeTestError(f"no records were written to: {output}")
    validate_smoke_record(records[-1])
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        output_path = run_smoke_test(
            metadata_file=args.metadata_file,
            output=args.output,
            append=args.append,
        )
    except (SmokeTestError, JsonlStorageError, OSError) as exc:
        print(f"KamiKnows offline smoke test failed: {exc}", file=sys.stderr)
        return 1

    print("\nSMOKE TEST PASSED")
    print("Checked path:")
    print("  offline metadata -> validation -> fake backend -> extraction -> JSONL")
    print(f"Output file: {output_path}")
    print("Inspect it with:")
    print(f"  cat {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
