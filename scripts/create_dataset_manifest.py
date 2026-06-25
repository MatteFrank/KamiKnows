"""Create a KamiKnows dataset manifest for generated outputs.

The manifest is a machine-readable inventory of files produced by a tutorial run.
It records paths, sizes, SHA-256 hashes, JSONL record counts, backend/model
context, prompt identity, and missing files.

Examples from the repository root:

    python scripts/create_dataset_manifest.py \
      --from-mini-benchmark-dir outputs/model_mini_benchmark \
      --output outputs/model_mini_benchmark/dataset_manifest.json

    python scripts/create_dataset_manifest.py \
      --metadata-file outputs/metadata/calorimeter_metadata.json \
      --jsonl-file outputs/qwen_calorimeter_extractions.jsonl \
      --summary-file outputs/qwen_calorimeter_summary.json \
      --output outputs/dataset_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/create_dataset_manifest.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.dataset_manifest import (
    DatasetManifestError,
    build_dataset_manifest,
    write_dataset_manifest,
)
from kamiknows.extraction.prompt_registry import (
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
    ABSTRACT_TO_JSON_PROMPT_VERSION,
    EXTRACTION_SCHEMA_VERSION,
)

DEFAULT_OUTPUT_PATH = Path("outputs/dataset_manifest.json")
DEFAULT_MANIFEST_NAME = "kamiknows_fase0_dataset"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a KamiKnows dataset manifest for JSON/JSONL outputs."
    )
    parser.add_argument(
        "--from-mini-benchmark-dir",
        type=Path,
        help=(
            "Auto-register files from a run_model_mini_benchmark.py output directory. "
            "This collects metadata_*.json, *.jsonl, *_summary.json, and "
            "mini_benchmark_report.json when present."
        ),
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        action="append",
        default=[],
        help="Metadata JSON file to register. Can be repeated.",
    )
    parser.add_argument(
        "--jsonl-file",
        type=Path,
        action="append",
        default=[],
        help="Extraction JSONL file to register. Can be repeated.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        action="append",
        default=[],
        help="Formal summary JSON file to register. Can be repeated.",
    )
    parser.add_argument(
        "--checklist-file",
        type=Path,
        action="append",
        default=[],
        help="Manual quality checklist Markdown file to register. Can be repeated.",
    )
    parser.add_argument(
        "--manual-review-summary-file",
        type=Path,
        action="append",
        default=[],
        help="Manual review summary JSON file to register. Can be repeated.",
    )
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        help="Benchmark report JSON file to register.",
    )
    parser.add_argument(
        "--quality-gate-report",
        type=Path,
        help="Quality gate report JSON file to register.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Manifest output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", default=ABSTRACT_TO_JSON_PROMPT_VERSION)
    parser.add_argument(
        "--prompt-template-sha256",
        default=ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
    )
    parser.add_argument("--extraction-schema-version", default=EXTRACTION_SCHEMA_VERSION)
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def _load_benchmark_context(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "backend": data.get("backend"),
        "model": data.get("model"),
        "prompt": data.get("prompt", {}) if isinstance(data.get("prompt"), dict) else {},
    }


def collect_files_from_mini_benchmark_dir(directory: Path) -> list[tuple[Path, str]]:
    """Collect known mini-benchmark output files with roles."""
    if not directory.exists():
        raise DatasetManifestError(f"mini-benchmark directory does not exist: {directory}")
    if not directory.is_dir():
        raise DatasetManifestError(f"not a directory: {directory}")

    files: list[tuple[Path, str]] = []
    for path in sorted(directory.glob("metadata_*.json")):
        files.append((path, "metadata"))
    for path in sorted(directory.glob("*.jsonl")):
        files.append((path, "jsonl_extractions"))
    for path in sorted(directory.glob("*_summary.json")):
        if not path.name.endswith("_manual_review_summary.json"):
            files.append((path, "summary"))
    for path in sorted(directory.glob("*_manual_quality_checklist.md")):
        files.append((path, "manual_quality_checklist"))
    for path in sorted(directory.glob("*_manual_review_summary.json")):
        files.append((path, "manual_review_summary"))

    workflow_report_path = directory / "benchmark_quality_workflow_report.json"
    if workflow_report_path.exists():
        files.append((workflow_report_path, "workflow_report"))

    quality_gate_report_path = directory / "quality_gate_report.json"
    if quality_gate_report_path.exists():
        files.append((quality_gate_report_path, "quality_gate_report"))

    pilot_report_path = directory / "pilot_report.json"
    if pilot_report_path.exists():
        files.append((pilot_report_path, "pilot_report"))

    report_path = directory / "mini_benchmark_report.json"
    if report_path.exists():
        files.append((report_path, "benchmark_report"))

    return files


def build_files_from_args(args: argparse.Namespace) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    if args.from_mini_benchmark_dir is not None:
        files.extend(collect_files_from_mini_benchmark_dir(args.from_mini_benchmark_dir))

    files.extend((path, "metadata") for path in args.metadata_file)
    files.extend((path, "jsonl_extractions") for path in args.jsonl_file)
    files.extend((path, "summary") for path in args.summary_file)
    files.extend((path, "manual_quality_checklist") for path in args.checklist_file)
    files.extend(
        (path, "manual_review_summary")
        for path in args.manual_review_summary_file
    )

    if args.benchmark_report is not None:
        files.append((args.benchmark_report, "benchmark_report"))
    if args.quality_gate_report is not None:
        files.append((args.quality_gate_report, "quality_gate_report"))

    # Preserve order but avoid duplicate path+role entries.
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[Path, str]] = []
    for path, role in files:
        key = (str(path), role)
        if key not in seen:
            unique.append((path, role))
            seen.add(key)
    return unique


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        files = build_files_from_args(args)
        benchmark_report = args.benchmark_report
        if args.from_mini_benchmark_dir is not None and benchmark_report is None:
            candidate = args.from_mini_benchmark_dir / "mini_benchmark_report.json"
            benchmark_report = candidate if candidate.exists() else None

        context = _load_benchmark_context(benchmark_report) if benchmark_report else {}
        prompt_context = context.get("prompt", {}) if isinstance(context.get("prompt"), dict) else {}

        backend = args.backend or context.get("backend")
        model = args.model or context.get("model")
        prompt_version = args.prompt_version or prompt_context.get("prompt_version")
        prompt_template_sha256 = (
            args.prompt_template_sha256 or prompt_context.get("prompt_template_sha256")
        )
        extraction_schema_version = (
            args.extraction_schema_version
            or prompt_context.get("extraction_schema_version")
        )

        manifest = build_dataset_manifest(
            name=args.name,
            files=files,
            backend=backend,
            model=model,
            prompt_version=prompt_version,
            prompt_template_sha256=prompt_template_sha256,
            extraction_schema_version=extraction_schema_version,
            notes=args.notes,
            base_dir=REPO_ROOT,
        )
        write_dataset_manifest(manifest, args.output)
    except (DatasetManifestError, OSError, ValueError) as exc:
        print(f"KamiKnows dataset manifest failed: {exc}", file=sys.stderr)
        return 1

    print("KamiKnows dataset manifest written")
    print(f"Manifest: {args.output}")
    print(f"Status: {manifest['status']}")
    print(f"Files: {len(manifest['files'])}")
    if manifest["missing_paths"]:
        print(f"Missing files: {len(manifest['missing_paths'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
