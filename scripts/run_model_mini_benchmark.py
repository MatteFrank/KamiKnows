"""Run a small formal model benchmark over two arXiv query groups.

This script compares one model/backend across two input query groups. It does
not compare fake vs Ollama, and it does not judge scientific correctness.

The benchmark checks formal properties only:

- metadata records can be loaded or downloaded;
- the selected model produces parseable JSON;
- required extraction/source/run fields are present;
- confidence labels are within low/medium/high;
- basic counts and timings are recorded.

The ingestion phase and interpretation phase are separable:

1. Download metadata only with scripts/download_arxiv_metadata_batch.py.
2. Run this script with --metadata-a and --metadata-b.

Or run both phases together by passing --query-a and --query-b.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/run_model_mini_benchmark.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from kamiknows.extraction.prompt_registry import (
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
    EXTRACTION_SCHEMA_VERSION,
)
from kamiknows.ingestion.arxiv_downloader import ArxivIngestionError, search_arxiv_metadata
from kamiknows.ingestion.arxiv_metadata import (
    ArxivMetadataError,
    load_arxiv_metadata_list_file,
    validate_arxiv_metadata,
    write_arxiv_metadata_list_file,
)
from kamiknows.storage.jsonl import JsonlStorageError, read_jsonl_records
from scripts.run_batch_arxiv_extraction import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    run_batch,
)
from scripts.summarize_jsonl import summarize_records, write_summary_json

DEFAULT_QUERY_A = "cat:hep-ex AND calorimeter"
DEFAULT_QUERY_B = "cat:hep-ex AND Higgs"
DEFAULT_OUTPUT_DIR = Path("outputs/model_mini_benchmark")
DEFAULT_MAX_RESULTS = 2
DEFAULT_TIMEOUT_SECONDS = 30


class MiniBenchmarkError(RuntimeError):
    """Raised when the mini benchmark cannot be completed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a formal mini benchmark for one model/backend over two arXiv "
            "query groups."
        )
    )
    parser.add_argument("--query-a", default=DEFAULT_QUERY_A)
    parser.add_argument("--query-b", default=DEFAULT_QUERY_B)
    parser.add_argument(
        "--metadata-a",
        type=Path,
        help="Optional pre-downloaded metadata JSON list for query group A.",
    )
    parser.add_argument(
        "--metadata-b",
        type=Path,
        help="Optional pre-downloaded metadata JSON list for query group B.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"Maximum remote arXiv results per query. Default: {DEFAULT_MAX_RESULTS}",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "fake"],
        default="ollama",
        help="Backend to benchmark. Default: ollama. Use fake only for dry-run tests.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name when --backend ollama. Default: {DEFAULT_OLLAMA_MODEL}",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama base URL when --backend ollama.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Default: 0.0",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout for arXiv calls. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prompt-version",
        default="abstract_to_json_v0",
        help="Prompt/schema version label recorded in JSONL run metadata.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help=(
            "Optional dataset manifest output path. Default: "
            "<output-dir>/dataset_manifest.json"
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write a dataset manifest for the mini-benchmark run.",
    )
    return parser.parse_args(argv)


def _validate_positive_int(value: int, *, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _safe_model_slug(backend: str, model: str) -> str:
    if backend == "fake":
        return "fake"
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def _write_json(data: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_or_download_group_metadata(
    *,
    label: str,
    query: str,
    metadata_path: Path | None,
    max_results: int,
    timeout_seconds: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Load pre-downloaded metadata or fetch it once and save it."""
    start = time.perf_counter()

    if metadata_path is not None:
        records = load_arxiv_metadata_list_file(metadata_path)
        elapsed = time.perf_counter() - start
        info = {
            "label": label,
            "source_type": "local_metadata_file",
            "query": query,
            "records": len(records),
            "metadata_path": str(metadata_path),
            "metadata_seconds": elapsed,
        }
        return records, metadata_path, info

    records = search_arxiv_metadata(
        query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
    )
    if not records:
        raise MiniBenchmarkError(f"query group {label} returned no records")
    validated = [validate_arxiv_metadata(record) for record in records]
    saved_path = output_dir / f"metadata_{label}.json"
    write_arxiv_metadata_list_file(validated, saved_path)
    elapsed = time.perf_counter() - start
    info = {
        "label": label,
        "source_type": "remote_arxiv_query",
        "query": query,
        "records": len(validated),
        "metadata_path": str(saved_path),
        "metadata_seconds": elapsed,
    }
    return validated, saved_path, info


def run_formal_group_benchmark(
    *,
    label: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    backend: str,
    model: str,
    base_url: str,
    temperature: float,
    prompt_version: str,
) -> dict[str, Any]:
    """Run extraction for one query group and compute formal summary."""
    model_slug = _safe_model_slug(backend, model)
    jsonl_path = output_dir / f"{label}_{backend}_{model_slug}.jsonl"
    summary_path = output_dir / f"{label}_{backend}_{model_slug}_summary.json"

    # Benchmarks should be repeatable. Avoid appending to a previous run.
    for path in (jsonl_path, summary_path):
        if path.exists():
            path.unlink()

    start = time.perf_counter()
    saved_records = run_batch(
        metadata_records=records,
        output_path=jsonl_path,
        backend=backend,
        model_name=model,
        base_url=base_url,
        temperature=temperature,
        prompt_version=prompt_version,
    )
    extraction_seconds = time.perf_counter() - start

    parsed_records = read_jsonl_records(jsonl_path)
    summary = summarize_records(parsed_records, require_source=True)
    write_summary_json(summary, summary_path)

    return {
        "label": label,
        "input_records": len(records),
        "output_records": len(saved_records),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "extraction_seconds": extraction_seconds,
        "seconds_per_record": extraction_seconds / len(saved_records),
        "summary": summary,
    }


def build_benchmark_report(
    *,
    backend: str,
    model: str,
    group_a_metadata_info: dict[str, Any],
    group_b_metadata_info: dict[str, Any],
    group_a_result: dict[str, Any],
    group_b_result: dict[str, Any],
    prompt_version: str,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable mini benchmark report."""
    summaries = [group_a_result["summary"], group_b_result["summary"]]
    formal_status = "PASS"
    if any(summary["evaluation_status"] != "PASS" for summary in summaries):
        formal_status = "WARN"
    if group_a_result["input_records"] != group_a_result["output_records"]:
        formal_status = "WARN"
    if group_b_result["input_records"] != group_b_result["output_records"]:
        formal_status = "WARN"

    return {
        "benchmark_type": "formal_two_query_model_mini_benchmark",
        "scope": "formal JSON/schema checks only; no scientific correctness judgment",
        "backend": backend,
        "model": model if backend == "ollama" else "fake",
        "prompt": {
            "prompt_version": prompt_version,
            "prompt_template_sha256": ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
            "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
        },
        "formal_status": formal_status,
        "dataset_manifest_path": manifest_path,
        "groups": {
            "query_a": {
                "metadata": group_a_metadata_info,
                "result": group_a_result,
            },
            "query_b": {
                "metadata": group_b_metadata_info,
                "result": group_b_result,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        _validate_positive_int(args.max_results, name="--max-results")
        _validate_positive_int(args.timeout_seconds, name="--timeout-seconds")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        records_a, _metadata_path_a, metadata_info_a = load_or_download_group_metadata(
            label="query_a",
            query=args.query_a,
            metadata_path=args.metadata_a,
            max_results=args.max_results,
            timeout_seconds=args.timeout_seconds,
            output_dir=args.output_dir,
        )
        records_b, _metadata_path_b, metadata_info_b = load_or_download_group_metadata(
            label="query_b",
            query=args.query_b,
            metadata_path=args.metadata_b,
            max_results=args.max_results,
            timeout_seconds=args.timeout_seconds,
            output_dir=args.output_dir,
        )

        result_a = run_formal_group_benchmark(
            label="query_a",
            records=records_a,
            output_dir=args.output_dir,
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            prompt_version=args.prompt_version,
        )
        result_b = run_formal_group_benchmark(
            label="query_b",
            records=records_b,
            output_dir=args.output_dir,
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            prompt_version=args.prompt_version,
        )

        manifest_path: Path | None = None
        if not args.no_manifest:
            manifest_path = args.manifest_output or (args.output_dir / "dataset_manifest.json")

        report = build_benchmark_report(
            backend=args.backend,
            model=args.model,
            group_a_metadata_info=metadata_info_a,
            group_b_metadata_info=metadata_info_b,
            group_a_result=result_a,
            group_b_result=result_b,
            prompt_version=args.prompt_version,
            manifest_path=str(manifest_path) if manifest_path is not None else None,
        )
        report_path = _write_json(report, args.output_dir / "mini_benchmark_report.json")

        if manifest_path is not None:
            manifest_files = [
                (Path(metadata_info_a["metadata_path"]), "metadata"),
                (Path(metadata_info_b["metadata_path"]), "metadata"),
                (Path(result_a["jsonl_path"]), "jsonl_extractions"),
                (Path(result_b["jsonl_path"]), "jsonl_extractions"),
                (Path(result_a["summary_path"]), "summary"),
                (Path(result_b["summary_path"]), "summary"),
                (report_path, "benchmark_report"),
            ]
            manifest = build_dataset_manifest(
                name="kamiknows_model_mini_benchmark",
                files=manifest_files,
                backend=report["backend"],
                model=report["model"],
                prompt_version=report["prompt"]["prompt_version"],
                prompt_template_sha256=report["prompt"]["prompt_template_sha256"],
                extraction_schema_version=report["prompt"]["extraction_schema_version"],
                benchmark_report=report_path,
                notes="Auto-generated by run_model_mini_benchmark.py",
                base_dir=REPO_ROOT,
            )
            write_dataset_manifest(manifest, manifest_path)
    except (
        ArxivIngestionError,
        ArxivMetadataError,
        JsonlStorageError,
        MiniBenchmarkError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"KamiKnows model mini benchmark failed: {exc}", file=sys.stderr)
        if args.backend == "ollama":
            print(
                "Hint: check that Ollama is running and the model is installed, "
                f"for example: ollama pull {args.model}",
                file=sys.stderr,
            )
        return 1

    print("KamiKnows model mini benchmark completed")
    print(f"Backend/model: {report['backend']} / {report['model']}")
    print(f"Formal status: {report['formal_status']}")
    print("")
    for group_key, group_data in report["groups"].items():
        metadata = group_data["metadata"]
        result = group_data["result"]
        print(f"{group_key}:")
        print(f"  query: {metadata['query']}")
        print(f"  metadata source: {metadata['source_type']}")
        print(f"  metadata records: {metadata['records']}")
        print(f"  output records: {result['output_records']}")
        print(f"  evaluation status: {result['summary']['evaluation_status']}")
        print(f"  extraction seconds: {result['extraction_seconds']:.3f}")
        print(f"  JSONL: {result['jsonl_path']}")
        print(f"  summary: {result['summary_path']}")
    print("")
    print(f"Report JSON: {report_path}")
    if report.get("dataset_manifest_path"):
        print(f"Dataset manifest: {report['dataset_manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
