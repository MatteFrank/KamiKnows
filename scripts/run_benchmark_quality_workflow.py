"""Run the KamiKnows mini-benchmark and prepare manual quality checklists.

This is the first end-to-end *review workflow* for Fase 0:

    metadata group A/B -> model extraction -> formal summary -> manual checklist

It keeps the two conceptual phases separate:

1. metadata ingestion: either use --metadata-a/--metadata-b or download once from
   arXiv with --query-a/--query-b;
2. model interpretation: run one backend/model over the frozen metadata and
   produce JSONL records plus review checklists.

The checklist is intentionally manual. Formal PASS/WARN only means that JSONL
records are complete and schema-like; it is not a scientific correctness score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the script runnable with: python scripts/run_benchmark_quality_workflow.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from kamiknows.quality.manual_checklist import (
    DEFAULT_MAX_ABSTRACT_CHARS,
    DEFAULT_REVIEW_LIMIT,
    ManualChecklistError,
    build_manual_quality_checklist_from_jsonl,
)
from scripts import run_model_mini_benchmark
from scripts.run_batch_arxiv_extraction import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
)
from scripts.run_model_mini_benchmark import (
    DEFAULT_MAX_RESULTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUERY_A,
    DEFAULT_QUERY_B,
    DEFAULT_TIMEOUT_SECONDS,
)

DEFAULT_WORKFLOW_REPORT_NAME = "benchmark_quality_workflow_report.json"


class BenchmarkQualityWorkflowError(RuntimeError):
    """Raised when the benchmark quality workflow cannot be completed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the model mini-benchmark and generate manual quality checklist "
            "Markdown files for query A and query B."
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
        help="Backend to run. Default: ollama. Use fake only for offline dry-runs.",
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
        "--review-limit",
        type=int,
        default=DEFAULT_REVIEW_LIMIT,
        help=f"Maximum records per checklist. Default: {DEFAULT_REVIEW_LIMIT}",
    )
    parser.add_argument(
        "--max-abstract-chars",
        type=int,
        default=DEFAULT_MAX_ABSTRACT_CHARS,
        help=(
            "Maximum characters of abstract shown per record. "
            f"Default: {DEFAULT_MAX_ABSTRACT_CHARS}"
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Pass through to run_model_mini_benchmark.py.",
    )
    return parser.parse_args(argv)


def _append_optional_arg(argv: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def build_mini_benchmark_argv(args: argparse.Namespace) -> list[str]:
    """Translate workflow CLI args into run_model_mini_benchmark.py args."""
    argv = [
        "--query-a",
        str(args.query_a),
        "--query-b",
        str(args.query_b),
        "--max-results",
        str(args.max_results),
        "--backend",
        str(args.backend),
        "--model",
        str(args.model),
        "--base-url",
        str(args.base_url),
        "--temperature",
        str(args.temperature),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output-dir",
        str(args.output_dir),
        "--prompt-version",
        str(args.prompt_version),
    ]
    _append_optional_arg(argv, "--metadata-a", args.metadata_a)
    _append_optional_arg(argv, "--metadata-b", args.metadata_b)
    if args.no_manifest:
        argv.append("--no-manifest")
    return argv


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BenchmarkQualityWorkflowError(f"missing expected JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise BenchmarkQualityWorkflowError(f"expected JSON object in {path}")
    return data


def _write_json(data: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_checklist(*, jsonl_path: Path, output_path: Path, limit: int, max_abstract_chars: int) -> Path:
    checklist = build_manual_quality_checklist_from_jsonl(
        jsonl_path=jsonl_path,
        limit=limit,
        max_abstract_chars=max_abstract_chars,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(checklist, encoding="utf-8")
    return output_path


def build_workflow_report(
    *,
    benchmark_report: dict[str, Any],
    benchmark_report_path: Path,
    checklist_a_path: Path,
    checklist_b_path: Path,
    review_limit: int,
) -> dict[str, Any]:
    """Build a compact machine-readable workflow report."""
    return {
        "workflow_type": "mini_benchmark_plus_manual_quality_checklist",
        "scope": (
            "formal benchmark plus human review checklist; no automatic "
            "scientific correctness score"
        ),
        "backend": benchmark_report.get("backend"),
        "model": benchmark_report.get("model"),
        "formal_status": benchmark_report.get("formal_status"),
        "review_limit": review_limit,
        "benchmark_report_path": str(benchmark_report_path),
        "checklists": {
            "query_a": str(checklist_a_path),
            "query_b": str(checklist_b_path),
        },
        "groups": {
            "query_a": benchmark_report.get("groups", {}).get("query_a", {}),
            "query_b": benchmark_report.get("groups", {}).get("query_b", {}),
        },
    }




def _workflow_manifest_files(
    *,
    benchmark_report: dict[str, Any],
    benchmark_report_path: Path,
    workflow_report_path: Path,
    checklist_a_path: Path,
    checklist_b_path: Path,
) -> list[tuple[Path, str]]:
    """Collect files produced by the benchmark-quality workflow."""
    groups = benchmark_report.get("groups", {})
    files: list[tuple[Path, str]] = []
    for group_key in ("query_a", "query_b"):
        group = groups.get(group_key, {}) if isinstance(groups, dict) else {}
        metadata = group.get("metadata", {}) if isinstance(group.get("metadata"), dict) else {}
        result = group.get("result", {}) if isinstance(group.get("result"), dict) else {}
        if metadata.get("metadata_path"):
            files.append((Path(metadata["metadata_path"]), "metadata"))
        if result.get("jsonl_path"):
            files.append((Path(result["jsonl_path"]), "jsonl_extractions"))
        if result.get("summary_path"):
            files.append((Path(result["summary_path"]), "summary"))

    files.extend(
        [
            (benchmark_report_path, "benchmark_report"),
            (workflow_report_path, "workflow_report"),
            (checklist_a_path, "manual_quality_checklist"),
            (checklist_b_path, "manual_quality_checklist"),
        ]
    )
    return files

def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    """Run benchmark then create manual quality checklists."""
    if args.review_limit < 1:
        raise BenchmarkQualityWorkflowError("--review-limit must be >= 1")
    if args.max_abstract_chars < 100:
        raise BenchmarkQualityWorkflowError("--max-abstract-chars must be >= 100")

    benchmark_exit = run_model_mini_benchmark.main(build_mini_benchmark_argv(args))
    if benchmark_exit != 0:
        raise BenchmarkQualityWorkflowError(
            f"mini benchmark failed with exit code {benchmark_exit}"
        )

    benchmark_report_path = args.output_dir / "mini_benchmark_report.json"
    benchmark_report = _read_json(benchmark_report_path)

    try:
        query_a_jsonl = Path(
            benchmark_report["groups"]["query_a"]["result"]["jsonl_path"]
        )
        query_b_jsonl = Path(
            benchmark_report["groups"]["query_b"]["result"]["jsonl_path"]
        )
    except KeyError as exc:
        raise BenchmarkQualityWorkflowError(
            "mini benchmark report does not contain expected JSONL paths"
        ) from exc

    checklist_a_path = args.output_dir / "query_a_manual_quality_checklist.md"
    checklist_b_path = args.output_dir / "query_b_manual_quality_checklist.md"
    _build_checklist(
        jsonl_path=query_a_jsonl,
        output_path=checklist_a_path,
        limit=args.review_limit,
        max_abstract_chars=args.max_abstract_chars,
    )
    _build_checklist(
        jsonl_path=query_b_jsonl,
        output_path=checklist_b_path,
        limit=args.review_limit,
        max_abstract_chars=args.max_abstract_chars,
    )

    workflow_report = build_workflow_report(
        benchmark_report=benchmark_report,
        benchmark_report_path=benchmark_report_path,
        checklist_a_path=checklist_a_path,
        checklist_b_path=checklist_b_path,
        review_limit=args.review_limit,
    )
    workflow_report_path = _write_json(
        workflow_report,
        args.output_dir / DEFAULT_WORKFLOW_REPORT_NAME,
    )
    workflow_report["workflow_report_path"] = str(workflow_report_path)

    if not args.no_manifest:
        manifest_path_text = benchmark_report.get("dataset_manifest_path")
        manifest_path = (
            Path(manifest_path_text)
            if isinstance(manifest_path_text, str) and manifest_path_text.strip()
            else args.output_dir / "dataset_manifest.json"
        )
        prompt = benchmark_report.get("prompt", {})
        prompt = prompt if isinstance(prompt, dict) else {}
        manifest = build_dataset_manifest(
            name="kamiknows_benchmark_quality_workflow",
            files=_workflow_manifest_files(
                benchmark_report=benchmark_report,
                benchmark_report_path=benchmark_report_path,
                workflow_report_path=workflow_report_path,
                checklist_a_path=checklist_a_path,
                checklist_b_path=checklist_b_path,
            ),
            backend=benchmark_report.get("backend"),
            model=benchmark_report.get("model"),
            prompt_version=prompt.get("prompt_version"),
            prompt_template_sha256=prompt.get("prompt_template_sha256"),
            extraction_schema_version=prompt.get("extraction_schema_version"),
            benchmark_report=benchmark_report_path,
            notes=(
                "Auto-generated by run_benchmark_quality_workflow.py. "
                "Includes formal benchmark files and manual quality checklist Markdown files; "
                "manual review summary JSON files can be added later with create_dataset_manifest.py."
            ),
            base_dir=REPO_ROOT,
        )
        write_dataset_manifest(manifest, manifest_path)
        workflow_report["dataset_manifest_path"] = str(manifest_path)

    return workflow_report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        workflow_report = run_workflow(args)
    except (BenchmarkQualityWorkflowError, ManualChecklistError, OSError, json.JSONDecodeError) as exc:
        print(f"KamiKnows benchmark quality workflow failed: {exc}", file=sys.stderr)
        if args.backend == "ollama":
            print(
                "Hint: check that Ollama is running and the model is installed, "
                f"for example: ollama pull {args.model}",
                file=sys.stderr,
            )
        return 1

    print("KamiKnows benchmark quality workflow completed")
    print(f"Backend/model: {workflow_report['backend']} / {workflow_report['model']}")
    print(f"Formal status: {workflow_report['formal_status']}")
    print(f"Benchmark report: {workflow_report['benchmark_report_path']}")
    print(f"Workflow report: {workflow_report['workflow_report_path']}")
    if workflow_report.get("dataset_manifest_path"):
        print(f"Dataset manifest: {workflow_report['dataset_manifest_path']}")
    print("Manual checklists:")
    for label, checklist_path in workflow_report["checklists"].items():
        print(f"  {label}: {checklist_path}")
    print("")
    print("Next: open the checklist Markdown files and review main_claim/method/limitations manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
