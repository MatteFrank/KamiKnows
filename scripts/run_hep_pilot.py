"""Run a controlled KamiKnows HEP pilot over 10-20 arXiv-style papers.

This is the first Fase 1 pilot runner. It keeps the phases explicit:

1. metadata ingestion
   - load a frozen metadata JSON list, or
   - run one controlled arXiv query once and save the metadata snapshot;
2. model interpretation
   - run one backend/model over the frozen title+abstract inputs;
3. formal checks
   - summarize JSONL completeness and schema-like fields;
4. manual review preparation
   - create a checklist over a small sample;
5. traceability
   - write a pilot report and dataset manifest.

It does not download PDFs, parse LaTeX, build RAG, benchmark scientific quality,
or fine-tune any model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from kamiknows.extraction.prompt_registry import (
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
    EXTRACTION_SCHEMA_VERSION,
)
from kamiknows.ingestion.arxiv_downloader import (
    ArxivIngestionError,
    fetch_arxiv_metadata_by_id,
    search_arxiv_metadata,
)
from kamiknows.ingestion.arxiv_metadata import (
    ArxivMetadataError,
    load_arxiv_id_list_file,
    load_arxiv_metadata_list_file,
    validate_arxiv_metadata,
    write_arxiv_metadata_list_file,
)
from kamiknows.quality.manual_checklist import (
    DEFAULT_MAX_ABSTRACT_CHARS,
    ManualChecklistError,
    build_manual_quality_checklist_from_jsonl,
)
from kamiknows.storage.jsonl import JsonlStorageError, read_jsonl_records
from scripts.run_batch_arxiv_extraction import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    run_batch,
)
from scripts.summarize_jsonl import summarize_records, write_summary_json

DEFAULT_PILOT_QUERY = "cat:hep-ex AND calorimeter"
DEFAULT_OUTPUT_DIR = Path("outputs/hep_pilot")
DEFAULT_MIN_PAPERS = 10
DEFAULT_MAX_PAPERS = 20
DEFAULT_REVIEW_LIMIT = 5
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_PROMPT_VERSION = "abstract_to_json_v0"


class HepPilotError(RuntimeError):
    """Raised when the HEP pilot cannot be completed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled 10-20 paper KamiKnows HEP pilot."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--metadata-list",
        type=Path,
        help="Frozen arXiv-style metadata JSON list to use instead of remote arXiv.",
    )
    source.add_argument(
        "--ids-file",
        type=Path,
        help=(
            "Plain-text file with one arXiv ID or arXiv abs/pdf URL per line. "
            "The file is fetched once and frozen as pilot_metadata.json."
        ),
    )
    source.add_argument(
        "--query",
        default=DEFAULT_PILOT_QUERY,
        help=f"Controlled arXiv query. Default: {DEFAULT_PILOT_QUERY!r}",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MIN_PAPERS,
        help=(
            "Number of remote arXiv metadata records to request. "
            f"For the pilot use {DEFAULT_MIN_PAPERS}-{DEFAULT_MAX_PAPERS}. "
            f"Default: {DEFAULT_MIN_PAPERS}"
        ),
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
        help=f"Ollama model when --backend ollama. Default: {DEFAULT_OLLAMA_MODEL}",
    )
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Remote arXiv timeout. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt/schema version label recorded in run metadata.",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=DEFAULT_REVIEW_LIMIT,
        help=f"Records to include in the manual checklist. Default: {DEFAULT_REVIEW_LIMIT}",
    )
    parser.add_argument(
        "--max-abstract-chars",
        type=int,
        default=DEFAULT_MAX_ABSTRACT_CHARS,
        help="Maximum abstract characters shown in the manual checklist.",
    )
    parser.add_argument(
        "--allow-small-sample",
        action="store_true",
        help="Allow fewer than 10 records. Intended only for tests and dry-runs.",
    )
    parser.add_argument(
        "--no-checklist",
        action="store_true",
        help="Do not generate a manual review checklist.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write dataset_manifest.json.",
    )
    return parser.parse_args(argv)


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


def validate_pilot_sample_size(
    count: int,
    *,
    allow_small_sample: bool = False,
    min_papers: int = DEFAULT_MIN_PAPERS,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> None:
    """Validate the pilot size rule: normally 10-20 papers."""
    if count > max_papers:
        raise HepPilotError(
            f"pilot sample has {count} records, above the maximum of {max_papers}; "
            "use a smaller --max-results or a smaller metadata list"
        )
    if count < min_papers and not allow_small_sample:
        raise HepPilotError(
            f"pilot sample has {count} records, below the minimum of {min_papers}; "
            "use --allow-small-sample only for dry-runs/tests"
        )


def load_or_download_pilot_metadata(
    *,
    metadata_list: Path | None,
    ids_file: Path | None,
    query: str,
    max_results: int,
    timeout_seconds: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Load frozen metadata or download one controlled arXiv query and save it."""
    start = time.perf_counter()
    output_metadata_path = output_dir / "pilot_metadata.json"

    if metadata_list is not None:
        records = load_arxiv_metadata_list_file(metadata_list)
        write_arxiv_metadata_list_file(records, output_metadata_path)
        elapsed = time.perf_counter() - start
        return records, output_metadata_path, {
            "source_type": "local_metadata_file",
            "input_path": str(metadata_list),
            "frozen_metadata_path": str(output_metadata_path),
            "query": None,
            "records": len(records),
            "metadata_seconds": elapsed,
        }

    if ids_file is not None:
        input_ids = load_arxiv_id_list_file(ids_file)
        records = []
        for arxiv_id in input_ids:
            record = fetch_arxiv_metadata_by_id(
                arxiv_id,
                timeout_seconds=timeout_seconds,
            )
            records.append(validate_arxiv_metadata(record))
        write_arxiv_metadata_list_file(records, output_metadata_path)
        elapsed = time.perf_counter() - start
        return records, output_metadata_path, {
            "source_type": "remote_arxiv_ids_file",
            "input_path": str(ids_file),
            "frozen_metadata_path": str(output_metadata_path),
            "query": None,
            "records": len(records),
            "metadata_seconds": elapsed,
        }

    records = search_arxiv_metadata(
        query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
    )
    if not records:
        raise HepPilotError("controlled arXiv query returned no records")
    validated = [validate_arxiv_metadata(record) for record in records]
    write_arxiv_metadata_list_file(validated, output_metadata_path)
    elapsed = time.perf_counter() - start
    return validated, output_metadata_path, {
        "source_type": "remote_arxiv_query",
        "input_path": None,
        "frozen_metadata_path": str(output_metadata_path),
        "query": query,
        "records": len(validated),
        "metadata_seconds": elapsed,
    }


def build_pilot_report(
    *,
    metadata_info: dict[str, Any],
    backend: str,
    model: str,
    jsonl_path: Path,
    summary_path: Path,
    checklist_path: Path | None,
    manifest_path: Path | None,
    input_records: int,
    output_records: int,
    extraction_seconds: float,
    summary: dict[str, Any],
    prompt_version: str,
) -> dict[str, Any]:
    formal_status = "PASS"
    if summary.get("evaluation_status") != "PASS":
        formal_status = "WARN"
    if input_records != output_records:
        formal_status = "WARN"

    next_steps = [
        "Inspect the JSONL records.",
        "Complete the manual quality checklist on a sample of records.",
        "Summarize the completed checklist into manual_review_summary.json.",
        "Regenerate dataset_manifest.json and run the quality gate.",
    ]

    return {
        "pilot_type": "hep_10_20_paper_controlled_query_pilot",
        "scope": (
            "abstract-level extraction-to-JSON pilot; no PDF/LaTeX parsing, "
            "no RAG, no fine-tuning, no automatic scientific correctness score"
        ),
        "formal_status": formal_status,
        "metadata": metadata_info,
        "backend": backend,
        "model": model if backend == "ollama" else "fake",
        "prompt": {
            "prompt_version": prompt_version,
            "prompt_template_sha256": ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
            "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
        },
        "counts": {
            "input_records": input_records,
            "output_records": output_records,
            "manual_review_records_requested": None if checklist_path is None else "see checklist",
        },
        "timing": {
            "extraction_seconds": extraction_seconds,
            "seconds_per_record": extraction_seconds / output_records if output_records else None,
        },
        "outputs": {
            "jsonl_path": str(jsonl_path),
            "summary_path": str(summary_path),
            "manual_quality_checklist_path": str(checklist_path) if checklist_path else None,
            "dataset_manifest_path": str(manifest_path) if manifest_path else None,
        },
        "summary": summary,
        "next_steps": next_steps,
    }


def run_hep_pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Run the HEP pilot and return the machine-readable report."""
    if args.max_results < 1:
        raise HepPilotError("--max-results must be >= 1")
    if args.timeout_seconds < 1:
        raise HepPilotError("--timeout-seconds must be >= 1")
    if args.review_limit < 1:
        raise HepPilotError("--review-limit must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, metadata_path, metadata_info = load_or_download_pilot_metadata(
        metadata_list=args.metadata_list,
        ids_file=args.ids_file,
        query=args.query,
        max_results=args.max_results,
        timeout_seconds=args.timeout_seconds,
        output_dir=args.output_dir,
    )
    validate_pilot_sample_size(
        len(records),
        allow_small_sample=args.allow_small_sample,
    )

    model_slug = _safe_model_slug(args.backend, args.model)
    jsonl_path = args.output_dir / f"pilot_{args.backend}_{model_slug}.jsonl"
    summary_path = args.output_dir / f"pilot_{args.backend}_{model_slug}_summary.json"
    report_path = args.output_dir / "pilot_report.json"
    checklist_path = None if args.no_checklist else args.output_dir / "pilot_manual_quality_checklist.md"
    manifest_path = None if args.no_manifest else args.output_dir / "dataset_manifest.json"

    for path in (jsonl_path, summary_path, report_path):
        if path.exists():
            path.unlink()
    if checklist_path is not None and checklist_path.exists():
        checklist_path.unlink()

    start = time.perf_counter()
    saved_records = run_batch(
        metadata_records=records,
        output_path=jsonl_path,
        backend=args.backend,
        model_name=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        prompt_version=args.prompt_version,
    )
    extraction_seconds = time.perf_counter() - start

    parsed_records = read_jsonl_records(jsonl_path)
    summary = summarize_records(parsed_records, require_source=True)
    write_summary_json(summary, summary_path)

    if checklist_path is not None:
        checklist = build_manual_quality_checklist_from_jsonl(
            jsonl_path=jsonl_path,
            limit=args.review_limit,
            max_abstract_chars=args.max_abstract_chars,
        )
        checklist_path.write_text(checklist, encoding="utf-8")

    report = build_pilot_report(
        metadata_info=metadata_info,
        backend=args.backend,
        model=args.model,
        jsonl_path=jsonl_path,
        summary_path=summary_path,
        checklist_path=checklist_path,
        manifest_path=manifest_path,
        input_records=len(records),
        output_records=len(saved_records),
        extraction_seconds=extraction_seconds,
        summary=summary,
        prompt_version=args.prompt_version,
    )
    _write_json(report, report_path)

    if manifest_path is not None:
        files: list[tuple[Path, str]] = [
            (metadata_path, "metadata"),
            (jsonl_path, "jsonl_extractions"),
            (summary_path, "summary"),
            (report_path, "pilot_report"),
        ]
        if checklist_path is not None:
            files.append((checklist_path, "manual_quality_checklist"))
        manifest = build_dataset_manifest(
            name="kamiknows_hep_pilot",
            files=files,
            backend=report["backend"],
            model=report["model"],
            prompt_version=report["prompt"]["prompt_version"],
            prompt_template_sha256=report["prompt"]["prompt_template_sha256"],
            extraction_schema_version=report["prompt"]["extraction_schema_version"],
            benchmark_report=report_path,
            notes="Auto-generated by run_hep_pilot.py",
            base_dir=REPO_ROOT,
        )
        write_dataset_manifest(manifest, manifest_path)

    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        report = run_hep_pilot(args)
    except (
        ArxivIngestionError,
        ArxivMetadataError,
        HepPilotError,
        JsonlStorageError,
        ManualChecklistError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"KamiKnows HEP pilot failed: {exc}", file=sys.stderr)
        if args.backend == "ollama":
            print(
                "Hint: check that Ollama is running and the model is installed, "
                f"for example: ollama pull {args.model}",
                file=sys.stderr,
            )
        return 1

    print("KamiKnows HEP pilot completed")
    print(f"Backend/model: {report['backend']} / {report['model']}")
    print(f"Formal status: {report['formal_status']}")
    print(f"Metadata source: {report['metadata']['source_type']}")
    if report["metadata"].get("query"):
        print(f"Controlled query: {report['metadata']['query']}")
    print(f"Input/output records: {report['counts']['input_records']} / {report['counts']['output_records']}")
    print(f"Extraction seconds: {report['timing']['extraction_seconds']:.3f}")
    print("")
    print(f"JSONL: {report['outputs']['jsonl_path']}")
    print(f"Formal summary: {report['outputs']['summary_path']}")
    if report["outputs"].get("manual_quality_checklist_path"):
        print(f"Manual checklist: {report['outputs']['manual_quality_checklist_path']}")
    if report["outputs"].get("dataset_manifest_path"):
        print(f"Dataset manifest: {report['outputs']['dataset_manifest_path']}")
    print(f"Pilot report: {args.output_dir / 'pilot_report.json'}")
    print("")
    print("Next:")
    print("  1. Inspect the JSONL output.")
    print("  2. Complete the manual checklist sample.")
    print("  3. Summarize the checklist and rerun the quality gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
