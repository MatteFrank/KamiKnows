"""Run the KamiKnows Fase 1F minimal full-text parsing pilot.

This script processes a small arXiv ID list into per-paper LaTeX-derived
artifacts. It does not call an LLM, does not parse PDFs, and does not build RAG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.fulltext.paper_outputs import process_arxiv_paper, write_json
from kamiknows.fulltext.quality_report import (
    build_fulltext_manifest,
    build_quality_report,
    write_quality_reports,
)
from kamiknows.ingestion.arxiv_metadata import (
    ArxivMetadataError,
    load_arxiv_id_list_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal 3-5 paper full-text LaTeX parsing pilot."
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        required=True,
        help="Plain-text arXiv ID list for the full-text parsing pilot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for full-text parsing artifacts.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Network timeout for metadata/source downloads. Default: 60.",
    )
    return parser.parse_args(argv)


def _format_missing_input_message(ids_file: Path) -> str:
    cwd = Path.cwd()
    data_input = cwd / "data" / "input"
    if data_input.exists():
        files = sorted(str(path.relative_to(cwd)) for path in data_input.rglob("*") if path.is_file())
        files_text = "\n".join(f"- {path}" for path in files) if files else "<no files>"
        data_input_status = "exists"
    else:
        files_text = "<data/input missing>"
        data_input_status = "missing"
    return "\n".join(
        [
            f"Input file missing: {ids_file}",
            f"Current working directory: {cwd}",
            f"data/input status: {data_input_status}",
            "Files found under data/input:",
            files_text,
        ]
    )


def run_pilot(args: argparse.Namespace) -> dict:
    """Run the full-text parsing pilot and return the quality report."""
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be >= 1")
    if not args.ids_file.exists():
        raise FileNotFoundError(_format_missing_input_message(args.ids_file))

    requested_ids = load_arxiv_id_list_file(args.ids_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    papers_dir = args.output_dir / "processed" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    paper_results = []
    for arxiv_id in requested_ids:
        result = process_arxiv_paper(
            arxiv_id=arxiv_id,
            papers_dir=papers_dir,
            timeout_seconds=args.timeout_seconds,
        )
        paper_results.append(result)

    manifest = build_fulltext_manifest(
        output_dir=args.output_dir,
        ids_file=args.ids_file,
        paper_results=paper_results,
    )
    write_json(manifest, args.output_dir / "fulltext_manifest.json")

    report = build_quality_report(
        requested_ids=requested_ids,
        paper_results=paper_results,
    )
    write_quality_reports(report, args.output_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_pilot(args)
    except (ArxivMetadataError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"KamiKnows full-text parsing pilot failed: {exc}", file=sys.stderr)
        return 1

    print("KamiKnows full-text parsing pilot completed")
    print(f"Requested papers: {report['requested_papers']}")
    print(f"Successfully processed: {report['successfully_processed']}")
    print(f"Partially processed: {report['partially_processed']}")
    print(f"Failed: {report['failed']}")
    print(f"Output directory: {args.output_dir}")
    print(f"Quality report: {args.output_dir / 'parsing_quality_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
