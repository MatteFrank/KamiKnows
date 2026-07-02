"""Run KamiKnows Phase 2 Task 05 manual grounding review artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag.load_dataset import RagDatasetError
from kamiknows.rag.manual_grounding_review_v1 import (
    DEFAULT_MANUAL_REVIEW_OUTPUT_DIR,
    DEFAULT_QA_OUTPUT_DIR,
    DEFAULT_REPORT_PATH,
    run_manual_grounding_review_v1,
)

DEFAULT_CONFIG_PATH = Path("configs/phase2_p2t5_manual_review_v1.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Phase 2 Task 05 manual grounding review artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Manual review config JSON. Default: {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument("--qa-output-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--answer-file",
        type=Path,
        action="append",
        default=None,
        help="Answer JSONL file. Can be passed multiple times and overrides config answer_files.",
    )
    return parser.parse_args(argv)


def load_manual_review_config(config_path: Path) -> dict[str, Any]:
    """Load a Phase 2 Task 05 JSON config."""
    if not config_path.exists():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return data


def resolve_run_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Merge config and CLI overrides."""
    config = load_manual_review_config(args.config) if args.config else {}
    input_config = _section(config, "input")
    output_config = _section(config, "output")

    qa_output_dir = (
        args.qa_output_dir
        or _path_value(input_config.get("qa_output_dir"))
        or DEFAULT_QA_OUTPUT_DIR
    )
    output_dir = (
        args.output_dir
        or _path_value(output_config.get("output_dir"))
        or DEFAULT_MANUAL_REVIEW_OUTPUT_DIR
    )
    report_path = (
        args.report_path
        or _path_value(output_config.get("report_path"))
        or DEFAULT_REPORT_PATH
    )
    answer_files = args.answer_file if args.answer_file is not None else input_config.get("answer_files")

    return {
        "qa_output_dir": qa_output_dir,
        "output_dir": output_dir,
        "report_path": report_path,
        "answer_files": answer_files,
        "eval_set_file": _path_value(input_config.get("eval_set_file")),
        "retrieved_contexts_file": _path_value(input_config.get("retrieved_contexts_file")),
        "citation_audit_file": _path_value(input_config.get("citation_audit_file")),
        "citation_validation_summary_file": _path_value(
            input_config.get("citation_validation_summary_file")
        ),
        "rag_eval_summary_file": _path_value(input_config.get("rag_eval_summary_file")),
        "chunks_file": _path_value(input_config.get("chunks_file")),
        "review_schema_options": _section(config, "review_schema_options"),
        "hardening_settings": _section(config, "hardening_settings"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = resolve_run_settings(args)
        result = run_manual_grounding_review_v1(**settings)
    except (OSError, RagDatasetError, ValueError, json.JSONDecodeError) as exc:
        print(f"KamiKnows manual grounding review v1 failed: {exc}", file=sys.stderr)
        return 1

    summary = result["manual_review_summary"]
    print("KamiKnows manual grounding review v1 completed")
    print(f"Output directory: {settings['output_dir']}")
    print(f"Report: {settings['report_path']}")
    print(f"Review records: {summary['total_review_records']}")
    print(f"Pending reviews: {summary['pending_count']}")
    print(f"Models: {', '.join(summary['models'])}")
    print(
        "Retrieval misses: "
        + (", ".join(summary["retrieval_miss_questions"]) or "<none>")
    )
    print(f"Citation validation clean: {summary['citation_validation_clean']}")
    return 0


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section must be an object: {key}")
    return value


def _path_value(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value))


if __name__ == "__main__":
    raise SystemExit(main())
