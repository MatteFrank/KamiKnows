"""Summarize a completed KamiKnows manual quality checklist.

This script reads a Markdown checklist produced by
``create_manual_quality_checklist.py`` after a human reviewer has edited it. It
produces a small machine-readable JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.quality.manual_review import (
    ManualReviewError,
    summarize_manual_quality_checklist_file,
)

DEFAULT_OUTPUT_PATH = Path("outputs/manual_quality_review_summary.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a completed KamiKnows manual quality checklist."
    )
    parser.add_argument(
        "checklist_path",
        type=Path,
        help="Path to a completed Markdown manual quality checklist.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination JSON summary. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--fail-on-non-pass",
        action="store_true",
        help="Return exit code 1 unless the manual review summary status is PASS.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        summary = summarize_manual_quality_checklist_file(args.checklist_path)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (ManualReviewError, OSError, json.JSONDecodeError) as exc:
        print(f"KamiKnows manual review summary failed: {exc}", file=sys.stderr)
        return 1

    print("Manual quality review summary")
    print(f"Checklist: {args.checklist_path}")
    print(f"Records: {summary['total_records']}")
    print(f"Status: {summary['status']}")
    print("Outcomes:")
    for outcome, count in summary["outcome_counts"].items():
        print(f"  {outcome}: {count}")
    print(f"JSON summary: {args.json_output}")

    if args.fail_on_non_pass and summary["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
