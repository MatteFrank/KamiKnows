"""Run the KamiKnows quality gate on a dataset manifest.

The gate combines formal traceability from ``dataset_manifest.json`` with manual
review summaries produced by ``summarize_manual_quality_review.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.quality.quality_gate import (
    QualityGateError,
    evaluate_quality_gate_from_files,
)

DEFAULT_MANIFEST_PATH = Path("outputs/model_mini_benchmark/dataset_manifest.json")
DEFAULT_OUTPUT_PATH = Path("outputs/model_mini_benchmark/quality_gate_report.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the KamiKnows ACCEPT/REVISE/REJECT quality gate."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Dataset manifest path. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--manual-review-summary",
        type=Path,
        action="append",
        default=[],
        help=(
            "Manual review summary JSON. Can be repeated. If omitted, the script "
            "discovers files with role manual_review_summary from the manifest."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Quality gate report output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help=(
            "Do not require manual review summaries for ACCEPT. Useful for checking "
            "formal traceability only; not recommended before pilot progression."
        ),
    )
    parser.add_argument(
        "--fail-on-non-accept",
        action="store_true",
        help="Return exit code 1 unless the gate decision is ACCEPT.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        report = evaluate_quality_gate_from_files(
            args.manifest,
            args.manual_review_summary,
            require_manual_review=not args.allow_unreviewed,
            discover_from_manifest=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (QualityGateError, OSError, json.JSONDecodeError) as exc:
        print(f"KamiKnows quality gate failed: {exc}", file=sys.stderr)
        return 1

    print("KamiKnows quality gate")
    print(f"Manifest: {args.manifest}")
    print(f"Decision: {report['decision']}")
    print(f"Report: {args.output}")
    print("Reasons:")
    for reason in report["reasons"]:
        print(f"  - {reason}")

    if args.fail_on_non_accept and report["decision"] != "ACCEPT":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
