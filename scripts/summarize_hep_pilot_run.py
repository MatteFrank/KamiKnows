"""Summarize a completed KamiKnows HEP pilot run.

This script is intended after running the pilot, manual review and quality gate.
It reads ``dataset_manifest.json`` plus an optional ``quality_gate_report.json``
and writes a compact operational post-pilot analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.pilot.post_pilot_analysis import (  # noqa: E402
    PostPilotAnalysisError,
    build_post_pilot_analysis_from_manifest,
    write_post_pilot_analysis,
)

DEFAULT_MANIFEST = Path("outputs/hep_pilot/dataset_manifest.json")
DEFAULT_OUTPUT = Path("outputs/hep_pilot/post_pilot_analysis.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a completed KamiKnows HEP pilot run."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Dataset manifest path. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--quality-gate-report",
        type=Path,
        help=(
            "Optional quality_gate_report.json. If omitted, the script tries "
            "quality_gate_report.json next to the manifest."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Post-pilot analysis output path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        report = build_post_pilot_analysis_from_manifest(
            args.manifest,
            quality_gate_report_path=args.quality_gate_report,
        )
        write_post_pilot_analysis(report, args.output)
    except (PostPilotAnalysisError, OSError) as exc:
        print(f"KamiKnows post-pilot analysis failed: {exc}", file=sys.stderr)
        return 1

    print("KamiKnows HEP post-pilot analysis")
    print(f"Manifest: {args.manifest}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Report: {args.output}")
    print(f"Manifest status: {report['manifest']['status']}")
    print(f"JSONL records: {report['artifacts']['jsonl_record_count_from_manifest']}")
    if report.get("quality_gate"):
        print(f"Quality gate: {report['quality_gate']['decision']}")
    else:
        print("Quality gate: <not found>")
    print("Next actions:")
    for action in report["next_actions"]:
        print(f"  - {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
