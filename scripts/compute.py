"""Compute: run the expected plan of every question on the extracted data and check the results.

Each question with an ``expected_plan`` annotation is executed with the
deterministic operations of kamiknows.compute, and its results are compared
with ``expected_results`` within a relative tolerance. No language model runs.

Example:
    python scripts/compute.py --questions data/questions/fastcalo_plots_values_plans.jsonl \\
        --digitized-dir outputs/plots_<corpus>_<setting> --output-dir outputs/compute_<evalset>_expected
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.compute.pipeline import run_compute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, required=True,
                        help="JSONL questions with expected_plan and expected_results")
    parser.add_argument("--digitized-dir", type=Path, required=True,
                        help="digitization run of extract_plots.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--rel-tol", type=float, default=0.01,
                        help="relative tolerance when comparing numbers (default: 0.01)")
    args = parser.parse_args(argv)
    try:
        summary = run_compute(questions_file=args.questions, digitized_dir=args.digitized_dir,
                              output_dir=args.output_dir, rel_tol=args.rel_tol)
    except (OSError, ValueError) as exc:
        print(f"compute failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
