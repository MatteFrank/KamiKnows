"""Step 5 - review: build the manual review sheet of one or more generation runs, or summarize it.

Examples:
    python scripts/review.py prepare \\
        --qa-dirs outputs/qa_..._qwen4b_paired_top5 outputs/qa_..._mistral7b_paired_top5 \\
        --output-dir outputs/review_qwen4b_vs_mistral7b_paired

    python scripts/review.py summarize --csv outputs/review_qwen4b_vs_mistral7b_paired/review.csv

Reference answers, required points and evidence come from the questions file.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.review import run_prepare, summarize_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare", help="write review.csv with empty reviewer columns")
    prepare.add_argument("--qa-dirs", type=Path, nargs="+", required=True, help="outputs of generate.py")
    prepare.add_argument("--output-dir", type=Path, required=True, help="new directory for this review")
    summarize = actions.add_parser("summarize", help="count reviewer labels of a completed review.csv")
    summarize.add_argument("--csv", type=Path, required=True, help="review.csv filled in by the reviewers")
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = run_prepare(qa_dirs=args.qa_dirs, output_dir=args.output_dir)
        else:
            result = summarize_review(args.csv)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"review failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
