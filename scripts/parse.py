"""Step 1 - parse: download arXiv LaTeX sources and extract sections and equations.

Example:
    python scripts/parse.py --ids-file data/input/arxiv_ids_fastCalo_5fulltext.txt --output-dir outputs/parsed_fastcalo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.parsing import run_parsing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids-file", type=Path, required=True, help="one arXiv id per line, '#' for comments")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="per HTTP request (default: 60)")
    args = parser.parse_args(argv)
    try:
        summary = run_parsing(ids_file=args.ids_file, output_dir=args.output_dir,
                              timeout_seconds=args.timeout_seconds)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"parse failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
