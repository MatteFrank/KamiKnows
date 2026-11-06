"""Inventory figures in an existing KamiKnows parse run (Task 01 only)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.kamiplotextractor import run_inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, required=True, help="existing parse run")
    parser.add_argument("--output-dir", type=Path, required=True, help="new inventory run directory")
    parser.add_argument("--mode", choices=["inventory"], default="inventory")
    args = parser.parse_args(argv)
    try:
        summary = run_inventory(parsed_dir=args.parsed_dir, output_dir=args.output_dir, mode=args.mode)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"extract_plots failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
