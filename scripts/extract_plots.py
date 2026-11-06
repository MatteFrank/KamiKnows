"""Inventory figures from a parse run or classify an existing figure inventory."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.kamiplotextractor import run_classification, run_inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parsed-dir", type=Path, help="existing parse run (inventory mode)")
    source.add_argument("--inventory-dir", type=Path, help="existing inventory v1.0 run (classify mode)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output run directory")
    parser.add_argument("--mode", choices=["inventory", "classify"], default="inventory")
    args = parser.parse_args(argv)
    if args.mode == "inventory" and args.parsed_dir is None:
        parser.error("--mode inventory requires --parsed-dir")
    if args.mode == "classify" and args.inventory_dir is None:
        parser.error("--mode classify requires --inventory-dir")
    try:
        if args.mode == "inventory":
            summary = run_inventory(parsed_dir=args.parsed_dir, output_dir=args.output_dir)
        else:
            summary = run_classification(inventory_dir=args.inventory_dir, output_dir=args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"extract_plots failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
