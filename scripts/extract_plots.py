"""Inventory figures, classify metadata, or recover explicit native/HEPData numerical data."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.kamiplotextractor import run_classification, run_inventory, run_recovery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, help="existing parse run (inventory/recover)")
    parser.add_argument("--inventory-dir", type=Path, help="inventory v1.0 (classify, or relocated recovery provenance)")
    parser.add_argument("--classified-dir", type=Path, help="classified v1.1 run (recover)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output run directory")
    parser.add_argument("--mode", choices=["inventory", "classify", "recover"], default="inventory")
    parser.add_argument("--source-method", choices=["native", "hepdata", "auto", "compare"], default="native")
    parser.add_argument("--hepdata-dir", type=Path, help="local HEPData root; defaults to mapping file's directory")
    parser.add_argument("--hepdata-map", type=Path, help="explicit JSON/YAML paper-figure-table mapping")
    parser.add_argument("--enable-hepdata-network", action="store_true", help="opt in to bounded HEPData table downloads")
    parser.add_argument("--include-generic", action="store_true", help="recover explicit sources even in generic figures/panels")
    args = parser.parse_args(argv)
    if args.mode == "inventory" and args.parsed_dir is None:
        parser.error("--mode inventory requires --parsed-dir")
    if args.mode == "classify" and args.inventory_dir is None:
        parser.error("--mode classify requires --inventory-dir")
    if args.mode == "recover" and (args.classified_dir is None or args.parsed_dir is None):
        parser.error("--mode recover requires --classified-dir and --parsed-dir")
    if args.mode != "recover":
        if (args.classified_dir or args.hepdata_dir or args.hepdata_map or args.enable_hepdata_network
                or args.include_generic or args.source_method != "native"):
            parser.error("numerical source options require --mode recover")
        if args.parsed_dir and args.inventory_dir:
            parser.error("--parsed-dir and --inventory-dir cannot be combined outside recover mode")
    try:
        if args.mode == "inventory":
            summary = run_inventory(parsed_dir=args.parsed_dir, output_dir=args.output_dir)
        elif args.mode == "classify":
            summary = run_classification(inventory_dir=args.inventory_dir, output_dir=args.output_dir)
        else:
            summary = run_recovery(classified_dir=args.classified_dir, parsed_dir=args.parsed_dir,
                                   output_dir=args.output_dir, inventory_dir=args.inventory_dir,
                                   source_method=args.source_method, hepdata_dir=args.hepdata_dir,
                                   hepdata_map=args.hepdata_map, enable_hepdata_network=args.enable_hepdata_network,
                                   include_generic=args.include_generic)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"extract_plots failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
