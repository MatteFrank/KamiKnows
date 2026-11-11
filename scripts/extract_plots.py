"""Inventory, classify, recover, digitize and benchmark scientific figures."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.kamiplotextractor import (run_benchmark, run_classification, run_digitization,
                                      run_inventory, run_recovery)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, help="existing parse run (inventory/recover/digitize)")
    parser.add_argument("--inventory-dir", type=Path, help="inventory v1.0 (classify, or relocated recovery/digitization provenance)")
    parser.add_argument("--classified-dir", type=Path, help="classified v1.1 run (recover/digitize)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output run directory")
    parser.add_argument("--mode", choices=["inventory", "classify", "recover", "digitize", "benchmark"], default="inventory")
    parser.add_argument("--source-method", choices=["native", "hepdata", "auto", "compare"], default="native")
    parser.add_argument("--hepdata-dir", type=Path, help="local HEPData root; defaults to mapping file's directory")
    parser.add_argument("--hepdata-map", type=Path, help="explicit JSON/YAML paper-figure-table mapping")
    parser.add_argument("--enable-hepdata-network", action="store_true", help="opt in to bounded HEPData table downloads")
    parser.add_argument("--include-generic", action="store_true", help="recover explicit sources even in generic figures/panels")
    parser.add_argument("--backend", choices=["auto", "pymupdf", "plotdigitizer", "chartdetective", "plotsift", "compare"], default="auto")
    parser.add_argument("--calibration-file", type=Path, help="explicit source-coordinate calibration JSON")
    parser.add_argument("--imports-dir", type=Path, help="local external exports and imports.json sidecar")
    parser.add_argument("--recovery-dir", type=Path, help="compatible schema 1.2 original data (digitize/benchmark)")
    parser.add_argument("--selection", choices=["unresolved", "all"], default="unresolved")
    parser.add_argument("--raster-dpi", type=int, default=144, help="PDF overlay rendering DPI, 72–300")
    parser.add_argument("--digitized-dir", type=Path, help="schema 1.3 digitization run (benchmark)")
    parser.add_argument("--ground-truth-file", type=Path, help="independent reference JSON (benchmark)")
    args = parser.parse_args(argv)
    if args.mode == "inventory" and args.parsed_dir is None:
        parser.error("--mode inventory requires --parsed-dir")
    if args.mode == "classify" and args.inventory_dir is None:
        parser.error("--mode classify requires --inventory-dir")
    if args.mode in {"recover", "digitize"} and (args.classified_dir is None or args.parsed_dir is None):
        parser.error(f"--mode {args.mode} requires --classified-dir and --parsed-dir")
    if args.mode == "benchmark" and (args.digitized_dir is None or (args.ground_truth_file is None) == (args.recovery_dir is None)):
        parser.error("benchmark requires --digitized-dir and exactly one of --ground-truth-file / --recovery-dir")
    # Reject even explicitly supplied default-valued options in the wrong mode.
    supplied = {arg.split("=", 1)[0] for arg in (argv if argv is not None else sys.argv[1:]) if arg.startswith("--")}
    allowed = {"--mode", "--output-dir", "--help"}
    allowed |= {
        "inventory": {"--parsed-dir"},
        "classify": {"--inventory-dir"},
        "recover": {"--parsed-dir", "--inventory-dir", "--classified-dir", "--source-method", "--hepdata-dir",
                    "--hepdata-map", "--enable-hepdata-network", "--include-generic"},
        "digitize": {"--parsed-dir", "--inventory-dir", "--classified-dir", "--recovery-dir", "--backend",
                     "--calibration-file", "--imports-dir", "--selection", "--raster-dpi", "--include-generic"},
        "benchmark": {"--digitized-dir", "--ground-truth-file", "--recovery-dir"},
    }[args.mode]
    if supplied - allowed:
        parser.error(f"options not allowed in {args.mode} mode: {', '.join(sorted(supplied - allowed))}")
    if not 72 <= args.raster_dpi <= 300:
        parser.error("--raster-dpi must be between 72 and 300")
    try:
        if args.mode == "inventory":
            summary = run_inventory(parsed_dir=args.parsed_dir, output_dir=args.output_dir)
        elif args.mode == "classify":
            summary = run_classification(inventory_dir=args.inventory_dir, output_dir=args.output_dir)
        elif args.mode == "recover":
            summary = run_recovery(classified_dir=args.classified_dir, parsed_dir=args.parsed_dir,
                                   output_dir=args.output_dir, inventory_dir=args.inventory_dir,
                                   source_method=args.source_method, hepdata_dir=args.hepdata_dir,
                                   hepdata_map=args.hepdata_map, enable_hepdata_network=args.enable_hepdata_network,
                                   include_generic=args.include_generic)
        elif args.mode == "digitize":
            summary = run_digitization(classified_dir=args.classified_dir, parsed_dir=args.parsed_dir,
                                       output_dir=args.output_dir, inventory_dir=args.inventory_dir,
                                       backend=args.backend, calibration_file=args.calibration_file,
                                       imports_dir=args.imports_dir, recovery_dir=args.recovery_dir,
                                       selection=args.selection, raster_dpi=args.raster_dpi,
                                       include_generic=args.include_generic)
        else:
            summary = run_benchmark(digitized_dir=args.digitized_dir, output_dir=args.output_dir,
                                    ground_truth_file=args.ground_truth_file, recovery_dir=args.recovery_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"extract_plots failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
