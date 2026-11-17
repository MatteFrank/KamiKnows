"""Prepare, extract and inspect numerical figure evidence. Use <command> --help."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.plot_extractor import (
    run_benchmark,
    run_calibration,
    run_classification,
    run_digitization,
    run_inventory,
    run_recovery,
)
from kamiknows.plot_extractor.validation import run_calibration_review, run_digitization_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, required=True, help="new run directory; never overwritten")

    inventory = commands.add_parser(
        "inventory", parents=[common], help="resolve figures, captions and source graphics"
    )
    inventory.add_argument("--parsed-dir", type=Path, required=True)

    classify = commands.add_parser(
        "classify", parents=[common], help="classify assets with vision or textual evidence"
    )
    classify.add_argument("--inventory-dir", type=Path, required=True)
    classify.add_argument("--parsed-dir", type=Path, required=True)
    classify.add_argument("--classifier", choices=["vision", "text"], default="vision")
    classify.add_argument("--model", default="gemma3:4b", help="Ollama visual model")
    classify.add_argument("--base-url", default="http://127.0.0.1:11434")
    classify.add_argument("--figure-ids", nargs="+")
    classify.add_argument("--asset-ids", nargs="+")
    classify.add_argument("--describe-skipped", action="store_true")
    classify.add_argument("--prepare-only", action="store_true")
    classify.add_argument("--prepared-images-dir", type=Path)

    upstream = argparse.ArgumentParser(add_help=False)
    upstream.add_argument("--parsed-dir", type=Path, required=True)
    upstream.add_argument("--classified-dir", type=Path, required=True)
    upstream.add_argument("--inventory-dir", type=Path, help="upstream inventory if relocated")
    calibrate = commands.add_parser(
        "calibrate", parents=[common, upstream], help="automatically propose axes and generate visual review"
    )
    calibrate.add_argument("--figure-ids", nargs="+")
    calibrate.add_argument("--asset-ids", nargs="+")

    review = commands.add_parser(
        "calibration-review", parents=[common], help="render an existing calibration run"
    )
    review.add_argument("--calibration-dir", type=Path, required=True)

    digitize = commands.add_parser(
        "digitize", parents=[common, upstream], help="extract values using proposed axis mappings"
    )
    digitize.add_argument("--backend", choices=["auto", "pymupdf", "raster"], default="auto")
    digitize.add_argument("--calibration-file", type=Path)
    digitize.add_argument("--recovery-dir", type=Path)
    digitize.add_argument("--selection", choices=["unresolved", "all"], default="unresolved")
    digitize.add_argument("--raster-dpi", type=int, default=144)
    digitize.add_argument("--include-generic", action="store_true")
    digitize.add_argument("--figure-ids", nargs="+")

    review = commands.add_parser(
        "digitization-review", parents=[common], help="source/reconstruction report and numerical exports"
    )
    review.add_argument("--digitized-dir", type=Path, required=True)

    recover = commands.add_parser(
        "recover",
        parents=[common, upstream],
        help="recover original native or explicitly mapped HEPData sources",
    )
    recover.add_argument(
        "--source-method", choices=["native", "hepdata", "auto", "compare"], default="native"
    )
    recover.add_argument("--hepdata-dir", type=Path)
    recover.add_argument("--hepdata-map", type=Path)
    recover.add_argument("--enable-hepdata-network", action="store_true")
    recover.add_argument("--include-generic", action="store_true")

    benchmark = commands.add_parser(
        "benchmark", parents=[common], help="compare extraction against independent numerical evidence"
    )
    benchmark.add_argument("--digitized-dir", type=Path, required=True)
    reference = benchmark.add_mutually_exclusive_group(required=True)
    reference.add_argument("--ground-truth-file", type=Path)
    reference.add_argument("--recovery-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    kwargs = vars(args).copy()
    command = kwargs.pop("command")
    if command == "classify":
        if args.classifier == "text" and (
            args.prepare_only
            or args.describe_skipped
            or args.prepared_images_dir
            or args.figure_ids
            or args.asset_ids
        ):
            parser.error("asset selection and image preparation require --classifier vision")
    if command == "digitize" and not 72 <= args.raster_dpi <= 300:
        parser.error("--raster-dpi must be between 72 and 300")
    functions = {
        "inventory": run_inventory,
        "classify": run_classification,
        "calibrate": run_calibration,
        "calibration-review": run_calibration_review,
        "digitize": run_digitization,
        "digitization-review": run_digitization_review,
        "recover": run_recovery,
        "benchmark": run_benchmark,
    }
    try:
        summary = functions[command](**kwargs)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"extract_plots {command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
