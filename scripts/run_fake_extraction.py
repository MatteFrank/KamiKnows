"""Run a visible KamiKnows abstract-to-JSON extraction demo.

This script is intentionally small. It can run in two modes:

1. fake mode (default): deterministic, no external service;
2. ollama mode: calls a local Ollama model such as qwen3:0.6b.

Current Fase 0 flow:

model backend -> abstract_to_json -> schema validation -> JSONL save

Usage from the repository root:

    python scripts/run_fake_extraction.py

Use Qwen through local Ollama:

    python scripts/run_fake_extraction.py --backend ollama --model qwen3:0.6b

Optional output path:

    python scripts/run_fake_extraction.py --output outputs/demo_extractions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the script runnable with: python scripts/run_fake_extraction.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.extraction.abstract_to_json import ExtractionError, abstract_to_json
from kamiknows.models.base import ModelPlugin
from kamiknows.models.fake import FakeExtractionModel
from kamiknows.models.ollama_plugin import OllamaPlugin
from kamiknows.run_metadata import DEFAULT_PROMPT_VERSION, build_run_metadata
from kamiknows.storage.jsonl import DEFAULT_EXTRACTIONS_PATH, append_jsonl_record


DEFAULT_TITLE = "Fast calorimeter simulation for HEP"

DEFAULT_ABSTRACT = (
    "We present a lightweight method for fast calorimeter response simulation "
    "in high energy physics. The approach uses parameterized shower shapes "
    "calibrated on reference detector samples. It reduces inference time "
    "compared with detailed simulation while preserving key observables within "
    "limited validation regions. Further validation is required on full detector "
    "geometries."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal KamiKnows abstract-to-JSON demo."
    )
    parser.add_argument(
        "--backend",
        choices=["fake", "ollama"],
        default="fake",
        help="Model backend to use. Default: fake",
    )
    parser.add_argument(
        "--model",
        default="qwen3:0.6b",
        help="Ollama model name used when --backend ollama. Default: qwen3:0.6b",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL used when --backend ollama.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Default: 0.0",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Paper title to extract from.",
    )
    parser.add_argument(
        "--abstract",
        default=DEFAULT_ABSTRACT,
        help="Paper abstract to extract from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTIONS_PATH,
        help="Destination JSONL file. Default: outputs/extractions.jsonl",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help=(
            "Prompt/schema version label recorded in JSONL run metadata. "
            f"Default: {DEFAULT_PROMPT_VERSION}"
        ),
    )
    return parser.parse_args(argv)


def build_model(args: argparse.Namespace) -> ModelPlugin:
    """Create a model plugin from CLI arguments."""
    if args.backend == "fake":
        return FakeExtractionModel(title=args.title)
    if args.backend == "ollama":
        return OllamaPlugin(model=args.model, base_url=args.base_url)
    raise ValueError(f"unsupported backend: {args.backend}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = build_model(args)

    try:
        extraction = abstract_to_json(
            model=model,
            title=args.title,
            abstract=args.abstract,
            temperature=args.temperature,
        )
        record = {
            "extraction": extraction,
            "run": build_run_metadata(
                backend=args.backend,
                model=args.model if args.backend == "ollama" else "fake",
                prompt_version=args.prompt_version,
            ),
        }
        saved_path = append_jsonl_record(record, args.output)
    except (ExtractionError, RuntimeError, ValueError) as exc:
        print(f"KamiKnows extraction failed: {exc}", file=sys.stderr)
        if args.backend == "ollama":
            print(
                "Hint: check that Ollama is running and the model is installed, "
                f"for example: ollama pull {args.model}",
                file=sys.stderr,
            )
        return 1

    print(f"Backend: {args.backend}")
    if args.backend == "ollama":
        print(f"Model: {args.model}")
    print("Validated extraction JSON:")
    print(json.dumps(extraction, indent=2, ensure_ascii=False))
    print("Run metadata:")
    print(json.dumps(record["run"], indent=2, ensure_ascii=False))
    print(f"\nSaved one JSONL record to: {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
