"""Run KamiKnows Phase 2 Task 06 light reranking v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag.load_dataset import RagDatasetError
from kamiknows.rag.reranker_v1 import run_light_reranking_v1

DEFAULT_CONFIG_PATH = Path("configs/phase2_p2t6_light_reranking_v1.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic light reranking over TF-IDF candidates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Light reranking config JSON. Default: {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument("--rag-ready-dir", type=Path, default=None)
    parser.add_argument("--eval-questions-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--candidate-top-n", type=int, default=None)
    parser.add_argument("--final-top-k", type=int, default=None)
    return parser.parse_args(argv)


def load_light_reranking_config(config_path: Path) -> dict[str, Any]:
    """Load a light reranking JSON config."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return data


def resolve_run_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Merge config values with CLI overrides."""
    config = load_light_reranking_config(args.config) if args.config else {}
    input_config = _section(config, "input")
    output_config = _section(config, "output")
    retrieval_config = _section(config, "retrieval")
    reranking_config = _section(config, "reranking")

    rag_ready_dir = args.rag_ready_dir or _path_value(input_config.get("rag_ready_dir"))
    eval_questions_file = args.eval_questions_file or _path_value(input_config.get("eval_questions_file"))
    output_dir = args.output_dir or _path_value(output_config.get("output_dir"))
    report_path = args.report_path or _path_value(output_config.get("report_path"))
    if rag_ready_dir is None:
        raise ValueError("rag_ready_dir is required")
    if eval_questions_file is None:
        raise ValueError("eval_questions_file is required")
    if output_dir is None:
        raise ValueError("output_dir is required")
    if report_path is None:
        raise ValueError("report_path is required")

    return {
        "rag_ready_dir": rag_ready_dir,
        "eval_questions_file": eval_questions_file,
        "output_dir": output_dir,
        "report_path": report_path,
        "config_path": args.config,
        "candidate_top_n": _resolve_int(args.candidate_top_n, retrieval_config.get("candidate_top_n"), 20),
        "final_top_k": _resolve_int(args.final_top_k, retrieval_config.get("final_top_k"), 5),
        "score_weights": reranking_config.get("score_weights", {}),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = resolve_run_settings(args)
        result = run_light_reranking_v1(**settings)
    except (OSError, RagDatasetError, ValueError, json.JSONDecodeError) as exc:
        print(f"KamiKnows light reranking v1 failed: {exc}", file=sys.stderr)
        return 1

    summary = result["eval_summary"]
    print("KamiKnows light reranking v1 completed")
    print(f"Output directory: {settings['output_dir']}")
    print(f"Report: {settings['report_path']}")
    print(f"Questions: {summary['questions_total']}")
    print(f"Candidate top-n: {summary['candidate_top_n']}")
    print(f"Final top-k: {summary['final_top_k']}")
    print(f"Baseline hit@5: {summary['baseline_hit_at_5']:.3f}")
    print(f"Reranked hit@5: {summary['reranked_hit_at_5']:.3f}")
    print(f"Improved: {', '.join(summary['improved_questions']) or '<none>'}")
    print(f"Worsened: {', '.join(summary['worsened_questions']) or '<none>'}")
    return 0


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section must be an object: {key}")
    return value


def _path_value(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value))


def _resolve_int(cli_value: Any, config_value: Any, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    if config_value is not None:
        return int(config_value)
    return default


if __name__ == "__main__":
    raise SystemExit(main())
