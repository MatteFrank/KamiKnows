"""Run KamiKnows Phase 2 Task 07 RAG generator benchmark v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.rag.generator_benchmark_v1 import run_rag_generator_benchmark_v1
from kamiknows.rag.load_dataset import RagDatasetError

DEFAULT_CONFIG_PATH = Path("configs/phase2_p2t7_rag_generator_benchmark_v1.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG generator/configuration benchmark v1."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--rag-ready-dir", type=Path, default=None)
    parser.add_argument("--eval-questions-file", type=Path, default=None)
    parser.add_argument("--reranking-output-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--prompt-profiles", nargs="+", default=None)
    parser.add_argument("--context-sources", nargs="+", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    return parser.parse_args(argv)


def load_generator_benchmark_config(config_path: Path) -> dict[str, Any]:
    """Load the P2T7 config JSON."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return data


def resolve_run_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Merge config values and CLI overrides."""
    config = load_generator_benchmark_config(args.config) if args.config else {}
    input_config = _section(config, "input")
    output_config = _section(config, "output")
    contexts_config = _section(config, "contexts")
    generation_config = _section(config, "answer_generation")

    prompt_profiles = config.get("prompt_profiles", [])
    if args.prompt_profiles is not None:
        descriptions = {
            str(profile.get("name")): str(profile.get("description") or "")
            for profile in prompt_profiles
            if isinstance(profile, dict)
        }
        prompt_profiles = [
            {"name": name, "description": descriptions.get(name, "")}
            for name in args.prompt_profiles
        ]

    return {
        "rag_ready_dir": args.rag_ready_dir or _path_value(input_config.get("rag_ready_dir")),
        "eval_questions_file": args.eval_questions_file or _path_value(input_config.get("eval_questions_file")),
        "reranking_output_dir": args.reranking_output_dir or _path_value(input_config.get("reranking_output_dir")),
        "output_dir": args.output_dir or _path_value(output_config.get("output_dir")),
        "report_path": args.report_path or _path_value(output_config.get("report_path")),
        "config_path": args.config,
        "include_tfidf_baseline": _bool(contexts_config.get("include_tfidf_baseline"), True),
        "include_light_reranked": _bool(contexts_config.get("include_light_reranked"), True),
        "fallback_to_tfidf_if_reranking_missing": _bool(
            contexts_config.get("fallback_to_tfidf_if_reranking_missing"),
            True,
        ),
        "top_k": int(contexts_config.get("top_k") or 5),
        "prompt_profiles": prompt_profiles,
        "model_names": args.models or [str(model) for model in generation_config.get("models", [])],
        "parameter_sets": list(generation_config.get("parameter_sets", []) or []),
        "backend": str(generation_config.get("backend") or "ollama"),
        "base_url": args.base_url or str(generation_config.get("base_url") or "http://localhost:11434"),
        "timeout_seconds_override": args.timeout_seconds,
        "context_sources": args.context_sources,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = resolve_run_settings(args)
        _require_settings(settings)
        result = run_rag_generator_benchmark_v1(**settings)
    except (OSError, RagDatasetError, ValueError, json.JSONDecodeError) as exc:
        print(f"KamiKnows RAG generator benchmark v1 failed: {exc}", file=sys.stderr)
        return 1

    summary = result["summary"]
    recommended = result["recommended_config"]
    print("KamiKnows RAG generator benchmark v1 completed")
    print(f"Output directory: {settings['output_dir']}")
    print(f"Report: {settings['report_path']}")
    print(f"Configs: {summary['configs_total']}")
    print(f"Answer records: {summary['answer_records_total']}")
    print(f"Recommended for manual review: {recommended.get('recommended_for_manual_review')}")
    return 0


def _require_settings(settings: dict[str, Any]) -> None:
    for key in ("rag_ready_dir", "eval_questions_file", "output_dir", "report_path"):
        if settings.get(key) is None:
            raise ValueError(f"{key} is required")


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section must be an object: {key}")
    return value


def _path_value(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value))


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    raise ValueError(f"expected boolean value, got {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
