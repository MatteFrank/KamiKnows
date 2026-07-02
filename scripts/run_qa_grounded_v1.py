"""Run KamiKnows Phase 2 QA grounded v1."""

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
from kamiknows.rag.qa_grounded_v1 import DEFAULT_BASE_URL, DEFAULT_TOP_K, run_qa_grounded_v1

DEFAULT_CONFIG_PATH = Path("configs/phase2_p2t4_qa_grounded_v1.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run citation-constrained QA grounded v1 over chunks_v1."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Optional QA grounded v1 config JSON. Canonical path: {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument("--rag-ready-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Ollama model names to attempt. Defaults to qwen3:0.6b qwen3:4b-instruct mistral:7b.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--static-v1-questions-path", type=Path, default=None)
    return parser.parse_args(argv)


def load_qa_grounded_config(config_path: Path) -> dict[str, Any]:
    """Load a QA grounded v1 JSON config."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return data


def resolve_run_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Merge optional JSON config with explicit CLI overrides."""
    config = load_qa_grounded_config(args.config) if args.config else {}
    input_config = _section(config, "input")
    output_config = _section(config, "output")
    retrieval_config = _section(config, "retrieval")
    generation_config = _section(config, "answer_generation")
    eval_config = _section(config, "eval_set")

    rag_ready_dir = args.rag_ready_dir or _path_value(input_config.get("rag_ready_dir"))
    output_dir = args.output_dir or _path_value(output_config.get("output_dir"))
    report_path = (
        args.report_path
        or _path_value(output_config.get("report_path"))
        or Path("docs/task_reports/phase2_task04_qa_grounded_v1.md")
    )
    if rag_ready_dir is None:
        raise ValueError("--rag-ready-dir is required when config input.rag_ready_dir is not set")
    if output_dir is None:
        raise ValueError("--output-dir is required when config output.output_dir is not set")

    models = args.models if args.models is not None else generation_config.get("models")
    if models is not None:
        models = [str(model) for model in models]

    return {
        "rag_ready_dir": rag_ready_dir,
        "output_dir": output_dir,
        "report_path": report_path,
        "model_names": models,
        "top_k": _resolve_number(args.top_k, retrieval_config.get("top_k"), DEFAULT_TOP_K, int),
        "base_url": args.base_url or str(generation_config.get("base_url") or DEFAULT_BASE_URL),
        "temperature": _resolve_number(args.temperature, generation_config.get("temperature"), 0.0, float),
        "top_p": _resolve_number(args.top_p, generation_config.get("top_p"), 0.9, float),
        "num_predict": _resolve_number(args.num_predict, generation_config.get("num_predict"), 512, int),
        "timeout_seconds": _resolve_number(
            args.timeout_seconds,
            generation_config.get("timeout_seconds"),
            120,
            int,
        ),
        "eval_set_mode": str(eval_config.get("mode") or "v0_plus_static_v1"),
        "static_v1_questions_enabled": _resolve_bool(
            eval_config.get("static_v1_questions_enabled"),
            True,
        ),
        "static_v1_questions_path": (
            args.static_v1_questions_path
            or _path_value(eval_config.get("static_v1_questions_path"))
        ),
        "llm_question_generation_enabled": _resolve_bool(
            eval_config.get("llm_question_generation_enabled"),
            False,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = resolve_run_settings(args)
        result = run_qa_grounded_v1(
            rag_ready_dir=settings["rag_ready_dir"],
            output_dir=settings["output_dir"],
            report_path=settings["report_path"],
            model_names=settings["model_names"],
            top_k=settings["top_k"],
            base_url=settings["base_url"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            num_predict=settings["num_predict"],
            timeout_seconds=settings["timeout_seconds"],
            eval_set_mode=settings["eval_set_mode"],
            static_v1_questions_enabled=settings["static_v1_questions_enabled"],
            static_v1_questions_path=settings["static_v1_questions_path"],
            llm_question_generation_enabled=settings["llm_question_generation_enabled"],
        )
    except (OSError, RagDatasetError, ValueError, requests_error()) as exc:
        print(f"KamiKnows QA grounded v1 failed: {exc}", file=sys.stderr)
        return 1

    summary = result["eval_summary"]
    print("KamiKnows QA grounded v1 completed")
    print(f"Output directory: {settings['output_dir']}")
    print(f"Report: {settings['report_path']}")
    print(f"Questions: {summary['questions_total']}")
    print(f"Top-k: {summary['top_k']}")
    print(f"Models attempted: {', '.join(summary['models_attempted'])}")
    print(f"Models available: {', '.join(summary['models_available']) or '<none>'}")
    print(f"Answers succeeded: {summary['answers_succeeded']}")
    print(f"Backend errors: {summary['backend_errors']}")
    print(f"Expected-source hit rate: {summary['expected_source_hit_rate']:.3f}")
    print(f"Invalid citation records: {summary['invalid_citation_count']}")
    return 0


def requests_error() -> type[Exception]:
    try:
        import requests
    except ModuleNotFoundError:
        return RuntimeError
    return requests.RequestException


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section must be an object: {key}")
    return value


def _path_value(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value))


def _resolve_number(cli_value: Any, config_value: Any, default: Any, coerce) -> Any:
    if cli_value is not None:
        return coerce(cli_value)
    if config_value is not None:
        return coerce(config_value)
    return default


def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"expected boolean config value, got: {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
