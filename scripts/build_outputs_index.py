"""Build docs/outputs_index.md: one row per run directory under outputs/.

The index is read-only with respect to outputs/: it only reads manifests and
run configs. Free-text notes (status, superseded-by, why archived) live in
docs/outputs_index_notes.json so they survive regeneration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR_NAME = "_archive"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def short_name(path: str | None) -> str:
    if not path:
        return "-"
    return Path(path).name


def chunker_label(dataset_dir: Path) -> str:
    """Human label for a rag_ready dataset, derived from its manifest."""
    manifests = sorted(dataset_dir.glob("rag_manifest_v*.json"))
    if not manifests:
        return "word"
    manifest = read_json(manifests[-1])
    chunking = manifest.get("chunking", {})
    strategy = chunking.get("strategy") or ""
    if strategy.startswith("semantic"):
        label = "semantic"
        if chunking.get("cut_policy") == "first_drop":
            label += f" first_drop(min{chunking.get('min_words')})"
    elif strategy.startswith("sentence"):
        label = "sentence"
    else:
        label = "word"
    if manifest.get("section_filter", {}).get("enabled"):
        label += " no-backmatter"
    return label


def chunk_count(dataset_dir: Path) -> str:
    chunks = dataset_dir / "chunks_v1.jsonl"
    if not chunks.exists():
        chunks = dataset_dir / "chunks.jsonl"
    if not chunks.exists():
        return "-"
    with chunks.open(encoding="utf-8") as handle:
        return str(sum(1 for line in handle if line.strip()))


def classify(run_dir: Path) -> str:
    names = {p.name for p in run_dir.iterdir()}
    if "qa_run_config_v1.json" in names or "qa_model_registry_v1.json" in names:
        return "qa"
    if any(n.startswith("rag_manifest_v") for n in names) or run_dir.name.startswith("rag_ready_"):
        return "dataset"
    if "retrieval_comparison_manifest_v1.json" in names or "reranking_manifest_v1.json" in names:
        return "retrieval"
    if run_dir.name.startswith("embedding_selection") or run_dir.name.startswith("retriever_baseline"):
        return "retrieval"
    if run_dir.name.startswith("review_"):
        return "review"
    if names & {"replay_manifest_v1.json", "experiment_manifest_v1.json", "run_manifest.json"}:
        return "replay"
    if run_dir.name == "chat_rag":
        return "chat"
    return "base"


def qa_row(run_dir: Path, datasets_root: Path) -> dict[str, str]:
    config = read_json(run_dir / "qa_run_config_v1.json")
    summary = read_json(run_dir / "rag_eval_summary_v1.json")
    registry = read_json(run_dir / "qa_model_registry_v1.json")
    rag_ready_dir = config.get("rag_ready_dir")
    dataset_dir = datasets_root / short_name(rag_ready_dir)
    models = summary.get("models_attempted") or [
        m.get("model_name") for m in registry.get("models", []) if m.get("enabled")
    ]
    answers = list(run_dir.glob("rag_answers_*.jsonl"))
    if answers and summary:
        status = f"ok {summary.get('answers_succeeded')}/{summary.get('answers_attempted')}"
    elif answers:
        status = "partial (no summary)"
    else:
        status = "INCOMPLETE (no answers)"
    return {
        "chunker": chunker_label(dataset_dir) if dataset_dir.is_dir() else short_name(rag_ready_dir),
        "dataset": short_name(rag_ready_dir),
        "retriever": config.get("retriever", {}).get("name", "-"),
        "source_policy": config.get("source_policy") or "none",
        "eval_set": short_name(config.get("eval_set", {}).get("questions_file")),
        "top_k": str(config.get("top_k", "-")),
        "models": ", ".join(m for m in models if m) or "-",
        "status": status,
        "created": (config.get("created_at") or summary.get("created_at") or "-")[:10],
    }


def retrieval_row(run_dir: Path) -> dict[str, str]:
    comparison = read_json(run_dir / "retrieval_comparison_manifest_v1.json")
    reranking = read_json(run_dir / "reranking_manifest_v1.json")
    if comparison:
        return {
            "kind": "tfidf vs minilm",
            "dataset": short_name(comparison.get("rag_ready_dir")),
            "eval_set": short_name(comparison.get("questions_file")),
            "source_policy": comparison.get("source_policy") or "none",
        }
    if reranking:
        inputs = reranking.get("input_files", {})
        return {
            "kind": "tfidf + light reranker",
            "dataset": short_name(str(Path(inputs.get("chunks", "-")).parent)),
            "eval_set": short_name(inputs.get("eval_questions")),
            "source_policy": "none",
        }
    return {"kind": run_dir.name.split("_v1")[0], "dataset": "-", "eval_set": "-", "source_policy": "-"}


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def build_index(outputs_dir: Path, notes: dict[str, dict[str, str]]) -> str:
    groups: dict[str, list[Path]] = {}
    for run_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        if run_dir.name.startswith("_"):  # _archive, _slurm_logs, ...
            continue
        groups.setdefault(classify(run_dir), []).append(run_dir)

    def note(name: str) -> str:
        return notes.get(name, {}).get("note", "")

    parts = [
        "# Outputs index",
        "",
        "Generated by `python3 scripts/build_outputs_index.py`. Do not edit by hand:",
        "put notes in `docs/outputs_index_notes.json` and regenerate.",
        "",
        "`outputs/` is git-ignored: this file is the versioned record of what each run is.",
        "",
        "## Base data",
        "",
        md_table(["Directory", "Note"], [[f"`{p.name}`", note(p.name)] for p in groups.get("base", [])]),
        "",
        "## Chunked datasets (step 4)",
        "",
        md_table(
            ["Directory", "Chunker", "Chunks", "Note"],
            [[f"`{p.name}`", chunker_label(p), chunk_count(p), note(p.name)] for p in groups.get("dataset", [])],
        ),
        "",
        "## Retrieval runs (step 6)",
        "",
    ]
    retrieval_rows = []
    for p in groups.get("retrieval", []):
        row = retrieval_row(p)
        retrieval_rows.append(
            [f"`{p.name}`", row["kind"], f"`{row['dataset']}`", row["eval_set"], row["source_policy"], note(p.name)]
        )
    parts += [md_table(["Directory", "Kind", "Dataset", "Eval set", "Source policy", "Note"], retrieval_rows), ""]

    qa_rows = []
    for p in sorted(groups.get("qa", []), key=lambda d: qa_row(d, outputs_dir)["created"]):
        row = qa_row(p, outputs_dir)
        qa_rows.append(
            [
                f"`{p.name}`",
                row["created"],
                row["chunker"],
                row["retriever"],
                row["source_policy"],
                row["models"],
                row["eval_set"],
                row["top_k"],
                row["status"],
                note(p.name),
            ]
        )
    parts += [
        "## QA runs: generation + citation validation (steps 7-8)",
        "",
        md_table(
            ["Directory", "Date", "Chunker", "Retriever", "Sources", "Model", "Eval set", "k", "Status", "Note"],
            qa_rows,
        ),
        "",
        "## Prompt replays on saved contexts",
        "",
        md_table(["Directory", "Note"], [[f"`{p.name}`", note(p.name)] for p in groups.get("replay", [])]),
        "",
        "## Manual reviews",
        "",
        md_table(["Directory", "Note"], [[f"`{p.name}`", note(p.name)] for p in groups.get("review", [])]),
        "",
        "## Interactive chat sessions",
        "",
        md_table(["Directory", "Note"], [[f"`{p.name}`", note(p.name)] for p in groups.get("chat", [])]),
        "",
    ]

    archive_dir = outputs_dir / ARCHIVE_DIR_NAME
    if archive_dir.is_dir():
        archived = sorted(archive_dir.iterdir())
        parts += [
            f"## Archived (`outputs/{ARCHIVE_DIR_NAME}/`)",
            "",
            "Not deleted, only moved out of the way. Paths in older task reports still point to `outputs/<name>`.",
            "",
            md_table(["Entry", "Why archived"], [[f"`{p.name}`", note(p.name)] for p in archived]),
            "",
        ]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--notes", type=Path, default=ROOT / "docs" / "outputs_index_notes.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "outputs_index.md")
    args = parser.parse_args()

    if not args.outputs_dir.is_dir():
        print(f"Outputs directory not found: {args.outputs_dir}", file=sys.stderr)
        return 1
    notes = read_json(args.notes) if args.notes.exists() else {}
    args.output.write_text(build_index(args.outputs_dir, notes), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
