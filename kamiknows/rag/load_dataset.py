"""Load RAG-ready dataset v0 files for mini-RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RagDatasetError(RuntimeError):
    """Raised when a RAG-ready dataset cannot be loaded."""


REQUIRED_RAG_READY_FILES = {
    "chunks": "chunks.jsonl",
    "papers": "papers.jsonl",
    "equations": "equations.jsonl",
    "eval_questions": "eval_questions_v0.jsonl",
    "manifest": "rag_manifest_v0.json",
}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RagDatasetError(f"expected JSON object: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL object records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise RagDatasetError(f"{path}:{line_number} is not a JSON object")
            records.append(data)
    return records


def load_rag_ready_dataset(rag_ready_dir: Path) -> dict[str, Any]:
    """Load all required RAG-ready dataset files."""
    rag_ready_dir = Path(rag_ready_dir)
    if not rag_ready_dir.exists():
        raise RagDatasetError(f"RAG-ready directory does not exist: {rag_ready_dir}")

    paths = {
        key: rag_ready_dir / filename
        for key, filename in REQUIRED_RAG_READY_FILES.items()
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RagDatasetError("missing RAG-ready file(s): " + ", ".join(missing))

    chunks = read_jsonl(paths["chunks"])
    papers = read_jsonl(paths["papers"])
    equations = read_jsonl(paths["equations"])
    eval_questions = read_jsonl(paths["eval_questions"])
    manifest = read_json(paths["manifest"])

    return {
        "rag_ready_dir": str(rag_ready_dir),
        "paths": {key: str(path) for key, path in paths.items()},
        "chunks": chunks,
        "papers": papers,
        "equations": equations,
        "eval_questions": eval_questions,
        "manifest": manifest,
        "counts": {
            "papers": len(papers),
            "chunks": len(chunks),
            "equations": len(equations),
            "eval_questions": len(eval_questions),
        },
    }
