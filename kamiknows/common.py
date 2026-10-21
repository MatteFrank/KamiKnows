"""File I/O and run manifests shared by every pipeline step."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "manifest.json"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open_jsonl(path) as write:
        for record in records:
            write(record)


@contextmanager
def open_jsonl(path: Path) -> Iterator:
    """Yield a function that appends one record and flushes it immediately.

    Long runs (generation, chat) keep every completed record even if the job
    is killed halfway.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        def write(record: dict[str, Any]) -> None:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
        yield write


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_output_dir(path: Path) -> Path:
    """Create a new run directory; existing results are never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def write_manifest(
    output_dir: Path,
    *,
    step: str,
    params: dict[str, Any],
    inputs: dict[str, Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Write the manifest that every run directory carries.

    ``inputs`` maps a role ("questions", "chunks", ...) to a file; its sha256
    is recorded so later steps can detect changed inputs.
    """
    manifest = {
        "step": step,
        "created_at": utc_now(),
        "command": [Path(sys.executable).name, *sys.argv],
        "git": git_state(),
        "params": {key: _jsonable(value) for key, value in params.items()},
        "inputs": {
            role: {"path": str(path), "sha256": sha256_file(path)}
            for role, path in inputs.items()
        },
        "summary": summary,
    }
    write_json(Path(output_dir) / MANIFEST_NAME, manifest)
    return manifest


def read_manifest(run_dir: Path, step: str) -> dict[str, Any]:
    """Read a run manifest and check that it comes from the expected step."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"missing {MANIFEST_NAME} in {run_dir}")
    manifest = read_json(path)
    if manifest.get("step") != step:
        raise ValueError(f"{run_dir} is a '{manifest.get('step')}' run, expected '{step}'")
    return manifest


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
