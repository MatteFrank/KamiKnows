"""Dataset manifest helpers for KamiKnows tutorial outputs.

A dataset manifest is a small machine-readable inventory of files produced by a
run: metadata JSON, extraction JSONL, summaries, benchmark reports, and optional
notes. It does not replace the data files. It makes them easier to audit later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from kamiknows.run_metadata import utc_now_iso
from kamiknows.storage.jsonl import JsonlStorageError, read_jsonl_records

MANIFEST_VERSION = "dataset_manifest_v0"


class DatasetManifestError(RuntimeError):
    """Raised when a dataset manifest cannot be created or validated."""


def compute_file_sha256(path: Path) -> str:
    """Compute a SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_top_level_type(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(data, dict):
        return "object"
    if isinstance(data, list):
        return "list"
    return type(data).__name__


def build_file_entry(path: Path, *, role: str, base_dir: Path | None = None) -> dict[str, Any]:
    """Build a manifest entry for one output/input file.

    Args:
        path: File path to register.
        role: Human-readable role, e.g. ``metadata``, ``jsonl_extractions``.
        base_dir: Optional directory used to also store a relative path.

    Returns:
        A JSON-serializable manifest entry.
    """
    if not role.strip():
        raise DatasetManifestError("file role must be a non-empty string")

    path = Path(path)
    entry: dict[str, Any] = {
        "path": str(path),
        "role": role.strip(),
        "exists": path.exists(),
    }

    if base_dir is not None:
        try:
            entry["relative_path"] = str(path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            entry["relative_path"] = str(path)

    if not path.exists():
        entry.update(
            {
                "size_bytes": None,
                "sha256": None,
                "record_count": None,
                "json_top_level_type": None,
            }
        )
        return entry

    if not path.is_file():
        raise DatasetManifestError(f"manifest path is not a file: {path}")

    entry["size_bytes"] = path.stat().st_size
    entry["sha256"] = compute_file_sha256(path)

    if path.suffix == ".jsonl":
        try:
            entry["record_count"] = len(read_jsonl_records(path))
        except JsonlStorageError as exc:
            raise DatasetManifestError(f"cannot read JSONL file {path}: {exc}") from exc
        entry["json_top_level_type"] = "jsonl"
    elif path.suffix == ".json":
        entry["record_count"] = None
        entry["json_top_level_type"] = _json_top_level_type(path)
    else:
        entry["record_count"] = None
        entry["json_top_level_type"] = None

    return entry


def build_dataset_manifest(
    *,
    name: str,
    files: Iterable[tuple[Path, str]],
    backend: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    prompt_template_sha256: str | None = None,
    extraction_schema_version: str | None = None,
    benchmark_report: Path | None = None,
    notes: str | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a dataset manifest dictionary.

    ``files`` is an iterable of ``(path, role)`` pairs. ``benchmark_report`` can
    be supplied separately for convenience and is included in the files list if
    not already present.
    """
    name = name.strip()
    if not name:
        raise DatasetManifestError("manifest name must be a non-empty string")

    file_pairs = list(files)
    if benchmark_report is not None:
        benchmark_report_path = Path(benchmark_report)
        if all(Path(path) != benchmark_report_path for path, _role in file_pairs):
            file_pairs.append((benchmark_report_path, "benchmark_report"))

    if not file_pairs:
        raise DatasetManifestError("manifest must include at least one file")

    file_entries = [
        build_file_entry(Path(path), role=role, base_dir=base_dir)
        for path, role in file_pairs
    ]
    missing_paths = [entry["path"] for entry in file_entries if not entry["exists"]]

    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": utc_now_iso(),
        "name": name,
        "status": "WARN" if missing_paths else "PASS",
        "scope": "file inventory and formal traceability only; no scientific correctness judgment",
        "run_context": {
            "backend": backend,
            "model": model,
            "prompt_version": prompt_version,
            "prompt_template_sha256": prompt_template_sha256,
            "extraction_schema_version": extraction_schema_version,
        },
        "files": file_entries,
        "missing_paths": missing_paths,
        "notes": notes or "",
    }


def write_dataset_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    """Write a dataset manifest JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def read_dataset_manifest(path: Path) -> dict[str, Any]:
    """Read a dataset manifest JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DatasetManifestError("dataset manifest must be a JSON object")
    return data
