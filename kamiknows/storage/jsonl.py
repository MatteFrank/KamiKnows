"""Small JSONL writer for validated KamiKnows outputs.

JSONL means "one JSON object per line". It is a good early format for
KamiKnows because it is simple, append-only, easy to inspect with a text editor,
and compatible with later dataset / fine-tuning workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kamiknows.extraction.abstract_to_json import validate_extraction

DEFAULT_EXTRACTIONS_PATH = Path("outputs/extractions.jsonl")


class JsonlStorageError(RuntimeError):
    """Raised when a JSONL record cannot be saved or loaded."""


def append_jsonl_record(record: dict[str, Any], output_path: str | Path) -> Path:
    """Append one dictionary as one JSON line.

    Args:
        record: One JSON-serializable dictionary.
        output_path: Destination .jsonl file.

    Returns:
        The normalized path where the record was saved.
    """
    if not isinstance(record, dict):
        raise JsonlStorageError("record must be a dictionary")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise JsonlStorageError(f"record is not JSON serializable: {exc}") from exc

    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")

    return path


def append_extraction_jsonl(
    extraction: dict[str, Any],
    output_path: str | Path = DEFAULT_EXTRACTIONS_PATH,
) -> Path:
    """Validate and append one abstract extraction object to JSONL.

    Validation happens before writing, so invalid model outputs are not saved.
    """
    validated = validate_extraction(extraction)
    return append_jsonl_record(validated, output_path)


def read_jsonl_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Read a small JSONL file into memory.

    This helper is intended for tests and tiny tutorial examples, not for large
    production datasets.
    """
    path = Path(input_path)
    if not path.exists():
        raise JsonlStorageError(f"JSONL file does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise JsonlStorageError(
                    f"invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise JsonlStorageError(
                    f"line {line_number} of {path} is not a JSON object"
                )
            records.append(parsed)

    return records
