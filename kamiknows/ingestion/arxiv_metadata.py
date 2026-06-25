"""Validation helpers for minimal arXiv-style metadata records.

This module is intentionally separate from the network downloader and from the
CLI scripts. It validates the small metadata shape that KamiKnows Fase 0 expects
before a record reaches the LLM extraction step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArxivMetadataError(RuntimeError):
    """Raised when an arXiv-style metadata record is malformed."""


REQUIRED_ARXIV_METADATA_FIELDS = {
    "arxiv_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "published",
    "url",
}

_STRING_FIELDS = {"arxiv_id", "title", "abstract", "published", "url"}
_LIST_FIELDS = {"authors", "categories"}


def _field_type_name(value: Any) -> str:
    return type(value).__name__


def validate_arxiv_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one minimal arXiv-style metadata record.

    The validator accepts extra fields, for example a tutorial ``note``. It only
    enforces the minimal contract needed by the extraction workflow:

    - required string fields: arxiv_id, title, abstract, published, url
    - required list fields: authors, categories
    - list items must be strings
    - title and abstract must not be empty

    The function returns the same dictionary unchanged when validation passes.
    """
    if not isinstance(record, dict):
        raise ArxivMetadataError("arXiv metadata must be a JSON object/dict")

    missing = sorted(REQUIRED_ARXIV_METADATA_FIELDS - set(record))
    if missing:
        raise ArxivMetadataError(
            f"arXiv metadata is missing required field(s): {', '.join(missing)}"
        )

    for field in sorted(_STRING_FIELDS):
        value = record[field]
        if not isinstance(value, str):
            raise ArxivMetadataError(
                f"arXiv metadata field '{field}' must be a string, "
                f"got {_field_type_name(value)}"
            )

    for field in sorted(_LIST_FIELDS):
        value = record[field]
        if not isinstance(value, list):
            raise ArxivMetadataError(
                f"arXiv metadata field '{field}' must be a list, "
                f"got {_field_type_name(value)}"
            )
        if not all(isinstance(item, str) for item in value):
            raise ArxivMetadataError(
                f"arXiv metadata field '{field}' must contain only strings"
            )

    if not record["title"].strip():
        raise ArxivMetadataError("arXiv metadata field 'title' must not be empty")
    if not record["abstract"].strip():
        raise ArxivMetadataError("arXiv metadata field 'abstract' must not be empty")
    if not record["arxiv_id"].strip():
        raise ArxivMetadataError("arXiv metadata field 'arxiv_id' must not be empty")
    if not record["url"].strip():
        raise ArxivMetadataError("arXiv metadata field 'url' must not be empty")

    return record


def _load_json_file(path: str | Path) -> Any:
    """Load JSON from disk and wrap file/JSON errors as ArxivMetadataError."""
    metadata_path = Path(path)
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except OSError as exc:
        raise ArxivMetadataError(
            f"could not read metadata file {metadata_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArxivMetadataError(
            f"metadata file is not valid JSON: {metadata_path}: {exc}"
        ) from exc



def load_arxiv_id_list_file(path: str | Path) -> list[str]:
    """Load a plain-text list of arXiv IDs or arXiv abs/pdf URLs.

    The file format is intentionally simple for controlled pilots:

    - one arXiv ID, ``arXiv:...`` reference, or arXiv URL per line;
    - blank lines are ignored;
    - lines starting with ``#`` are ignored;
    - trailing comments after ``#`` are ignored.

    The returned values are not fetched here. They are passed later to the
    arXiv ingestion functions, which normalize common arXiv ID forms.
    """
    ids_path = Path(path)
    try:
        lines = ids_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArxivMetadataError(
            f"could not read arXiv ID list file {ids_path}: {exc}"
        ) from exc

    arxiv_ids: list[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        value = stripped.split("#", maxsplit=1)[0].strip()
        if value:
            arxiv_ids.append(value)

    if not arxiv_ids:
        raise ArxivMetadataError(
            f"arXiv ID list file contains no IDs or URLs: {ids_path}"
        )
    return arxiv_ids

def load_arxiv_metadata_file(path: str | Path) -> dict[str, Any]:
    """Load and validate one offline arXiv-style metadata JSON file.

    The file must contain one JSON object, not a list of many records. This keeps
    the Fase 0 offline tutorial deterministic and easy to inspect.
    """
    record = _load_json_file(path)
    return validate_arxiv_metadata(record)


def validate_arxiv_metadata_list(records: Any) -> list[dict[str, Any]]:
    """Validate a non-empty list of arXiv-style metadata records.

    This keeps the list contract reusable by both local JSON loading and remote
    metadata download scripts. The function accepts extra fields inside each
    record, but every record must satisfy ``validate_arxiv_metadata``.
    """
    if not isinstance(records, list):
        raise ArxivMetadataError("metadata batch file must contain a JSON list")
    if not records:
        raise ArxivMetadataError("metadata batch file must contain at least one record")

    validated_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        try:
            validated_records.append(validate_arxiv_metadata(record))
        except ArxivMetadataError as exc:
            raise ArxivMetadataError(
                f"invalid metadata record at index {index}: {exc}"
            ) from exc

    return validated_records


def load_arxiv_metadata_list_file(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a small offline list of arXiv-style metadata records.

    This helper is intended for tiny Fase 0 batch demos, not for large corpus
    processing. The JSON file must contain a non-empty list of objects.
    """
    records = _load_json_file(path)
    return validate_arxiv_metadata_list(records)


def write_arxiv_metadata_list_file(
    records: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Validate and write a JSON list of arXiv-style metadata records.

    This is the persistence step for the ingestion phase. It deliberately writes
    metadata only: no LLM extraction, no interpretation, no JSONL output.
    """
    validated_records = validate_arxiv_metadata_list(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validated_records, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output_path
