"""Run metadata helpers for traceable KamiKnows outputs.

The extraction JSON describes what the model produced.
Run metadata describes how and when that extraction was produced.
Keeping the two concepts separate makes JSONL outputs easier to audit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from kamiknows.extraction.prompt_registry import (
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
    ABSTRACT_TO_JSON_PROMPT_VERSION,
    EXTRACTION_SCHEMA_VERSION,
)

DEFAULT_PROMPT_VERSION = ABSTRACT_TO_JSON_PROMPT_VERSION
DEFAULT_PROMPT_TEMPLATE_SHA256 = ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256
DEFAULT_EXTRACTION_SCHEMA_VERSION = EXTRACTION_SCHEMA_VERSION


class RunMetadataError(ValueError):
    """Raised when run metadata cannot be created or validated."""


def utc_now_iso() -> str:
    """Return a compact UTC timestamp suitable for JSON records."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def build_run_metadata(
    *,
    backend: str,
    model: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    prompt_template_sha256: str = DEFAULT_PROMPT_TEMPLATE_SHA256,
    extraction_schema_version: str = DEFAULT_EXTRACTION_SCHEMA_VERSION,
) -> dict[str, str]:
    """Create minimal run metadata for one extraction record.

    Args:
        backend: Backend family, for example ``fake`` or ``ollama``.
        model: Concrete model identifier, for example ``fake`` or ``qwen3:0.6b``.
        prompt_version: Human-readable prompt/schema version label.
        prompt_template_sha256: Stable SHA-256 of the prompt template, not of a
            specific filled prompt.
        extraction_schema_version: Version label for the extraction JSON schema.

    Returns:
        A JSON-serializable dictionary with run-level traceability fields.
    """
    backend = backend.strip()
    model = model.strip()
    prompt_version = prompt_version.strip()
    prompt_template_sha256 = prompt_template_sha256.strip()
    extraction_schema_version = extraction_schema_version.strip()

    if not backend:
        raise RunMetadataError("backend must be a non-empty string")
    if not model:
        raise RunMetadataError("model must be a non-empty string")
    if not prompt_version:
        raise RunMetadataError("prompt_version must be a non-empty string")
    if not prompt_template_sha256:
        raise RunMetadataError("prompt_template_sha256 must be a non-empty string")
    if len(prompt_template_sha256) != 64:
        raise RunMetadataError("prompt_template_sha256 must be a 64-character SHA-256 hex digest")
    if not extraction_schema_version:
        raise RunMetadataError("extraction_schema_version must be a non-empty string")

    return {
        "run_id": uuid4().hex,
        "created_at": utc_now_iso(),
        "backend": backend,
        "model": model,
        "prompt_version": prompt_version,
        "prompt_template_sha256": prompt_template_sha256,
        "extraction_schema_version": extraction_schema_version,
    }


def validate_run_metadata(record: dict[str, Any]) -> dict[str, str]:
    """Validate the minimal run metadata shape used in tutorial JSONL records."""
    if not isinstance(record, dict):
        raise RunMetadataError("run metadata must be a dictionary")

    required = (
        "run_id",
        "created_at",
        "backend",
        "model",
        "prompt_version",
        "prompt_template_sha256",
        "extraction_schema_version",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise RunMetadataError(f"missing run metadata keys: {', '.join(missing)}")

    validated: dict[str, str] = {}
    for key in required:
        value = record[key]
        if not isinstance(value, str) or not value.strip():
            raise RunMetadataError(f"run metadata field must be a non-empty string: {key}")
        validated[key] = value

    # Keep validation simple in Fase 0: enough to catch common malformed records
    # without turning run metadata into a heavy schema system.
    if not validated["created_at"].endswith("Z"):
        raise RunMetadataError("created_at must be a UTC ISO timestamp ending with 'Z'")
    if len(validated["prompt_template_sha256"]) != 64:
        raise RunMetadataError("prompt_template_sha256 must be a 64-character SHA-256 hex digest")

    return validated
