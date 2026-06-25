"""Tests for KamiKnows run metadata helpers."""

from __future__ import annotations

import pytest

from kamiknows.run_metadata import (
    DEFAULT_EXTRACTION_SCHEMA_VERSION,
    DEFAULT_PROMPT_TEMPLATE_SHA256,
    DEFAULT_PROMPT_VERSION,
    RunMetadataError,
    build_run_metadata,
    validate_run_metadata,
)


def test_build_run_metadata_contains_traceability_fields() -> None:
    metadata = build_run_metadata(backend="fake", model="fake")

    assert metadata["backend"] == "fake"
    assert metadata["model"] == "fake"
    assert metadata["prompt_version"] == DEFAULT_PROMPT_VERSION
    assert metadata["run_id"]
    assert metadata["created_at"].endswith("Z")
    assert metadata["prompt_template_sha256"] == DEFAULT_PROMPT_TEMPLATE_SHA256
    assert metadata["extraction_schema_version"] == DEFAULT_EXTRACTION_SCHEMA_VERSION


def test_validate_run_metadata_accepts_valid_record() -> None:
    metadata = build_run_metadata(
        backend="ollama",
        model="qwen3:0.6b",
        prompt_version="custom_prompt_v1",
    )

    validated = validate_run_metadata(metadata)

    assert validated["backend"] == "ollama"
    assert validated["model"] == "qwen3:0.6b"
    assert validated["prompt_version"] == "custom_prompt_v1"
    assert validated["prompt_template_sha256"] == DEFAULT_PROMPT_TEMPLATE_SHA256
    assert validated["extraction_schema_version"] == DEFAULT_EXTRACTION_SCHEMA_VERSION


def test_build_run_metadata_rejects_empty_fields() -> None:
    with pytest.raises(RunMetadataError):
        build_run_metadata(backend="", model="fake")

    with pytest.raises(RunMetadataError):
        build_run_metadata(backend="fake", model="")

    with pytest.raises(RunMetadataError):
        build_run_metadata(backend="fake", model="fake", prompt_version="")

    with pytest.raises(RunMetadataError):
        build_run_metadata(
            backend="fake",
            model="fake",
            prompt_template_sha256="not-a-sha",
        )


def test_validate_run_metadata_rejects_missing_keys() -> None:
    with pytest.raises(RunMetadataError):
        validate_run_metadata({"backend": "fake", "model": "fake"})
