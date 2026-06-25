"""Tests for prompt version metadata."""

from __future__ import annotations

from importlib import import_module

abstract_to_json_module = import_module("kamiknows.extraction.abstract_to_json")
from kamiknows.extraction.prompt_registry import (
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE,
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
    ABSTRACT_TO_JSON_PROMPT_VERSION,
    EXTRACTION_SCHEMA_VERSION,
    compute_template_sha256,
    get_abstract_to_json_prompt_spec,
)


def test_prompt_template_hash_is_stable_sha256() -> None:
    assert len(ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256) == 64
    assert (
        compute_template_sha256(ABSTRACT_TO_JSON_PROMPT_TEMPLATE)
        == ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256
    )


def test_prompt_spec_matches_abstract_to_json_constants() -> None:
    spec = get_abstract_to_json_prompt_spec()

    assert spec.task == "abstract_to_json"
    assert spec.prompt_version == ABSTRACT_TO_JSON_PROMPT_VERSION
    assert spec.extraction_schema_version == EXTRACTION_SCHEMA_VERSION
    assert spec.prompt_template_sha256 == ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256
    assert abstract_to_json_module.ABSTRACT_TO_JSON_PROMPT_VERSION == spec.prompt_version
    assert (
        abstract_to_json_module.ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256
        == spec.prompt_template_sha256
    )
