"""Minimal abstract-to-JSON extraction module.

This module is intentionally small. It does not download papers, parse PDFs,
or build a RAG pipeline. It only converts one title + abstract into one
validated JSON object through the ModelPlugin interface.
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import ValidationError, validate

from kamiknows.extraction.prompt_registry import (
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE,
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
    ABSTRACT_TO_JSON_PROMPT_VERSION,
    EXTRACTION_SCHEMA_VERSION,
)
from kamiknows.models.base import ModelPlugin


class ExtractionError(RuntimeError):
    """Raised when model output cannot be parsed or validated."""


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title",
        "field",
        "main_claim",
        "method",
        "limitations",
        "confidence",
    ],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "field": {"type": "string", "minLength": 1},
        "main_claim": {"type": "string", "minLength": 1},
        "method": {"type": "string", "minLength": 1},
        "limitations": {"type": "string", "minLength": 1},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}


def build_extraction_prompt(title: str, abstract: str) -> str:
    """Build a strict prompt for a title + abstract extraction task."""
    title = title.strip()
    abstract = abstract.strip()
    if not title:
        raise ValueError("title must not be empty")
    if not abstract:
        raise ValueError("abstract must not be empty")

    schema_example = {
        "title": "string",
        "field": "string",
        "main_claim": "string",
        "method": "string",
        "limitations": "string",
        "confidence": "low | medium | high",
    }

    return ABSTRACT_TO_JSON_PROMPT_TEMPLATE.format(
        schema_json=json.dumps(schema_example, indent=2, ensure_ascii=False),
        title=title,
        abstract=abstract,
    )


def _remove_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _find_first_json_object(text: str) -> str:
    """Return the first balanced JSON object found in text."""
    start = text.find("{")
    if start == -1:
        raise ExtractionError("model output does not contain a JSON object")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ExtractionError("model output contains an incomplete JSON object")


def parse_model_json(model_output: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""
    if not isinstance(model_output, str) or not model_output.strip():
        raise ExtractionError("model output is empty")

    cleaned = _remove_markdown_fences(model_output)
    json_text = _find_first_json_object(cleaned)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"model output is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ExtractionError("model output JSON is not an object")

    return parsed


def validate_extraction(data: dict[str, Any]) -> dict[str, Any]:
    """Validate one extracted JSON object and return it unchanged."""
    try:
        validate(instance=data, schema=EXTRACTION_SCHEMA)
    except ValidationError as exc:
        raise ExtractionError(f"extraction JSON does not match schema: {exc.message}") from exc
    return data


def abstract_to_json(
    model: ModelPlugin,
    title: str,
    abstract: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run title + abstract extraction through a replaceable model plugin."""
    prompt = build_extraction_prompt(title=title, abstract=abstract)
    raw_output = model.generate(prompt, temperature=temperature)
    parsed = parse_model_json(raw_output)
    return validate_extraction(parsed)
