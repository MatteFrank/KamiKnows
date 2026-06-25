"""Tests for the minimal abstract-to-JSON extraction module."""

from __future__ import annotations

import pytest

from kamiknows.extraction.abstract_to_json import (
    ExtractionError,
    abstract_to_json,
    build_extraction_prompt,
    parse_model_json,
    validate_extraction,
)
from kamiknows.models.base import ModelPlugin


class FakeModel(ModelPlugin):
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.last_prompt = prompt
        return self.response


def test_build_extraction_prompt_contains_title_and_abstract() -> None:
    prompt = build_extraction_prompt(
        title="Fast calorimeter simulation",
        abstract="We test a parameterized detector simulation method.",
    )

    assert "Fast calorimeter simulation" in prompt
    assert "parameterized detector simulation" in prompt
    assert "Return ONLY valid JSON" in prompt


def test_parse_model_json_accepts_fenced_json() -> None:
    output = '''```json
{
  "title": "Fast calorimeter simulation",
  "field": "High Energy Physics",
  "main_claim": "A parameterized method can speed up simulation.",
  "method": "Parameterized shower shapes.",
  "limitations": "Validated only on limited samples.",
  "confidence": "medium"
}
```'''

    parsed = parse_model_json(output)

    assert parsed["title"] == "Fast calorimeter simulation"
    assert parsed["confidence"] == "medium"


def test_validate_extraction_rejects_invalid_confidence() -> None:
    data = {
        "title": "A",
        "field": "HEP",
        "main_claim": "Claim",
        "method": "Method",
        "limitations": "Limitations",
        "confidence": "certain",
    }

    with pytest.raises(ExtractionError):
        validate_extraction(data)


def test_abstract_to_json_with_fake_model() -> None:
    fake_response = '''{
      "title": "Fast calorimeter simulation for HEP",
      "field": "High Energy Physics / detector simulation",
      "main_claim": "A lightweight parameterized method reduces simulation time within validated regions.",
      "method": "Parameterized shower shapes calibrated on reference samples.",
      "limitations": "Requires further validation on full detector geometries.",
      "confidence": "medium"
    }'''
    model = FakeModel(fake_response)

    result = abstract_to_json(
        model=model,
        title="Fast calorimeter simulation for HEP",
        abstract=(
            "We present a lightweight approach for fast calorimeter response "
            "simulation in high energy physics."
        ),
    )

    assert result["confidence"] == "medium"
    assert model.last_prompt is not None
    assert "Do not invent scientific results" in model.last_prompt


def test_parse_model_json_rejects_missing_json() -> None:
    with pytest.raises(ExtractionError):
        parse_model_json("No structured data here.")
