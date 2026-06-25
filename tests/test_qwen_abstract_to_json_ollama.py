"""Optional real Qwen extraction test through Ollama.

Run explicitly with:

    KAMIKNOWS_RUN_OLLAMA_TEST=1 pytest -s tests/test_qwen_abstract_to_json_ollama.py
"""

from __future__ import annotations

import json
import os

import pytest

from kamiknows.extraction.abstract_to_json import abstract_to_json
from kamiknows.models.ollama_plugin import OllamaPlugin


@pytest.mark.skipif(
    os.getenv("KAMIKNOWS_RUN_OLLAMA_TEST") != "1"
    and os.getenv("KNOWKAMI_RUN_OLLAMA_TEST") != "1",
    reason=(
        "Set KAMIKNOWS_RUN_OLLAMA_TEST=1 to call local Ollama/Qwen. "
        "KNOWKAMI_RUN_OLLAMA_TEST is still accepted as a deprecated alias."
    ),
)
def test_qwen_extracts_abstract_to_valid_json() -> None:
    model = OllamaPlugin(model="qwen3:0.6b")

    result = abstract_to_json(
        model=model,
        title="Fast calorimeter simulation for high energy physics experiments",
        abstract=(
            "We present a lightweight approach for fast calorimeter response "
            "simulation in high energy physics. The method uses parameterized "
            "shower shapes calibrated on reference detector samples. It reduces "
            "inference time compared with detailed simulation while preserving "
            "key observables within limited validation regions. The approach is "
            "intended for early analysis studies and requires further validation "
            "on full detector geometries."
        ),
    )

    print("\nQwen extraction JSON:\n", json.dumps(result, indent=2, ensure_ascii=False))
    assert result["title"]
    assert result["confidence"] in {"low", "medium", "high"}
