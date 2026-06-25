"""Minimal Qwen smoke test through the Ollama plugin.

Run explicitly with:

    KAMIKNOWS_RUN_OLLAMA_TEST=1 pytest -s tests/test_qwen_ollama_basic.py

The environment flag avoids failing CI or machines where Ollama is not yet
installed. This is still a real local model call when enabled.
"""

from __future__ import annotations

import os

import pytest

from kamiknows.models.ollama_plugin import OllamaPlugin


@pytest.mark.skipif(
    os.getenv("KAMIKNOWS_RUN_OLLAMA_TEST") != "1"
    and os.getenv("KNOWKAMI_RUN_OLLAMA_TEST") != "1",
    reason=(
        "Set KAMIKNOWS_RUN_OLLAMA_TEST=1 to call local Ollama/Qwen. "
        "KNOWKAMI_RUN_OLLAMA_TEST is still accepted as a deprecated alias."
    ),
)
def test_qwen_ollama_generates_short_response() -> None:
    model = OllamaPlugin(model="qwen3:0.6b")
    prompt = (
        "You are KnowKami, a scientific extraction assistant. "
        "Answer in one short sentence: what is a scientific abstract?"
    )

    response = model.generate(prompt, temperature=0.0)

    print("\nQwen response:\n", response)
    assert isinstance(response, str)
    assert response.strip()
