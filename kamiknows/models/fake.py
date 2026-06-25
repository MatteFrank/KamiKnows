"""Deterministic fake model plugins used by Fase 0 tutorials and tests.

Fake models are useful because they exercise the KamiKnows software path without
requiring Ollama, GPU memory, network access, or non-deterministic LLM output.
They are not scientific models and should never be used as evidence quality
checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from kamiknows.models.base import ModelPlugin


@dataclass(slots=True)
class FakeExtractionModel(ModelPlugin):
    """Return a stable valid extraction JSON object.

    The fake model implements the same interface as a real LLM backend, but it
    deliberately ignores the prompt content. This keeps tests and demos stable.
    """

    title: str

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Return deterministic JSON matching the abstract extraction schema."""
        return json.dumps(
            {
                "title": self.title,
                "field": "High Energy Physics / detector simulation",
                "main_claim": (
                    "A parameterized method can reduce calorimeter simulation "
                    "time within validated regions."
                ),
                "method": (
                    "Parameterized shower shapes calibrated on reference "
                    "detector samples."
                ),
                "limitations": (
                    "The method is validated only in limited regions and "
                    "requires further validation on full detector geometries."
                ),
                "confidence": "medium",
            }
        )
