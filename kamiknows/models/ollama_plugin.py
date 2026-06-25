"""Ollama implementation of the KamiKnows model plugin interface."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from kamiknows.models.base import ModelPlugin


@dataclass(slots=True)
class OllamaPlugin(ModelPlugin):
    """Text-generation backend that calls a local Ollama server.

    Ollama must be installed separately and the selected model must already
    be available locally, for example:

        ollama pull qwen3:0.6b
        ollama serve
    """

    model: str = "qwen3:0.6b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 120

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text using Ollama's /api/generate endpoint."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama request failed. Check that Ollama is running and "
                f"that model '{self.model}' is installed."
            ) from exc

        data = response.json()
        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama returned an empty or invalid response")

        return text.strip()
