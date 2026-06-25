"""Base interface for replaceable KamiKnows model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelPlugin(ABC):
    """Abstract interface for all text-generation backends.

    Every local or remote LLM backend should implement this small interface.
    Keeping this contract minimal helps KamiKnows swap Qwen, Mistral,
    DeepSeek, API models, or future fine-tuned adapters without changing
    extraction code.
    """

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input instruction or task prompt.
            temperature: Sampling temperature. Use 0.0 for deterministic
                extraction-style tests when supported by the backend.

        Returns:
            Generated text as a plain string.
        """
        raise NotImplementedError
