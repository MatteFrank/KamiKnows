"""Model backend plugins for KamiKnows."""

from kamiknows.models.base import ModelPlugin
from kamiknows.models.ollama_plugin import OllamaPlugin

__all__ = ["ModelPlugin", "OllamaPlugin"]
