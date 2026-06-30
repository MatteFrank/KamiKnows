"""Embedding model registry for Phase 2 retrieval experiments."""

from __future__ import annotations

from typing import Any


DEFAULT_EMBEDDING_MODELS: tuple[dict[str, Any], ...] = (
    {
        "model_key": "all-MiniLM-L6-v2",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "provider_library": "sentence-transformers",
        "requires_prefix": False,
        "query_prefix": "",
        "passage_prefix": "",
        "normalize_embeddings": True,
        "max_length": 256,
        "embedding_dimension": 384,
        "notes": "Compact general-purpose sentence-transformers baseline.",
    },
    {
        "model_key": "e5-small-v2",
        "model_name": "intfloat/e5-small-v2",
        "provider_library": "sentence-transformers",
        "requires_prefix": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "normalize_embeddings": True,
        "max_length": 512,
        "embedding_dimension": 384,
        "notes": "E5 models require explicit query/passsage prefixes for retrieval.",
    },
    {
        "model_key": "bge-m3",
        "model_name": "BAAI/bge-m3",
        "provider_library": "sentence-transformers",
        "requires_prefix": False,
        "query_prefix": "",
        "passage_prefix": "",
        "normalize_embeddings": True,
        "max_length": 8192,
        "embedding_dimension": 1024,
        "notes": "Multilingual/multifunction BGE embedding model; heavier than MiniLM/E5.",
    },
)


def get_embedding_registry() -> list[dict[str, Any]]:
    """Return a copy of the default embedding registry."""
    return [dict(model) for model in DEFAULT_EMBEDDING_MODELS]


def get_embedding_model_config(model_name_or_key: str) -> dict[str, Any]:
    """Return one model config by model key or full model name."""
    needle = model_name_or_key.strip()
    for config in DEFAULT_EMBEDDING_MODELS:
        if needle in {config["model_key"], config["model_name"]}:
            return dict(config)
    raise ValueError(f"unknown embedding model: {model_name_or_key}")


def select_embedding_model_configs(model_names_or_keys: list[str] | None) -> list[dict[str, Any]]:
    """Select model configs, preserving caller order."""
    if not model_names_or_keys:
        return get_embedding_registry()
    return [get_embedding_model_config(model_name) for model_name in model_names_or_keys]


def apply_embedding_prefix(config: dict[str, Any], text: str, *, text_kind: str) -> str:
    """Apply model-specific query or passage prefix."""
    if text_kind not in {"query", "passage"}:
        raise ValueError("text_kind must be 'query' or 'passage'")
    prefix_key = "query_prefix" if text_kind == "query" else "passage_prefix"
    prefix = str(config.get(prefix_key) or "")
    return prefix + (text or "")
