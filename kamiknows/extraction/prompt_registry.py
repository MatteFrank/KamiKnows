"""Prompt version registry for KamiKnows extraction tasks.

Fase 0 keeps prompt management intentionally small: one extraction prompt,
one version label, one schema version label, and one stable template hash.
This is enough to make JSONL records auditable without introducing a heavy
prompt-management system too early.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

ABSTRACT_TO_JSON_PROMPT_VERSION = "abstract_to_json_v0"
EXTRACTION_SCHEMA_VERSION = "extraction_schema_v0"

ABSTRACT_TO_JSON_PROMPT_TEMPLATE = """You are KnowKami, a scientific extraction assistant.

Extract information from the paper title and abstract.
Return ONLY valid JSON.
Do not use markdown.
Do not add explanations.
Do not invent scientific results beyond the input text.

Required JSON schema:
{schema_json}

Rules:
- Use the original title exactly when possible.
- Keep values concise.
- If the abstract is vague, use confidence "low" or "medium".
- Limitations must describe what is missing, restricted, or not validated.

Input title:
{title}

Input abstract:
{abstract}
"""


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Minimal prompt identity stored with each run record."""

    task: str
    prompt_version: str
    extraction_schema_version: str
    prompt_template_sha256: str


def compute_template_sha256(template: str) -> str:
    """Return a stable SHA-256 hash for a prompt template string."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256 = compute_template_sha256(
    ABSTRACT_TO_JSON_PROMPT_TEMPLATE
)


def get_abstract_to_json_prompt_spec() -> PromptSpec:
    """Return the prompt identity used by the abstract-to-JSON task."""
    return PromptSpec(
        task="abstract_to_json",
        prompt_version=ABSTRACT_TO_JSON_PROMPT_VERSION,
        extraction_schema_version=EXTRACTION_SCHEMA_VERSION,
        prompt_template_sha256=ABSTRACT_TO_JSON_PROMPT_TEMPLATE_SHA256,
    )
