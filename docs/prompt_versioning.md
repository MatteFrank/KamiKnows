# KamiKnows prompt versioning

This note documents the current Fase 0 prompt identity system.

## Why this exists

A JSONL extraction record is not traceable if it only says which model produced it.
For reproducibility, each record should also say which prompt template and schema family were used.

KamiKnows therefore records three prompt-related fields in `run`:

```json
{
  "prompt_version": "abstract_to_json_v0",
  "prompt_template_sha256": "...64 hex characters...",
  "extraction_schema_version": "extraction_schema_v0"
}
```

## What each field means

`prompt_version` is the human-readable version label.

`prompt_template_sha256` is the SHA-256 hash of the prompt template, before title and abstract are inserted. It changes if the template wording changes.

`extraction_schema_version` is the version label for the required extraction JSON shape.

## Current implementation

Prompt identity is centralized in:

```text
kamiknows/extraction/prompt_registry.py
```

The extraction prompt is rendered by:

```text
kamiknows/extraction/abstract_to_json.py
```

Run metadata is created by:

```text
kamiknows/run_metadata.py
```

The current extraction schema is still minimal:

```json
{
  "title": "string",
  "field": "string",
  "main_claim": "string",
  "method": "string",
  "limitations": "string",
  "confidence": "low | medium | high"
}
```

## What this does not solve yet

This does not prove scientific correctness.

It only makes records auditable at engineering level:

```text
which model
which backend
which prompt version
which prompt template hash
which extraction schema version
when the record was produced
```

## When to create a new prompt version

Create a new version when any of these change materially:

- task instruction wording;
- required JSON fields;
- field definitions;
- refusal or uncertainty rules;
- prompt examples;
- extraction schema semantics.

For small typo fixes that do not affect behavior, still update the template hash automatically, but decide case by case whether the human-readable version label should also change.
