# KamiKnows changelog

## Fase 0 closure - 2026-06-23

### Added

- Minimal modular repository for KamiKnows LLM work.
- `ModelPlugin` interface and `OllamaPlugin` backend.
- Qwen/Ollama smoke tests, skipped by default unless explicitly enabled.
- Abstract-to-JSON extraction with schema validation.
- JSONL persistence and inspection utilities.
- arXiv metadata ingestion from query, explicit IDs, metadata JSON files, and plain-text ID lists.
- Batch extraction runner for metadata lists, arXiv queries, explicit IDs, and ID-list files.
- HEP pilot runner for controlled 10-20 paper abstract-level pilots.
- Formal summary, manual quality checklist, manual review summary, dataset manifest, quality gate, and post-pilot analysis utilities.
- Prompt versioning metadata with prompt template hash and extraction schema version.
- `PROJECT_STATUS.md`, `docs/phase0_closure.md`, and `docs/phase1_handoff.md` for the Fase 1/Codex handoff.

### Changed

- Kept fake backend only as a deterministic software-test backend, not as a model-quality comparator.
- Kept metadata ingestion conceptually separate from LLM interpretation.
- Added `--ids-file` support to selected CLI workflows so controlled paper lists can be reused.
- Cleaned generated caches and removed obsolete field-by-field backend comparison utility.

### Not included yet

- PDF parsing.
- LaTeX source parsing.
- Chunking.
- Embeddings and vector database.
- RAG.
- Fine-tuning or LoRA/QLoRA.
- Automatic scientific correctness judging.
