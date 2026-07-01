# KamiKnows Chunk Schema v1

## Purpose

Chunk schema v1 standardizes RAG-ready chunk metadata for the fixed FastCaloSimulation corpus. It improves traceability from normalized chunks back to v0 chunks, papers, sections, and equations without changing the scientific text content.

This schema does not prove answer quality or scientific correctness.

## Required Fields

Each chunk v1 record must include at least:

```json
{
  "chunk_id": "...",
  "paper_id": "...",
  "arxiv_id": "...",
  "title": "...",
  "source_type": "abstract | section | equation_context | table_context | unknown",
  "section": "...",
  "section_normalized": "...",
  "text": "...",
  "text_length_chars": 123,
  "text_length_tokens_est": 45,
  "metadata": {
    "domain": "HEP / FastCaloSimulation / calorimetry",
    "source_file": "...",
    "version": "v1"
  },
  "quality_flags": [],
  "source_refs": {
    "parent_chunk_id": "...",
    "equation_ids": [],
    "paper_record_available": true
  }
}
```

## Field Notes

- `chunk_id`: stable v1 identifier, formatted as `<paper_id>::chunk_v1::<index>`.
- `paper_id`: source paper identifier. If absent in v0 and impossible to infer, `unknown_paper` is used and flagged.
- `arxiv_id`: arXiv source identifier, propagated from `papers.jsonl` when possible.
- `title`: paper title, propagated from `papers.jsonl` when possible.
- `source_type`: one of `abstract`, `section`, `equation_context`, `table_context`, or `unknown`.
- `section`: human-readable section label inherited from v0 `section_heading` where available.
- `section_heading`: compatibility alias for existing retrieval code.
- `section_normalized`: lowercase normalized section key for grouping and filtering.
- `text`: inherited v0 chunk text; v1 does not reparse full text.
- `text_length_chars`: character length of `text`.
- `text_length_tokens_est`: deterministic lexical token estimate using the local retriever tokenizer.
- `metadata.domain`: domain label for this corpus.
- `metadata.source_file`: source chunk file if available.
- `metadata.version`: always `v1`.
- `quality_flags`: diagnostic flags. Chunks are not dropped automatically.
- `source_refs.parent_chunk_id`: original v0 `chunk_id`.
- `source_refs.equation_ids`: equation IDs from the same paper/section when available.
- `source_refs.paper_record_available`: whether the normalized paper ID exists in `papers.jsonl`.

## Quality Flags

Current flags:

- `empty_text`
- `missing_paper_id`
- `missing_arxiv_id`
- `missing_title`
- `missing_section`
- `unknown_source_type`
- `very_short_text`
- `very_long_text`
- `orphan_chunk`
- `duplicate_text_candidate`

Flags are diagnostics for review and downstream filtering. They do not remove chunks and do not automatically determine scientific reliability.

## Compatibility

Chunk v1 preserves the original chunk ID in `source_refs.parent_chunk_id` and writes `chunk_id_map_v0_to_v1.json`. Evaluation questions remain compatible through paper/arXiv IDs. Existing retrievers can still read `title`, `source_type`, `section_heading`, and `text`.
