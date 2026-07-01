# Phase 2 Task 03 - Chunking and Metadata v1

Date: 2026-07-01

## Objective

Build a traceable RAG-ready chunk metadata v1 dataset from the existing FastCaloSimulation RAG-ready v0 dataset, without scaling the corpus and without changing the full-text parser.

Chunking and metadata v1 improves traceability and retrieval readiness, but does not by itself prove scientific answer quality.

## Inputs Used

```text
outputs/rag_ready_fastcalo_v0/chunks.jsonl
outputs/rag_ready_fastcalo_v0/papers.jsonl
outputs/rag_ready_fastcalo_v0/equations.jsonl
outputs/rag_ready_fastcalo_v0/eval_questions_v0.jsonl
outputs/rag_ready_fastcalo_v0/rag_manifest_v0.json
```

The original `outputs/rag_ready_fastcalo_v0/` directory was not overwritten.

## Outputs Produced

```text
outputs/rag_ready_fastcalo_v1/chunks_v1.jsonl
outputs/rag_ready_fastcalo_v1/chunk_audit_v0_summary.json
outputs/rag_ready_fastcalo_v1/chunk_audit_v0_details.jsonl
outputs/rag_ready_fastcalo_v1/chunk_id_map_v0_to_v1.json
outputs/rag_ready_fastcalo_v1/rag_manifest_v1.json
outputs/rag_ready_fastcalo_v1/eval_questions_v0.jsonl
outputs/rag_ready_fastcalo_v1/retrieval_smoke_summary_v1.json
docs/chunk_schema_v1.md
docs/task_reports/phase2_task03_chunking_metadata_v1.md
```

Build command:

```bash
python3 scripts/build_rag_ready_chunk_metadata_v1.py \
  --input-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_ready_fastcalo_v1
```

## Chunk Counts

```text
chunks v0: 108
chunks v1: 108
papers: 5
equations: 28
eval questions: 8
```

No chunk was silently dropped.

## Metadata Issues Found

Audit summary:

```text
missing paper_id: 0
missing arxiv_id: 0
missing title: 0
missing section: 0
unknown source_type: 0
orphan chunks: 0
empty chunks: 0
duplicate text candidates: 0
very short chunks: 0
very long chunks: 31
```

All v0 chunks had paper IDs, arXiv IDs, titles, section labels, valid source type, and matching paper records.

## Chunk Quality

Text length in characters:

```text
min: 110
mean: 2001.56
median: 2143
max: 3328
```

All chunks are `source_type: section`. The only quality flag raised in the real corpus is `very_long_text`, affecting 31 chunks. This suggests the next retrieval-readiness improvement should consider chunk-size policy or section subdivision, but this task intentionally did not reparse or split full text.

## Traceability Quality

Each chunk v1 now includes:

- stable v1 `chunk_id`, formatted as `<paper_id>::chunk_v1::<index>`;
- original v0 chunk ID in `source_refs.parent_chunk_id`;
- `chunk_id_map_v0_to_v1.json`;
- normalized `section_normalized`;
- deterministic `text_length_chars` and `text_length_tokens_est`;
- `metadata.version: v1`;
- equation IDs from the same paper/section when available;
- `paper_record_available` for orphan detection.

## Main Quality Flags

Real corpus flag counts:

```text
very_long_text: 31
```

The flag is diagnostic only. Flagged chunks remain in `chunks_v1.jsonl` so manual review and later retrieval experiments can decide whether to split, keep, or filter them.

## Eval Questions Compatibility

The existing eval set was copied to:

```text
outputs/rag_ready_fastcalo_v1/eval_questions_v0.jsonl
```

The current eval questions use expected arXiv source IDs, so compatibility is preserved. A v0-to-v1 chunk ID map was still written for future compatibility:

```text
outputs/rag_ready_fastcalo_v1/chunk_id_map_v0_to_v1.json
```

## Retrieval Smoke Test

Smoke test:

```text
retriever: tfidf_local_v1
top_k: 5
questions: 8
chunks: 108
expected-source hits: 6 / 8
expected-source hit rate: 0.750
misses: eval_v0_q002, eval_v0_q005
status: PASS
```

This verifies that `chunks_v1.jsonl` is usable by the existing retriever. It does not prove retrieval improvement because the scientific text content was intentionally preserved.

## What Improved

- Chunk records now have an explicit v1 schema.
- v0 chunk IDs are traceable through `source_refs.parent_chunk_id`.
- A stable v0-to-v1 ID map exists.
- Metadata fields are normalized and propagated from `papers.jsonl` when needed.
- Quality flags make chunk issues visible instead of silently discarding records.
- The dataset has a v1 manifest with quality summary and known limits.
- Retrieval smoke testing is part of the build.

## What Is Not Yet Demonstrated

- Scientific answer quality.
- Better retrieval than TF-IDF v1 baseline.
- Better chunk boundaries.
- Robust semantic retrieval.
- Generation grounding.
- Citation support correctness beyond deterministic chunk traceability.

## Limits

- Full-text parsing was not changed.
- Chunk text was inherited from v0.
- Very long chunks were flagged, not split.
- Duplicate detection is exact normalized-text candidate detection only.
- The smoke test checks usability, not scientific validity.
- Manual review remains required.

## Recommended Next Step

Use `chunks_v1.jsonl` as the new traceable input for a controlled retrieval comparison. The next practical step is to evaluate whether splitting only the `very_long_text` chunks improves expected-source recall without harming citation traceability.
