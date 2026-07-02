# Phase 2 Status - Robust Scientific RAG

Date: 2026-07-02

## Synthetic Status

Phase 2 is in progress. The current architectural focus is robust scientific
RAG over the fixed FastCaloSimulation corpus, with deterministic retrieval,
chunk traceability, citation hardening, and explicit manual-review boundaries.

Completed Phase 2 tasks:

- P2T1: retrieval baseline v1 plus citation hardening.
- P2T2: embedding model selection v1 as pipeline validation only.
- P2T3: chunking and metadata v1.

Next planned task:

- P2T4: QA grounded v1.

## Current Dataset

Canonical current dataset:

```text
outputs/rag_ready_fastcalo_v1/
```

Counts:

- papers: 5
- chunks: 108
- equations: 28
- eval questions v0: 8

Known data issue:

- 31 chunks have quality flag `very_long_text`.

## Produced Artifacts

P2T1 retrieval baseline artifacts:

```text
outputs/rag_v1_fastcalosim_retrieval/retriever_baseline_v1.jsonl
outputs/rag_v1_fastcalosim_retrieval/retriever_eval_summary_v1.json
outputs/rag_v1_fastcalosim_retrieval/retriever_error_analysis_v1.md
outputs/rag_v1_fastcalosim_retrieval/retrieval_debug_report.md
outputs/rag_v1_fastcalosim_retrieval/retrieval_manifest.json
docs/task_reports/2026-06-29_retriever_baseline_v1.md
```

P2T2 embedding-selection artifacts:

```text
outputs/rag_v1_fastcalosim_embedding_selection/embedding_model_registry_v1.json
outputs/rag_v1_fastcalosim_embedding_selection/embedding_benchmark_results_v1.jsonl
outputs/rag_v1_fastcalosim_embedding_selection/embedding_eval_summary_v1.json
outputs/rag_v1_fastcalosim_embedding_selection/citation_validation_summary_v1.json
outputs/rag_v1_fastcalosim_embedding_selection/embedding_model_comparison_v1.md
docs/task_reports/phase2_task02_embedding_model_selection_v1.md
```

P2T3 chunking and metadata artifacts:

```text
outputs/rag_ready_fastcalo_v1/chunks_v1.jsonl
outputs/rag_ready_fastcalo_v1/chunk_audit_v0_summary.json
outputs/rag_ready_fastcalo_v1/chunk_audit_v0_details.jsonl
outputs/rag_ready_fastcalo_v1/chunk_id_map_v0_to_v1.json
outputs/rag_ready_fastcalo_v1/eval_questions_v0.jsonl
outputs/rag_ready_fastcalo_v1/rag_manifest_v1.json
outputs/rag_ready_fastcalo_v1/retrieval_smoke_summary_v1.json
docs/chunk_schema_v1.md
docs/task_reports/phase2_task03_chunking_metadata_v1.md
```

## Known Metrics

Retrieval smoke test over `chunks_v1`:

- retriever: `tfidf_local_v1`
- top_k: 5
- questions: 8
- expected-source hits: 6 / 8
- expected-source hit rate: 0.750
- misses: `eval_v0_q002`, `eval_v0_q005`

P2T2 embedding-selection note:

- `sentence_transformers` was not installed.
- Candidate embedding models used deterministic hash fallback.
- Results validate pipeline wiring and citation compatibility, not semantic
  embedding quality.

P2T3 quality summary:

- chunks with quality flags: 31
- flag counts: `{"very_long_text": 31}`

## Known Limits

- The corpus is intentionally small and fixed.
- The full-text parser has not been changed during Phase 2.
- Very long chunks were flagged but not split.
- TF-IDF retrieval is a deterministic baseline, not a final retrieval solution.
- Hash fallback embeddings are not a scientific semantic comparison.
- Citation validation checks deterministic identifiers and labels; it does not
  prove factual entailment.
- Manual review remains required before any scientific reliability claim.

## Next Task: P2T4 QA Grounded v1

Goal:

- Build a QA grounded RAG loop over `chunks_v1`.
- Use deterministic citation labels tied to retrieved chunks.
- Compare configured local Ollama generators if available:
  `qwen3:0.6b`, `qwen3:4b`, and `mistral:7b`.
- Record backend errors or unavailable models without failing the whole run.
- Produce Markdown reporting in `docs/task_reports/`.

Non-claims:

- P2T4 should test citation-constrained answer generation.
- P2T4 must not be presented as proof of scientific reliability.
