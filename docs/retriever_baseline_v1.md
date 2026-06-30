# KamiKnows Fase 2.1 Retriever Baseline v1

## Goal

Fase 2.1 adds a retrieval-only baseline for the small FastCaloSimulation RAG corpus. The goal is to measure retrieval quality more explicitly before changing answer generation.

This is still a small scientific RAG scaffold. It does not claim scientific correctness, does not expand the corpus, and does not add fine-tuning, embeddings, vector databases, or external services.

## Why v0 Was Insufficient

Mini-RAG v0 proved that local retrieval and optional Ollama generation could run end to end, but its retrieval evidence was too coarse for debugging:

- it reported one expected-source hit rate for one top-k value;
- misses were counted but not explained;
- question-type behavior was not broken out;
- generated citations were not structured enough to validate automatically;
- manual review still had to inspect citations and grounding from loose answer text.

## Input Dataset

The baseline reads the existing Fase 1G dataset:

```text
outputs/rag_ready_fastcalo_v0/
  chunks.jsonl
  papers.jsonl
  equations.jsonl
  eval_questions_v0.jsonl
  rag_manifest_v0.json
```

The current dataset has 5 papers, 108 chunks, 28 equations, and 8 evaluation questions.

## Command

```bash
python3 scripts/run_retriever_baseline_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_v1_fastcalosim_retrieval \
  --top-k-values 3 5 8
```

The CLI also accepts `--rag-v0-summary`. If omitted, it uses `outputs/rag_v0_fastcalosim/rag_eval_summary.json` when that file exists.

## Output Layout

```text
outputs/rag_v1_fastcalosim_retrieval/
  retriever_baseline_v1.jsonl
  retriever_eval_summary_v1.json
  retriever_error_analysis_v1.md
  retrieval_debug_report.md
  retrieval_run_config.json
  retrieval_manifest.json
```

## Retriever

The baseline retriever is `tfidf_local_v1`, a deterministic standard-library TF-IDF retriever. It keeps the v0 local baseline spirit while adding:

- improved token cleanup;
- stopword filtering;
- title and section-heading weighting;
- deterministic question-type and expected-keyword query expansion;
- expected-source diagnostics for miss analysis.

It does not use FAISS, Chroma, sentence-transformers, a GPU, or network access.

## Metrics

`retriever_eval_summary_v1.json` records:

- `questions_total`;
- `top_k_values`;
- `hit_rate_by_top_k`;
- `mean_expected_source_hit_rank_by_top_k`;
- `miss_questions_by_top_k`;
- `per_question_type_hit_rate`;
- `comparison_to_rag_v0_if_available`;
- `recommendation`.

For the current FastCalo run:

```text
v0 expected-source hit rate at top_k=5: 0.625
v1 expected-source hit rate at top_k=3: 0.625
v1 expected-source hit rate at top_k=5: 0.750
v1 expected-source hit rate at top_k=8: 0.875
```

## How To Inspect Misses

Use:

```text
outputs/rag_v1_fastcalosim_retrieval/retriever_error_analysis_v1.md
outputs/rag_v1_fastcalosim_retrieval/retrieval_debug_report.md
```

The debug report lists, for each largest-top-k miss:

- the question;
- expected arXiv source;
- top retrieved sources;
- best retrieved chunks;
- whether expected-source chunks contain query terms;
- a likely reason such as `ambiguous_question`, `lexical_mismatch`, or `retriever_weakness`.

The current top-k 8 miss is `eval_v0_q002`, likely because the wording asks for "another selected paper" and the lexical signal is ambiguous across several GAN/calorimeter papers.

## How This Informs The Next QA Step

Fase 2.1 separates retrieval quality from answer generation. The next QA-grounded step can reuse the retrieved chunk IDs and citation contract to require structured model output, then reject answers whose citations are missing or reference chunks outside the retrieved context.

## Limitations

- Expected-source hit rate is not scientific correctness.
- Expected-source labels may be too strict or ambiguous.
- TF-IDF remains lexical and can miss semantic matches.
- Higher top-k can improve recall while adding noisier context.
- Manual review remains required before using answers as scientific QA.
