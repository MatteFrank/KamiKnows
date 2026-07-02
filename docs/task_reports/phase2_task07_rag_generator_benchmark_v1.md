# Phase 2 Task 07 - RAG Generator and Configuration Benchmark v1

Date: 2026-07-02

## Objective

Benchmark complete RAG configurations across context sources, prompt profiles, and local generators using automatic structural metrics.

This benchmark compares RAG configurations using automatic structural metrics, but it does not prove scientific answer correctness without manual grounding review.

## Why P2T7

P2T4 established grounded QA generation, P2T5 prepared manual review, and P2T6 compared baseline and reranked contexts. P2T7 combines those axes to identify which configuration should be prioritized for human grounding review.

## Benchmark Matrix

- Configs: 12
- Context sources: tfidf_baseline_top5, light_reranked_top5
- Prompt profiles: grounded_json_v1, strict_abstention_v2
- Models: qwen3:0.6b, qwen3:4b-instruct, mistral:7b
- Parameter sets: deterministic_512

## Inputs Used

- `outputs/rag_ready_fastcalo_v1/chunks_v1.jsonl`
- `outputs/rag_ready_fastcalo_v1/rag_manifest_v1.json`
- `data/eval/rag_eval_set_v1_hardened.jsonl`
- reranking output dir: `outputs/rag_v1_fastcalosim_reranking_light_v1`

## Outputs Produced

- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/benchmark_matrix_v1.json`
- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/benchmark_answers_v1.jsonl`
- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/benchmark_summary_v1.json`
- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/benchmark_config_comparison_v1.md`
- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/citation_validation_summary_v1.json`
- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/abstention_analysis_v1.json`
- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/benchmark_error_analysis_v1.md`
- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/recommended_rag_config_v1.json`
- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/benchmark_manifest_v1.json`
- `docs/task_reports/phase2_task07_rag_generator_benchmark_v1.md`

## Automatic Results

- Answer records: 156
- Backend errors: 156
- Citation invalid records: 0
- Invented citations: 0
- Answered without valid citation: 0
- Correct abstentions: 0
- Failed abstentions / possible overclaims: 0
- Over-abstentions: 0

## Recommended Config For Manual Review

`None`

No config passed hard citation filters.

- Rejected configs: 12

## Comparisons

- Generator, prompt, and context-source comparisons are tabulated in `benchmark_config_comparison_v1.md`.
- Baseline vs reranked context comparison is structural and depends on P2T6 context availability.
- Citation validation checks labels only; factual support remains manual.

## What Is Not Demonstrated

- Scientific answer correctness.
- Full factual entailment of every generated sentence.
- Production readiness or detector-simulation replacement decisions.

## Limits

- No training, LoRA, QLoRA, discovery, corpus scaling, parser changes, or chunks_v1 changes were performed.
- No LLM judge was used.
- Expected source IDs were not given to generation prompts and are used only post-hoc.
- Dataset counts: `{"chunks_v1": 108, "chunks_with_quality_flags": 31, "equations": 28, "papers": 5}`

## Next Step

Use the recommended config only to prioritize manual grounding review, then compare human labels before making any answer-quality claim.

## Top Structural Configs

- `cfg_001__tfidf_baseline_top5__grounded_json_v1__qwen3_0_6b__deterministic_512`: answered=0, insufficient=0, backend_errors=13, invalid_citations=0
- `cfg_002__tfidf_baseline_top5__grounded_json_v1__qwen3_4b_instruct__deterministic_512`: answered=0, insufficient=0, backend_errors=13, invalid_citations=0
- `cfg_003__tfidf_baseline_top5__grounded_json_v1__mistral_7b__deterministic_512`: answered=0, insufficient=0, backend_errors=13, invalid_citations=0
- `cfg_004__tfidf_baseline_top5__strict_abstention_v2__qwen3_0_6b__deterministic_512`: answered=0, insufficient=0, backend_errors=13, invalid_citations=0
- `cfg_005__tfidf_baseline_top5__strict_abstention_v2__qwen3_4b_instruct__deterministic_512`: answered=0, insufficient=0, backend_errors=13, invalid_citations=0
