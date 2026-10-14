# Phase 2 Task 06 - Light Reranking v1

Date: 2026-10-12

## Objective

Implement and evaluate a lightweight deterministic reranker over TF-IDF candidates for the fixed FastCaloSimulation RAG corpus.

Light reranking can improve ordering of already retrieved candidates, but it cannot recover sources absent from the candidate pool and does not prove scientific answer quality.

## Strategy

- Generate top-20 TF-IDF candidates per hardened eval question.
- Rerank candidates using base score, lexical overlap, title/section overlap, question-type section priors, and quality penalties.
- Use expected source IDs only after ranking for evaluation.
- No training, LLM reranker, LLM judge, discovery, parser change, or corpus scaling is performed.

## Inputs Used

- `outputs/rag_ready_fastcalo_semantic_v1/chunks_v1.jsonl`
- `outputs/rag_ready_fastcalo_semantic_v1/rag_manifest_v1.json`
- `data/eval/rag_eval_set_v2.jsonl`

## Outputs Produced

- `outputs/retrieval_semantic_eval_v2_top5/baseline_retrieval_results_v1.jsonl`
- `outputs/retrieval_semantic_eval_v2_top5/reranked_retrieval_results_v1.jsonl`
- `outputs/retrieval_semantic_eval_v2_top5/reranking_eval_summary_v1.json`
- `outputs/retrieval_semantic_eval_v2_top5/reranking_question_deltas_v1.jsonl`
- `outputs/retrieval_semantic_eval_v2_top5/reranking_error_analysis_v1.md`
- `outputs/retrieval_semantic_eval_v2_top5/reranking_manifest_v1.json`
- `docs/task_reports/phase2_task06_light_reranking_v1.md`
- `data/eval/rag_eval_set_v1_hardened.md`

## Settings

- candidate_top_n: 20
- final_top_k: 5
- no ground truth used for reranking: true

## Metrics

- Questions total: 13
- Expected-source questions: 11
- Baseline hit@5: 0.727 (8)
- Reranked hit@5: 0.818 (9)
- Baseline hit@1: 0.273 (3)
- Reranked hit@1: 0.273 (3)
- Baseline MRR: 0.458
- Reranked MRR: 0.445
- Candidate pool contains expected source count: 11

## Question-Level Changes

- Improved questions: eval_v0_q002, eval_v1_q012
- Worsened questions: eval_v0_q005, eval_v0_q006, eval_v1_q009
- Unchanged questions: eval_v0_q001, eval_v0_q003, eval_v0_q004, eval_v0_q007, eval_v1_q010, eval_v1_q011
- Baseline retrieval misses: eval_v0_q002, eval_v0_q003, eval_v1_q012
- Reranked retrieval misses: eval_v0_q003, eval_v0_q005
- Not fixable by reranking: <none>

## Sensitive Cases

### eval_v0_q002

- Category: `improved`
- Baseline rank: None
- Reranked rank: 4
- Candidate pool rank: 7
- Tags: fixed_by_reranking, very_long_text_context_risk

### eval_v0_q005

- Category: `worsened`
- Baseline rank: 5
- Reranked rank: None
- Candidate pool rank: 18
- Tags: worsened_by_reranking, multi_source_incomplete, very_long_text_context_risk

### eval_v1_q012

- Category: `improved`
- Baseline rank: None
- Reranked rank: 5
- Candidate pool rank: 20
- Tags: fixed_by_reranking, very_long_text_context_risk


## Multi-Source Questions

- Required-all expected-source questions: eval_v0_q005, eval_v0_q006, eval_v1_q012
- All expected sources hit@5: `{"baseline": 2, "reranked": 2, "total": 3}`

## Limits

- Reranking only reorders candidates already found by TF-IDF.
- The reranker is lexical and deterministic; it is not a semantic model comparison.
- Expected source IDs are not used during reranking.
- Retrieval improvement does not validate generated answer quality.
- Dataset counts: `{"chunks_v1": 118, "chunks_with_quality_flags": null, "equations": 28, "eval_questions_v0": 8, "papers": 5}`

## Recommended Next Step

Inspect question-level deltas, especially multi-source questions, before deciding whether to plug reranked contexts into a new QA-grounded generation run.
