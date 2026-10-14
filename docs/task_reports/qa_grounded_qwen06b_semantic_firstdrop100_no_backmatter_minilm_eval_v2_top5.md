# Phase 2 Task 04 - QA Grounded v1

Date: 2026-07-02

## Objective

Build a QA RAG grounded v1 loop over chunks_v1 with deterministic retrieved-chunk citations and local Ollama generator comparison.

QA grounded v1 tests citation-constrained answer generation, but does not prove scientific reliability.

## Inputs Used

```text
outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1/chunks_v1.jsonl
outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1/rag_manifest_v1.json
data/eval/rag_eval_set_v2.jsonl
```

The existing `outputs/rag_ready_fastcalo_v0/` and `outputs/rag_ready_fastcalo_v1/` directories were not overwritten.

## Chunks v1 Used

- Papers: 5
- Chunks: 169
- Equations: 28
- v0 eval questions imported: 0
- Known chunk quality flags: {}

## Models Configured

- qwen3:0.6b

## Models Effectively Available

- qwen3:0.6b

Unavailable models:

- <none>

Models with generation backend errors:

- <none>

## Eval Set Used

- Total questions: 13
- Answerable questions: 11
- Insufficient-evidence questions: 2
- Includes fact lookup, method, limitation, comparison, and no-answer/insufficient-evidence questions.

Question source:

- Complete eval questions are loaded from `data/eval/rag_eval_set_v2.jsonl`.
- No operational eval questions are hard-coded in the QA runtime.
- No LLM generates questions in QA grounded v1.

## Top-k

`top_k=5` using `minilm_local_v1`.

## Summary Retrieval

- Expected-source hit rate: 0.818
- Expected-source hits: 9 / 11
- Retrieval misses: eval_v0_q006, eval_v1_q010

## Summary QA Per Model

| Model | Availability | Answered | Insufficient | Backend errors | Invalid citations |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3:0.6b | available | 12 | 1 | 0 | 3 |

## Citation Validation

- Invalid citation records: 3
- Invented citation labels: 0
- Answered without valid citation: 3
- Status counts: `{"answered_without_valid_citation": 3, "insufficient_evidence_no_citation": 1, "valid": 9}`

## Principal Failure Cases

- Retrieval misses: eval_v0_q006, eval_v1_q010
- Backend/model availability warnings: `{"answered_without_valid_citation": 3, "json_parse_error": 5}`
- Manual grounding review remains pending for all generated answers.

## Model Comparison

Available local generators in this run: qwen3:0.6b. See `rag_model_comparison_v1.md` for citation and abstention metrics.

## Improvements Compared To Mini-RAG v0

- Uses v1 chunk IDs and quality flags.
- Uses deterministic `[C1]`, `[C2]` citation labels tied to retrieved chunks.
- Validates generated citation labels automatically.
- Produces per-model answer files and a comparative summary.
- Separates retrieval evidence, generated answer, citation audit, and manual grounding status.

## What Is Not Yet Demonstrated

- Scientific answer correctness.
- Robust semantic retrieval beyond the TF-IDF local baseline.
- Reliable abstention behavior across models.
- Production readiness for detector simulation replacement decisions.

## Limits

- No training, LoRA, QLoRA, discovery, corpus scaling, or parser changes were performed.
- Chunk text is inherited from v1; 31 chunks still carry `very_long_text` quality flags.
- Citation validation checks labels, not the full factual entailment of every sentence.
- Local Ollama availability determines whether model comparisons contain generated answers or backend-error records.

## Recommended Next Step

Run manual review on generated answers for available models, then address the retrieval misses and very-long chunk policy before claiming any answer-quality improvement.

## Output Directory

`outputs/qa_grounded_qwen06b_semantic_firstdrop100_no_backmatter_minilm_eval_v2_top5`
