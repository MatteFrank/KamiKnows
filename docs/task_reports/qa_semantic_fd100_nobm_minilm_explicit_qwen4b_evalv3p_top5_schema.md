# Phase 2 Task 04 - QA Grounded v1

Date: 2026-07-02

## Objective

Build a QA RAG grounded v1 loop over chunks_v1 with deterministic retrieved-chunk citations and local Ollama generator comparison.

QA grounded v1 tests citation-constrained answer generation, but does not prove scientific reliability.

## Inputs Used

```text
outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1/chunks_v1.jsonl
outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1/rag_manifest_v1.json
data/eval/rag_eval_set_v3_paired.jsonl
```

The existing `outputs/rag_ready_fastcalo_v0/` and `outputs/rag_ready_fastcalo_v1/` directories were not overwritten.

## Chunks v1 Used

- Papers: 5
- Chunks: 169
- Equations: 28
- v0 eval questions imported: 0
- Known chunk quality flags: {}

## Models Configured

- qwen3:4b-instruct

## Models Effectively Available

- qwen3:4b-instruct

Unavailable models:

- <none>

Models with generation backend errors:

- <none>

## Eval Set Used

- Total questions: 20
- Answerable questions: 16
- Insufficient-evidence questions: 4
- Includes fact lookup, method, limitation, comparison, and no-answer/insufficient-evidence questions.

Question source:

- Complete eval questions are loaded from `data/eval/rag_eval_set_v3_paired.jsonl`.
- No operational eval questions are hard-coded in the QA runtime.
- No LLM generates questions in QA grounded v1.

## Top-k

`top_k=5` using `minilm_local_v1`.

## Summary Retrieval

- Expected-source hit rate: 0.875
- Expected-source hits: 14 / 16
- Retrieval misses: eval_v3_pair_06_implicit, eval_v3_pair_08_implicit

## Summary QA Per Model

| Model | Availability | Answered | Insufficient | Backend errors | Invalid citations |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3:4b-instruct | available | 12 | 8 | 0 | 0 |

## Citation Validation

- Invalid citation records: 0
- Invented citation labels: 0
- Answered without valid citation: 0
- Status counts: `{"insufficient_evidence_no_citation": 7, "valid": 13}`

## Principal Failure Cases

- Retrieval misses: eval_v3_pair_06_implicit, eval_v3_pair_08_implicit
- Backend/model availability warnings: `{"json_parse_error": 4, "truncated_at_token_limit": 4}`
- Manual grounding review remains pending for all generated answers.

## Model Comparison

Available local generators in this run: qwen3:4b-instruct. See `rag_model_comparison_v1.md` for citation and abstention metrics.

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

`outputs/qa_semantic_fd100_nobm_minilm_explicit_qwen4b_evalv3p_top5_schema`
