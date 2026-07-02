# Phase 2 Task 04 - QA Grounded v1

Date: 2026-07-02

## Objective

Build a QA RAG grounded v1 loop over chunks_v1 with deterministic retrieved-chunk citations and local Ollama generator comparison.

QA grounded v1 tests citation-constrained answer generation, but does not prove scientific reliability.

## Inputs Used

```text
outputs/rag_ready_fastcalo_v1/chunks_v1.jsonl
outputs/rag_ready_fastcalo_v1/rag_manifest_v1.json
data/eval/rag_eval_set_v1_hardened.jsonl
```

The existing `outputs/rag_ready_fastcalo_v0/` and `outputs/rag_ready_fastcalo_v1/` directories were not overwritten.

## Chunks v1 Used

- Papers: 5
- Chunks: 108
- Equations: 28
- v0 eval questions imported: 0
- Known chunk quality flags: {"chunks_with_quality_flags": 31, "quality_flag_counts": {"very_long_text": 31}}

## Models Configured

- qwen3:0.6b
- qwen3:4b-instruct
- mistral:7b

## Models Effectively Available

- qwen3:0.6b
- qwen3:4b-instruct
- mistral:7b

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

- Complete eval questions are loaded from `data/eval/rag_eval_set_v1_hardened.jsonl`.
- No operational eval questions are hard-coded in the QA runtime.
- No LLM generates questions in QA grounded v1.

## Top-k

`top_k=5` using `tfidf_local_v1`.

## Summary Retrieval

- Expected-source hit rate: 0.727
- Expected-source hits: 8 / 11
- Retrieval misses: eval_v0_q002, eval_v0_q005, eval_v1_q012

## Summary QA Per Model

| Model | Availability | Answered | Insufficient | Backend errors | Invalid citations |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3:0.6b | available | 10 | 3 | 0 | 3 |
| qwen3:4b-instruct | available | 9 | 4 | 0 | 5 |
| mistral:7b | available | 11 | 2 | 0 | 4 |

## Citation Validation

- Invalid citation records: 12
- Invented citation labels: 3
- Answered without valid citation: 12
- Status counts: `{"answered_without_valid_citation": 11, "insufficient_evidence_no_citation": 6, "invented_citation": 1, "valid": 21}`

## Principal Failure Cases

- Retrieval misses: eval_v0_q002, eval_v0_q005, eval_v1_q012
- Backend/model availability warnings: `{"answered_without_valid_citation": 11, "invalid_answer_status_normalized": 5, "invalid_confidence_normalized": 1, "invented_citation": 1, "json_parse_error": 7}`
- Manual grounding review remains pending for all generated answers.

## Model Comparison

Available local generators in this run: qwen3:0.6b, qwen3:4b-instruct, mistral:7b. See `rag_model_comparison_v1.md` for citation and abstention metrics.

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

`outputs/rag_v1_fastcalosim_qa_grounded_hardened_v1`
