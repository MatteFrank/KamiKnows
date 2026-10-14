# Phase 2 Task 04 - QA Grounded v1

Date: 2026-07-02

## Objective

Build a QA RAG grounded v1 loop over chunks_v1 with deterministic retrieved-chunk citations and local Ollama generator comparison.

QA grounded v1 tests citation-constrained answer generation, but does not prove scientific reliability.

## Inputs Used

```text
outputs/rag_ready_fastcalo_semantic_v1/chunks_v1.jsonl
outputs/rag_ready_fastcalo_semantic_v1/rag_manifest_v1.json
data/eval/rag_eval_set_v2.jsonl
```

The existing `outputs/rag_ready_fastcalo_v0/` and `outputs/rag_ready_fastcalo_v1/` directories were not overwritten.

## Chunks v1 Used

- Papers: 5
- Chunks: 118
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

`top_k=5` using `tfidf_local_v1`.

## Summary Retrieval

- Expected-source hit rate: 0.727
- Expected-source hits: 8 / 11
- Retrieval misses: eval_v0_q002, eval_v0_q003, eval_v1_q012

## Summary QA Per Model

| Model | Availability | Answered | Insufficient | Backend errors | Invalid citations |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3:0.6b | available | 11 | 2 | 0 | 0 |

## Citation Validation

- Invalid citation records: 0
- Invented citation labels: 0
- Answered without valid citation: 0
- Status counts: `{"insufficient_evidence_no_citation": 1, "valid": 12}`

## Principal Failure Cases

- Retrieval misses: eval_v0_q002, eval_v0_q003, eval_v1_q012
- Backend/model availability warnings: `{"json_parse_error": 5}`
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

`outputs/qa_grounded_qwen06b_semantic_eval_v2_top5`


## Executed semantic run and three-strategy comparison

Slurm job 12138. Exact command and resource configuration are preserved in
`outputs/qa_grounded_qwen06b_semantic_eval_v2_top5/run.slurm`.
One GPU, four CPUs, 8 GB RAM, maximum 30 minutes. Ollama is started using the
existing cached container and is stopped by the exit trap. No source code,
corpus, question set, prompt or retrieval implementation was changed in this run.
The input dataset has 118 semantic chunks; TF-IDF top-k is 5, without reranking.
Qwen3:0.6b settings: temperature 0, top-p 0.9, num_predict 512, request timeout 120s.

| Chunking | Source hits | Answered status | Insufficient evidence | Backend errors | JSON warnings | Invalid citations |
|---|---:|---:|---:|---:|---:|---:|
| Word | 9/11 | 11 | 2 | 0 | 4 | 0 |
| Sentence | 8/11 | 12 | 1 | 0 | 7 | 1 |
| Semantic | 8/11 | 11 | 2 | 0 | 5 | 0 |

## Initial content inspection

- q001 incorrectly attributes WGAN to the first CaloGAN paper.
- q002 and q003 again mix source-specific CaloGAN questions with WGAN context.
- q005 lacks the requested detailed comparison and evidence from both papers.
- q006 is incomplete and explicitly notes absent validation information.
- q012 incorrectly states both CaloGAN and CaloFlow use normalizing flows and
  attributes the two-step flow to CaloGAN.
- q008 and q013 both have insufficient_evidence status.
- JSON parsing warnings affect q002, q003, q006, q007 and q011.

All 13 benchmark records were checked against the canonical v2 file. All answers
remain grounding_status=not_reviewed. Valid citation labels do not establish
scientific correctness; these observations are an initial inspection, not a
completed per-question scientific annotation. One run on 13 questions does not
establish general superiority of a chunking method. No additional model or
benchmark tests were run after successful generation.

## Next step

Review source relevance and citation support across all three runs before changing
parameters. Bibliographic distractors and generator/source confusion remain
important failure modes; evaluate fixes separately from the chunking comparison.
