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

`top_k=5` using `minilm_local_v1`.

## Summary Retrieval

- Expected-source hit rate: 0.818
- Expected-source hits: 9 / 11
- Retrieval misses: eval_v0_q006, eval_v1_q010

## Summary QA Per Model

| Model | Availability | Answered | Insufficient | Backend errors | Invalid citations |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3:0.6b | available | 11 | 2 | 0 | 0 |

## Citation Validation

- Invalid citation records: 0
- Invented citation labels: 0
- Answered without valid citation: 0
- Status counts: `{"valid": 13}`

## Principal Failure Cases

- Retrieval misses: eval_v0_q006, eval_v1_q010
- Backend/model availability warnings: `{"json_parse_error": 6}`
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

`outputs/qa_grounded_qwen06b_semantic_minilm_eval_v2_top5_retry1`


## Executed MiniLM-context QA comparison

Job 12143 COMPLETED, exit 0, elapsed 00:07:27. Exact Slurm command is saved in
`outputs/qa_grounded_qwen06b_semantic_minilm_eval_v2_top5_retry1/run.slurm`.
One GPU, four CPUs, 8 GB RAM, maximum 30 minutes. Dedicated Ollama port 11442.
Initial job 12142 was cancelled after finding port 11434 already occupied;
its logs remain in the unsuffixed output directory and are not comparison results.
No existing service was deliberately killed. The retry started its own service
and closed it via the exit trap.

The saved MiniLM ranking was reused without embedding inference or retrieval.
All 13 context rankings match the original saved chunk IDs and order exactly.
Complete text is hydrated from the same 118 semantic chunks; deterministic citation
labels are assigned by rank. The config records MiniLM and saved-retrieval hashes.
Qwen model, prompt, temperature 0, top-p 0.9 and 512-token limit are unchanged.

| Retriever | Source hits | Answered | Insufficient evidence | JSON errors | Invalid citations | Backend errors |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 8/11 | 11 | 2 | 5 | 0 | 0 |
| MiniLM | 9/11 | 11 | 2 | 6 | 0 | 0 |

## Initial content review

- q012 now correctly distinguishes CaloGAN GANs from CaloFlow normalizing flows,
  unlike the TF-IDF answer; however its JSON is incomplete and several broader
  claims still need citation-support review.
- q001 and q002 still incorrectly attribute WGAN to the requested CaloGAN papers.
- q003 gives an inadequate rationale for one-dimensional validation limitations.
- q006 lacks the requested jet-image paper and compares WGAN context instead.
- q010 incorrectly identifies a WGAN energy mechanism, matching its source miss.
- q011 does not provide the requested high-dimensional/classifier validation explanation.
- Both intended abstention questions are assigned insufficient_evidence.
- JSON errors: q003, q004, q005, q007, q011, q012.

## Limits and next step

This is an initial inspection, not completed scientific annotation. All grounding
fields remain not_reviewed. Formally valid citations do not prove source relevance
or entailment; one extra retrieval hit does not establish better overall answers.
Record a question-level evidence/answer review before adjusting corpus filtering,
prompt, model or retrieval parameters in a separate controlled experiment.
