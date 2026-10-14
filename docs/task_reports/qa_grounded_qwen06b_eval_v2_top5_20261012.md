# Phase 2 Task 04 - QA Grounded v1

Date: 2026-07-02

## Objective

Build a QA RAG grounded v1 loop over chunks_v1 with deterministic retrieved-chunk citations and local Ollama generator comparison.

QA grounded v1 tests citation-constrained answer generation, but does not prove scientific reliability.

## Inputs Used

```text
outputs/rag_ready_fastcalo_v1_enriched/chunks_v1.jsonl
outputs/rag_ready_fastcalo_v1_enriched/rag_manifest_v1.json
data/eval/rag_eval_set_v2.jsonl
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

- Expected-source hit rate: 0.818
- Expected-source hits: 9 / 11
- Retrieval misses: eval_v0_q003, eval_v1_q012

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

- Retrieval misses: eval_v0_q003, eval_v1_q012
- Backend/model availability warnings: `{"json_parse_error": 4}`
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

`outputs/qa_grounded_qwen06b_eval_v2_top5_20261012`


## Executed run and comparison to the previous Qwen experiment

- Date: 2026-10-12 (Europe/Rome).
- Slurm job: 12124; COMPLETED, exit code 0; elapsed 00:07:05.
- Resources: one GPU, four CPUs, 8 GB RAM; 30-minute maximum allocation.
- Launch command: `sbatch outputs/qa_grounded_qwen06b_eval_v2_top5_20261012/run.slurm`.
- Exact pipeline command and Ollama startup are preserved in `outputs/qa_grounded_qwen06b_eval_v2_top5_20261012/run.slurm`.
- Python environment: `/home/e/eapicell/.conda/envs/kamiknows/bin/python`.
- Existing cached Ollama container and qwen3:0.6b model were reused.
- Previous output directory: `outputs/qa_grounded_qwen06b_test`. New output directory: `outputs/qa_grounded_qwen06b_eval_v2_top5_20261012`.
- Corpus, chunks, TF-IDF implementation, prompt and generation settings were unchanged; the question file changed to v2.
- No source-code changes or parser/chunk rebuilds were performed. Ollama was stopped when the batch job exited.

| Measure | Previous hardened v1 | v2 |
|---|---:|---:|
| Expected-source hits | 8/11 | 9/11 |
| Expected-source hit rate | 0.7273 | 0.8182 |
| Reciprocal-rank mean from saved QA contexts | 0.4000 | 0.4576 |
| Complete multi-source coverage | 1/3 | 2/3 |
| Answered status | 10 | 11 |
| Insufficient-evidence status | 2 | 2 |
| Backend errors | 1 | 0 |
| JSON parsing warnings | 4 | 4 |
| Invalid citation records, excluding backend failures | 3 | 0 |

The reciprocal-rank mean is a post-run calculation over saved context records, using the existing QA hit-rank semantics and zero contribution for misses; it does not alter pipeline metrics. For required-all questions, QA uses the maximum matching retrieved rank when all sources are present.

## Initial answer inspection, not a completed scientific review

- v2 retrieval misses: eval_v0_q003 and eval_v1_q012.
- JSON parsing warnings: eval_v0_q002, eval_v0_q003, eval_v0_q005, eval_v1_q012. Several saved answers contain partial JSON text; syntactic fallback permits an answered status.
- Both intended abstention questions (eval_v0_q008, eval_v1_q013) receive insufficient_evidence status.
- eval_v0_q003 discusses a WGAN despite asking about CaloGAN, indicating source/method confusion.
- eval_v0_q006 identifies the simulated objects but does not supply the requested concrete validation observable for each study.
- Valid citation labels do not demonstrate factual support or answer completeness. All grounding_status fields remain not_reviewed.
- The benchmark wording and some query keywords changed, so improvements are not attributable to chunking, which stayed fixed.

## Next step

Use this v2 run as the word-based chunking reference. Review the problematic answers and retrieved passages before comparing sentence-based chunks with the same question set and retrieval settings.
