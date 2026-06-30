# KamiKnows Task Report - Fase 2.1 Retriever Baseline v1

Date: 2026-06-29

## Task Objective

Implement Retrieval Baseline v1 and citation hardening for the existing five-paper FastCaloSimulation RAG corpus.

Scope:

```text
question
-> deterministic local retrieval over existing chunks
-> explicit expected-source metrics at top-k 3/5/8
-> miss diagnostics and debug reports
-> citation validation helper and future JSON answer contract
```

This task did not add fine-tuning, LoRA, QLoRA, training, discovery generation, a knowledge graph, FAISS, Chroma, a vector DB, GPU dependencies, network tests, or corpus scaling.

## Input Files Used

```text
outputs/rag_ready_fastcalo_v0/chunks.jsonl
outputs/rag_ready_fastcalo_v0/papers.jsonl
outputs/rag_ready_fastcalo_v0/equations.jsonl
outputs/rag_ready_fastcalo_v0/eval_questions_v0.jsonl
outputs/rag_ready_fastcalo_v0/rag_manifest_v0.json
outputs/rag_v0_fastcalosim/rag_eval_summary.json
outputs/rag_v0_fastcalosim/retrieved_contexts.jsonl
outputs/rag_v0_fastcalosim/rag_answers.jsonl
outputs/rag_v0_fastcalosim/rag_error_analysis.md
```

## Files Created Or Modified

Created:

- `kamiknows/rag/retriever_v1.py`
- `kamiknows/rag/retriever_baseline_v1.py`
- `kamiknows/rag/citation_validation.py`
- `scripts/run_retriever_baseline_v1.py`
- `tests/test_retriever_baseline_v1.py`
- `docs/retriever_baseline_v1.md`
- `docs/rag_citation_contract_v1.md`
- `docs/task_reports/2026-06-29_retriever_baseline_v1.md`

Modified:

- `kamiknows/rag/__init__.py`

Generated output:

- `outputs/rag_v1_fastcalosim_retrieval/retriever_baseline_v1.jsonl`
- `outputs/rag_v1_fastcalosim_retrieval/retriever_eval_summary_v1.json`
- `outputs/rag_v1_fastcalosim_retrieval/retriever_error_analysis_v1.md`
- `outputs/rag_v1_fastcalosim_retrieval/retrieval_debug_report.md`
- `outputs/rag_v1_fastcalosim_retrieval/retrieval_run_config.json`
- `outputs/rag_v1_fastcalosim_retrieval/retrieval_manifest.json`

## Commands Executed

Focused tests and CLI help:

```bash
pytest -q tests/test_retriever_baseline_v1.py
python3 scripts/run_retriever_baseline_v1.py --help
```

Real v1 baseline run:

```bash
python3 scripts/run_retriever_baseline_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_v1_fastcalosim_retrieval \
  --top-k-values 3 5 8
```

Full test suite:

```bash
pytest -q
```

## Tests Run And Result

Focused retrieval/citation tests:

```text
7 passed
```

Full suite:

```text
172 passed, 2 skipped
```

The skipped tests are the existing optional Ollama/Qwen tests.

## Hit Rate v0 Baseline

From `outputs/rag_v0_fastcalosim/rag_eval_summary.json`:

```text
retriever: tfidf_local_v0
top_k: 5
expected-source hit rate: 0.625
expected-source hits: 5 / 8
```

## Hit Rate v1 By Top-K

From `outputs/rag_v1_fastcalosim_retrieval/retriever_eval_summary_v1.json`:

```text
top_k=3: 0.625
top_k=5: 0.750
top_k=8: 0.875
```

Best observed v1 result:

```text
top_k=8
expected-source hit rate: 0.875
delta vs v0 top_k=5: +0.250
```

## Retrieval Miss Analysis

Misses by top-k:

```text
top_k=3: eval_v0_q001, eval_v0_q002, eval_v0_q005
top_k=5: eval_v0_q002, eval_v0_q005
top_k=8: eval_v0_q002
```

The remaining largest-top-k miss is:

```text
eval_v0_q002
Question: Which neural or statistical modeling method is used for calorimeter shower generation in another selected paper?
Expected source: 1712.10321v1
Likely reason: ambiguous_question
```

The expected paper has chunks containing key terms, so this is not an expected-source-unavailable case. The question wording is ordinal and ambiguous across several GAN/calorimeter papers.

## Citation Contract Status

Created `docs/rag_citation_contract_v1.md` and `kamiknows/rag/citation_validation.py`.

The contract requires future generated answers to return structured JSON with:

```json
{
  "answer": "...",
  "citations": [],
  "insufficient_evidence": false,
  "limitations": "...",
  "used_chunk_ids": []
}
```

Citation validation now detects:

- `valid`
- `missing_citations`
- `unknown_citation`
- `cites_unretrieved_chunk`

No generation rewrite was performed in this phase.

## What This Demonstrates

- Retrieval quality can be measured independently from generation.
- `tfidf_local_v1` improves expected-source recall on this small corpus at top-k 5 and 8.
- Misses are easier to debug by question, expected source, retrieved chunks, and likely reason.
- Citation structure can be validated against retrieved chunk IDs before accepting future answers.
- Manual review remains part of the workflow.

## What This Does NOT Demonstrate

- Scientific correctness.
- Fully robust RAG.
- Reliable answer generation.
- Citation-support correctness beyond chunk-ID subset validation.
- Semantic retrieval quality.
- Performance on a larger corpus.
- Any benefit from fine-tuning, training, vector databases, or external services.

## Recommended Next Step

Keep the corpus fixed and update the generation step to require the citation contract JSON. Reject or mark for revision any answer with missing citations, malformed citations, or citations outside the retrieved chunks. Then manually review the cited text for actual support.

## Copy-Paste Summary For ChatGPT Continuity

Fase 2.1 Retrieval Baseline v1 is implemented. New command:

```bash
python3 scripts/run_retriever_baseline_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_v1_fastcalosim_retrieval \
  --top-k-values 3 5 8
```

Outputs exist in `outputs/rag_v1_fastcalosim_retrieval/`: `retriever_baseline_v1.jsonl`, `retriever_eval_summary_v1.json`, `retriever_error_analysis_v1.md`, `retrieval_debug_report.md`, `retrieval_run_config.json`, and `retrieval_manifest.json`. v0 expected-source hit rate was 0.625 at top_k=5. v1 hit rates are 0.625 at top_k=3, 0.750 at top_k=5, and 0.875 at top_k=8. Remaining top-k 8 miss is `eval_v0_q002`, likely because the question wording is ambiguous. Citation hardening docs and helper exist: `docs/rag_citation_contract_v1.md` and `kamiknows/rag/citation_validation.py`. Full tests pass: 172 passed, 2 skipped. Manual review remains required; no scientific correctness is claimed.
