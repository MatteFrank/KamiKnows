# KnowKami / M-Fond LLM - Codex Operating Notes

This repository is the KnowKami / M-Fond LLM project. It is an incremental,
traceable research-codebase for local LLM extraction, quality review, and
robust scientific RAG over a small FastCaloSimulation corpus.

## Current Phase

Phase 2 is in progress: robust scientific RAG.

Canonical Phase 2 state:

- P2T1 completed: retrieval baseline v1 plus citation hardening.
- P2T2 completed: embedding model selection as pipeline validation only.
  `sentence_transformers` was not installed during that run, so embedding
  candidates used the deterministic hash fallback and must not be reported as
  a real semantic model comparison.
- P2T3 completed: chunking and metadata v1.
- P2T4 completed: QA grounded v1.
- P2T4b completed: JSON config refactor for QA grounded v1 defaults.
- Current next practical step: manual grounding review of P2T4 answers.

## Non-Negotiable Constraints

- Do not perform training, LoRA, QLoRA, or fine-tuning unless a future task
  explicitly opens that scope.
- Do not introduce discovery generation.
- Do not scale the corpus beyond the task-approved input set.
- Do not modify the full-text parser unless the task explicitly requests it.
- Do not overwrite `outputs/rag_ready_fastcalo_v0/`.
- Do not overwrite `outputs/rag_ready_fastcalo_v1/`.
- Do not present RAG output as scientifically reliable without manual review.
- Do not invent citations, arXiv IDs, paper IDs, chunk IDs, or sources.
- Preserve existing outputs unless the task explicitly defines a new output
  directory or asks to regenerate a specific artifact.

## Main Data Structure

Current RAG-ready dataset:

```text
outputs/rag_ready_fastcalo_v1/
```

Known counts:

- papers: 5
- chunks: 108
- equations: 28
- eval questions v0: 8
- known issue: 31 chunks flagged `very_long_text`

Important files:

```text
outputs/rag_ready_fastcalo_v1/chunks_v1.jsonl
outputs/rag_ready_fastcalo_v1/eval_questions_v0.jsonl
outputs/rag_ready_fastcalo_v1/rag_manifest_v1.json
outputs/rag_ready_fastcalo_v1/retrieval_smoke_summary_v1.json
```

The v1 dataset is traceable to v0 through:

```text
outputs/rag_ready_fastcalo_v1/chunk_id_map_v0_to_v1.json
outputs/rag_ready_fastcalo_v1/chunk_audit_v0_summary.json
outputs/rag_ready_fastcalo_v1/chunk_audit_v0_details.jsonl
```

## Architectural Principles

- Prefer small deterministic local baselines before adding heavier models.
- Keep retrieval, generation, citation validation, and manual review as
  separate concerns.
- Keep every generated answer auditable back to retrieved chunks.
- Use structured JSON/JSONL outputs for machine-readable artifacts.
- Use Markdown reports in `docs/task_reports/` for task-level summaries.
- Add new output directories for new experiments instead of mutating prior
  output directories.
- Make optional model backends degrade gracefully. Missing Ollama models,
  missing local model caches, and unavailable dependencies should produce
  explicit warnings or backend-error records, not destroy the whole run.
- Use manual review language for answer quality. Automated checks can validate
  schemas, retrieval identifiers, and citation labels; they do not prove
  scientific correctness.

## Output Conventions

- Task outputs go under `outputs/<task_specific_name>/`.
- Task reports go under `docs/task_reports/`.
- Registry, manifest, summary, audit, and comparison files should be versioned
  with suffixes such as `_v1`.
- JSON should be pretty printed when it is a top-level summary or manifest.
- JSONL should contain one object per line and use stable identifiers.
- Every major output directory should include a manifest or run config when
  feasible.

## Main Commands

Build the RAG-ready v1 chunk metadata dataset:

```bash
python3 scripts/build_rag_ready_chunk_metadata_v1.py \
  --input-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_ready_fastcalo_v1
```

Run retrieval baseline v1:

```bash
python3 scripts/run_retriever_baseline_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_retrieval
```

Run embedding model selection v1:

```bash
python3 scripts/run_embedding_model_selection_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_embedding_selection \
  --top-k 5
```

Config-based P2T4 QA grounded v1 command:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --config configs/phase2_p2t4_qa_grounded_v1.json
```

The config command writes to a new output directory by default:

```text
outputs/rag_v1_fastcalosim_qa_grounded_config_v1
```

Equivalent explicit P2T4 QA grounded v1 command:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_qa_grounded \
  --report-path docs/task_reports/phase2_task04_qa_grounded_v1.md \
  --top-k 5 \
  --models qwen3:0.6b qwen3:4b-instruct mistral:7b
```

Run tests:

```bash
pytest -q
```

## Definition of Done

A task is done only when:

- required code or docs are created in the intended locations;
- required output artifacts exist and are traceable to their inputs;
- previous output directories are not overwritten unless explicitly requested;
- tests pass, or any unrun/unavailable test is clearly explained;
- a Markdown report exists in `docs/task_reports/` when the task requires one;
- limitations and manual-review requirements are stated explicitly;
- generated citations are validated only against deterministic retrieved
  evidence, not accepted because a model produced them.

## Markdown Report Rules

Every task report in `docs/task_reports/` should include:

- objective;
- inputs used;
- outputs produced;
- commands run;
- key metrics;
- failures or warnings;
- limits and non-claims;
- next recommended step.

Reports must distinguish:

- pipeline validation from scientific validation;
- model/backend availability from model quality;
- retrieval correctness from answer correctness;
- citation-label validity from factual entailment.

## Citation, Grounding, and Model Rules

- Citations must be deterministic and associated with retrieved chunks.
- Models may choose among provided citation labels but must not create free
  citations.
- If retrieved context is insufficient, the expected behavior is an explicit
  insufficient-evidence answer.
- `answer_status`, `grounding_status`, and citation-validation status are
  separate fields.
- `grounding_status: not_reviewed` means manual review is still pending.
- Local Ollama models should be configurable. Missing local models must produce
  `model_unavailable` or `backend_error` records rather than aborting the full
  comparison.
- Do not claim that qwen, mistral, embeddings, or TF-IDF results establish
  scientific reliability without manual review.
