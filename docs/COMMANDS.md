# Canonical Commands

This file records the main commands for the current KnowKami / M-Fond LLM
workflow. Commands should be run from the repository root.

## Tests

```bash
pytest -q
```

## Phase 2 Dataset

Build chunk metadata v1:

```bash
python3 scripts/build_rag_ready_chunk_metadata_v1.py \
  --input-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_ready_fastcalo_v1
```

## Retrieval Baseline v1

```bash
python3 scripts/run_retriever_baseline_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_retrieval
```

## Embedding Model Selection v1

```bash
python3 scripts/run_embedding_model_selection_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_embedding_selection \
  --top-k 5
```

Important: if `sentence_transformers` is unavailable, this command may use the
deterministic hash fallback. In that case, the result is pipeline validation,
not a semantic embedding comparison.

## QA Grounded v1

Config-based canonical command:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --config configs/phase2_p2t4_qa_grounded_v1.json
```

The config command writes to
`outputs/rag_v1_fastcalosim_qa_grounded_config_v1` by default so the completed
P2T4 output directory is not overwritten.

Equivalent explicit command:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_qa_grounded \
  --report-path docs/task_reports/phase2_task04_qa_grounded_v1.md \
  --top-k 5 \
  --models qwen3:0.6b qwen3:4b-instruct mistral:7b
```

This command requires local Ollama access for generation. Missing models or
backend failures should be recorded in outputs rather than treated as scientific
quality results.
