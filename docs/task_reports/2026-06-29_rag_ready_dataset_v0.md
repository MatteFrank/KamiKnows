# KamiKnows Task Report - Fase 1G RAG-Ready Dataset v0

Date: 2026-06-29

## Task Objective

Build a RAG-ready dataset v0 from the existing Fase 1F full-text parsing outputs. This task consolidated chunks, papers, equations, static evaluation-question scaffolding, a manifest, and chunk quality reports.

This task did not build RAG, create embeddings, create a vector database, implement retrieval, call Qwen/Mistral/Ollama or any LLM, fine-tune, add LoRA/QLoRA/training, or generate scientific answers.

## Input Directory Used

```text
outputs/fulltext_fastcalo_5_v0
```

The input directory was the completed Fase 1F full-text parsing pilot output.

## Files Created Or Modified

Created:

- `kamiknows/rag_ready/__init__.py`
- `kamiknows/rag_ready/build_dataset.py`
- `kamiknows/rag_ready/validate_chunks.py`
- `kamiknows/rag_ready/eval_questions.py`
- `kamiknows/rag_ready/manifest.py`
- `scripts/build_rag_ready_dataset.py`
- `tests/test_rag_ready_validate_chunks.py`
- `tests/test_rag_ready_eval_manifest.py`
- `tests/test_rag_ready_build_dataset.py`
- `docs/rag_ready_dataset_v0.md`
- `docs/task_reports/2026-06-29_rag_ready_dataset_v0.md`

Generated:

- `outputs/rag_ready_fastcalo_v0/chunks.jsonl`
- `outputs/rag_ready_fastcalo_v0/papers.jsonl`
- `outputs/rag_ready_fastcalo_v0/equations.jsonl`
- `outputs/rag_ready_fastcalo_v0/eval_questions_v0.jsonl`
- `outputs/rag_ready_fastcalo_v0/rag_manifest_v0.json`
- `outputs/rag_ready_fastcalo_v0/chunk_quality_report.json`
- `outputs/rag_ready_fastcalo_v0/chunk_quality_report.md`

## Commands Executed

Input/output inspection:

```bash
find outputs/fulltext_fastcalo_5_v0 -maxdepth 4 -type f | sort
test -e outputs/rag_ready_fastcalo_v0 && find outputs/rag_ready_fastcalo_v0 -maxdepth 3 -type f | sort || echo 'outputs/rag_ready_fastcalo_v0 does not exist'
```

Focused RAG-ready tests:

```bash
source ~/venvs/venv_kamiknows/bin/activate && pytest -q \
  tests/test_rag_ready_validate_chunks.py \
  tests/test_rag_ready_eval_manifest.py \
  tests/test_rag_ready_build_dataset.py
```

Full test suite:

```bash
source ~/venvs/venv_kamiknows/bin/activate && pytest -q
```

Dataset build:

```bash
source ~/venvs/venv_kamiknows/bin/activate && python3 scripts/build_rag_ready_dataset.py \
  --fulltext-dir outputs/fulltext_fastcalo_5_v0 \
  --output-dir outputs/rag_ready_fastcalo_v0 \
  --domain "HEP / FastCaloSimulation / calorimetry"
```

Output inspection:

```bash
find outputs/rag_ready_fastcalo_v0 -maxdepth 1 -type f | sort
sed -n '1,220p' outputs/rag_ready_fastcalo_v0/chunk_quality_report.md
```

## Tests Run And Results

Focused RAG-ready tests:

```text
6 passed
```

Full suite:

```text
158 passed, 2 skipped
```

The skipped tests are the optional Ollama/Qwen tests.

## Dataset Build Results

```text
Papers included: 5
Chunks generated: 108
Equations included: 28
Evaluation questions: 8
Validation status: PASS
```

Validation summary:

```text
orphan_chunks: 0
empty_chunks: 0
missing_required_fields: {}
duplicate_chunk_ids: []
papers_without_chunks: []
warnings: []
```

Chunk distribution:

| Paper ID | Chunks |
|---|---:|
| `1705_02355v2` | 11 |
| `1712_10321v1` | 20 |
| `1805_00850v2` | 18 |
| `1807_01954v2` | 25 |
| `2106_05285v3` | 34 |

Source type counts:

```text
section: 108
```

Parsing status counts:

```text
success: 4
partial: 1
```

Word count min/median/max:

```text
18 / 351.5 / 494
```

## Notes

The dataset is RAG-ready only in the sense that chunks, papers, equations, eval-question scaffolds, manifest, and validation reports exist. It is not yet RAG: there are no embeddings, no vector DB, no retrieval, and no generated answers.
