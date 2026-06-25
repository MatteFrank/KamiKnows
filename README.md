# KamiKnows - Fase 0 closure repository

KamiKnows is a modular scientific LLM prototype for turning scientific papers into structured, traceable, reviewable knowledge.

This repository closes **Fase 0**: tutorial, minimal LLM backend, arXiv metadata ingestion, abstract-to-JSON extraction, JSONL output, formal validation, manual review scaffolding, manifest, quality gate, and HEP pilot runner.

It is not yet a RAG system, PDF parser, LaTeX parser, fine-tuning project, or discovery engine.

## Phase status

Read first:

```text
PROJECT_STATUS.md
docs/phase0_closure.md
docs/phase1_handoff.md
```

Current boundary:

```text
Fase 0 complete
-> ready for controlled Fase 1 pilot in a fresh Codex/chat workflow
```

## What works now

```text
arXiv metadata
-> title + abstract
-> ModelPlugin
-> Qwen/Ollama or fake test backend
-> extraction JSON
-> schema validation
-> traceable JSONL
-> formal summary
-> manual quality checklist
-> manual review summary
-> dataset manifest
-> quality gate
-> post-pilot analysis
```

## What is intentionally out of scope

```text
PDF parsing
LaTeX parsing
chunking
embeddings
vector database
RAG
fine-tuning
automatic scientific correctness scoring
large-scale corpus processing
```

## Setup

Use Python 3.11 or 3.12 for the real ML stack. The current pure-Python tests also run on newer Python versions, but later ML dependencies are usually more stable on 3.11/3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For local Qwen tests:

```bash
ollama pull qwen3:0.6b
```

## Test suite

```bash
pytest -q
```

Ollama/Qwen tests are skipped unless explicitly enabled:

```bash
KAMIKNOWS_RUN_OLLAMA_TEST=1 pytest -s tests/test_qwen_ollama_basic.py
KAMIKNOWS_RUN_OLLAMA_TEST=1 pytest -s tests/test_qwen_abstract_to_json_ollama.py
```

During the project rename, the old `KNOWKAMI_RUN_OLLAMA_TEST=1` flag remains
accepted as a temporary deprecated alias.

## Metadata ingestion only

Metadata download is separate from model interpretation.

From a query:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 10 \
  --output outputs/metadata/calorimeter_metadata.json
```

From explicit IDs:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --ids 2301.00001 2301.00002 \
  --output outputs/metadata/selected_ids_metadata.json
```

From a plain-text ID-list file:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --output outputs/metadata/selected_ids_metadata.json
```

ID-list file format:

```text
# comments are ignored
2301.00001v1
arXiv:2301.00002v2
https://arxiv.org/abs/2301.00003v1
```

## Batch extraction

From frozen metadata:

```bash
python scripts/run_batch_arxiv_extraction.py \
  --metadata-list outputs/metadata/calorimeter_metadata.json \
  --backend ollama \
  --model qwen3:0.6b \
  --output outputs/qwen_calorimeter_extractions.jsonl
```

From a query, metadata and extraction in one command:

```bash
python scripts/run_batch_arxiv_extraction.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 10 \
  --backend ollama \
  --model qwen3:0.6b \
  --output outputs/qwen_calorimeter_extractions.jsonl
```

From an ID-list file:

```bash
python scripts/run_batch_arxiv_extraction.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --backend ollama \
  --model qwen3:0.6b \
  --output outputs/qwen_selected_ids_extractions.jsonl
```

## Controlled HEP pilot

Recommended Fase 1 entry point:

```bash
python scripts/run_hep_pilot.py \
  --metadata-list outputs/metadata/calorimeter_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

Or directly from an ID-list file:

```bash
python scripts/run_hep_pilot.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --backend ollama \
  --model qwen3:0.6b
```

Default output directory:

```text
outputs/hep_pilot/
```

Important files:

```text
pilot_metadata.json
pilot_ollama_qwen3_0_6b.jsonl
pilot_ollama_qwen3_0_6b_summary.json
pilot_manual_quality_checklist.md
dataset_manifest.json
pilot_report.json
```

## Manual review and quality gate

After manually completing `pilot_manual_quality_checklist.md`:

```bash
python scripts/summarize_manual_quality_review.py \
  outputs/hep_pilot/pilot_manual_quality_checklist.md \
  --json-output outputs/hep_pilot/pilot_manual_review_summary.json

python scripts/create_dataset_manifest.py \
  --from-mini-benchmark-dir outputs/hep_pilot \
  --output outputs/hep_pilot/dataset_manifest.json

python scripts/run_quality_gate.py \
  --manifest outputs/hep_pilot/dataset_manifest.json \
  --output outputs/hep_pilot/quality_gate_report.json

python scripts/summarize_hep_pilot_run.py \
  --manifest outputs/hep_pilot/dataset_manifest.json \
  --output outputs/hep_pilot/post_pilot_analysis.json
```

## Main documentation

```text
docs/learning_notes.md
docs/run_record_schema.md
docs/model_mini_benchmark.md
docs/manual_quality_checklist.md
docs/manual_quality_review_summary.md
docs/dataset_manifest.md
docs/quality_gate.md
docs/hep_pilot.md
docs/post_pilot_analysis.md
docs/phase0_closure.md
docs/phase1_handoff.md
```

## Development rule

Do not scale before review. For each pilot:

```text
formal summary PASS
+ manual review summary present
+ dataset manifest complete
+ quality gate decision understood
+ post-pilot analysis read
```

Only then choose one next change: prompt, model, query, or schema.
