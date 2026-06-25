# KamiKnows Fase 1 handoff for Codex

## Goal of the next chat

Continue from the Fase 0 repository and run a controlled HEP pilot over 10-20 papers.

The next chat should not rebuild Fase 0. It should use the existing scripts and modify only one thing at a time.

## Current repository capabilities

The repository can already:

- download arXiv metadata from a query;
- download arXiv metadata from explicit IDs;
- download arXiv metadata from a plain-text ID-list file;
- run batch extraction with `fake` or `ollama` backend;
- run a controlled HEP pilot;
- generate summary, checklist, manifest, quality gate, and post-pilot analysis.

## Recommended Fase 1 starting point

Use a frozen metadata file or a plain-text ID-list file.

Preferred metadata-first flow:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 10 \
  --output outputs/metadata/hep_pilot_metadata.json

python scripts/run_hep_pilot.py \
  --metadata-list outputs/metadata/hep_pilot_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

Preferred ID-list flow:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --output outputs/metadata/hep_selected_ids_metadata.json

python scripts/run_hep_pilot.py \
  --metadata-list outputs/metadata/hep_selected_ids_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

Or direct pilot from ID-list file:

```bash
python scripts/run_hep_pilot.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --backend ollama \
  --model qwen3:0.6b
```

## Files to inspect after a pilot

Typical output directory:

```text
outputs/hep_pilot/
```

Important files:

```text
pilot_metadata.json
pilot_ollama_qwen3_0_6b.jsonl
pilot_ollama_qwen3_0_6b_summary.json
pilot_manual_quality_checklist.md
pilot_manual_review_summary.json
dataset_manifest.json
quality_gate_report.json
post_pilot_analysis.json
```

## Manual review step

After generating `pilot_manual_quality_checklist.md`, manually mark each checklist item and choose one outcome per record:

```text
pass | revise | reject | unclear
```

Then run:

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

## What to ask Codex first

Recommended prompt:

```text
We are starting KamiKnows Fase 1 from the existing Fase 0 repository. Do not rebuild the project. First inspect PROJECT_STATUS.md, docs/phase0_closure.md, docs/phase1_handoff.md, and README.md. Then verify the test suite. After that, run or help debug a controlled 10-20 paper HEP pilot using run_hep_pilot.py with Qwen/Ollama or frozen metadata. Do not add RAG, PDF parsing, LaTeX parsing, or fine-tuning yet.
```

## Fase 1 boundaries

Allowed next steps:

- improve prompt v1 after manual error analysis;
- run a second controlled 10-20 paper pilot;
- compare Qwen vs Mistral only on frozen metadata and with identical schema;
- improve schema only if real pilot errors justify it.

Not allowed yet:

- full RAG;
- large-scale scraping;
- training or LoRA;
- automatic discovery claims;
- 100+ paper corpus without a quality gate.
