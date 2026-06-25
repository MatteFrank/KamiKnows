# KamiKnows project status

## Current status

Fase 0 is complete as a tutorial and minimum operational base.

The repository supports a controlled abstract-level workflow:

```text
arXiv metadata
-> title + abstract
-> replaceable ModelPlugin
-> extraction JSON
-> formal validation
-> traceable JSONL
-> formal summary
-> manual quality checklist
-> manual review summary
-> dataset manifest
-> quality gate
-> post-pilot analysis
```

## Current phase boundary

### Fase 0 completed

Implemented and tested:

- Python package skeleton.
- Model plugin interface.
- Fake backend for deterministic software tests.
- Ollama backend for local open-weight models such as Qwen.
- Abstract-to-JSON extraction schema.
- arXiv metadata download and validation.
- Batch extraction over local metadata, remote query, explicit IDs, and ID-list files.
- HEP pilot runner for 10-20 paper controlled abstract pilots.
- Formal summaries, manual review checklist, dataset manifest, quality gate, and post-pilot analysis.

### Fase 1 next

Use Codex in a fresh chat to continue from the stable Fase 0 repository.

Recommended first Fase 1 task:

```text
Run a controlled 10-20 paper HEP pilot from a frozen metadata file or ID-list file, then inspect summary, manual review, quality gate, and post-pilot analysis before changing model, prompt, or schema.
```

## Current model policy

- Qwen via Ollama is the first practical backend.
- Fake backend is only for tests and dry-runs.
- Do not select a definitive backbone model without benchmark evidence.
- Mistral and DeepSeek remain candidates for later controlled comparison.

## Current data policy

- Prefer arXiv metadata and abstract-level extraction first.
- Freeze metadata snapshots before LLM interpretation.
- Keep source metadata, extraction, and run metadata separate.
- Do not treat extracted claims as verified scientific truth.

## Main commands

Run tests:

```bash
pytest -q
```

Download metadata from a query:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 10 \
  --output outputs/metadata/calorimeter_metadata.json
```

Download metadata from a plain-text ID list:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --output outputs/metadata/selected_ids_metadata.json
```

Run HEP pilot from frozen metadata:

```bash
python scripts/run_hep_pilot.py \
  --metadata-list outputs/metadata/calorimeter_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

Run HEP pilot from an ID-list file:

```bash
python scripts/run_hep_pilot.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --backend ollama \
  --model qwen3:0.6b
```

## Stop condition before scaling

Before increasing paper count or adding RAG/PDF parsing, require:

- formal summary is `PASS`;
- manual review summary exists;
- quality gate has clear `ACCEPT` or documented `REVISE` reasons;
- post-pilot analysis has been read.
