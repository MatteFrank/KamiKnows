# KamiKnows HEP pilot

This document defines the first controlled Fase 1 pilot for KamiKnows.

The pilot is still abstract-level only. It does not download PDFs, parse LaTeX,
build a RAG index, fine-tune a model, or claim scientific correctness. It tests
whether the current extraction pipeline can process a small HEP corpus with
traceable outputs.

## Goal

Process 10-20 HEP papers from one controlled source of metadata and produce:

- a frozen metadata snapshot;
- a JSONL extraction file;
- a formal summary;
- a manual quality checklist over a small sample;
- a pilot report;
- a dataset manifest.

## Controlled query

The default query is:

```bash
cat:hep-ex AND calorimeter
```

This is intentionally narrow. It keeps the pilot focused on one HEP subdomain
where abstracts often mention detector response, simulation, reconstruction, or
calorimetry.

Other controlled queries can be used later, for example:

```bash
cat:hep-ex AND Higgs
cat:physics.ins-det AND detector
cat:hep-ex AND simulation AND calorimeter
```

Avoid broad queries at this stage. The aim is to debug the pipeline, not to
cover HEP.

## Recommended command

```bash
ollama pull qwen3:0.6b

python scripts/run_hep_pilot.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 10 \
  --backend ollama \
  --model qwen3:0.6b
```

Outputs are written to:

```text
outputs/hep_pilot/
  pilot_metadata.json
  pilot_ollama_qwen3_0_6b.jsonl
  pilot_ollama_qwen3_0_6b_summary.json
  pilot_manual_quality_checklist.md
  pilot_report.json
  dataset_manifest.json
```

## Separation of phases

The pilot keeps two concepts separate:

```text
metadata ingestion
-> arXiv query or frozen metadata file
-> pilot_metadata.json
```

and:

```text
model interpretation
-> title + abstract
-> Qwen/Ollama extraction
-> JSONL
```

You can also run the pilot from a frozen metadata file:

```bash
python scripts/run_hep_pilot.py \
  --metadata-list outputs/hep_pilot/pilot_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

This avoids calling arXiv again.

## Dry-run mode

For software tests only, use fake backend and allow a small sample:

```bash
python scripts/run_hep_pilot.py \
  --metadata-list data/examples/arxiv_metadata_batch_example.json \
  --backend fake \
  --allow-small-sample \
  --output-dir outputs/hep_pilot_dry_run
```

This is not a scientific benchmark.

## Formal status vs scientific quality

`pilot_report.json` and `*_summary.json` only check formal properties:

- record counts;
- required fields;
- valid confidence labels;
- run metadata;
- prompt/schema versioning;
- file traceability.

Scientific fidelity is checked by editing:

```text
outputs/hep_pilot/pilot_manual_quality_checklist.md
```

Then summarize the completed checklist:

```bash
python scripts/summarize_manual_quality_review.py \
  outputs/hep_pilot/pilot_manual_quality_checklist.md \
  --json-output outputs/hep_pilot/pilot_manual_review_summary.json
```

Regenerate the manifest:

```bash
python scripts/create_dataset_manifest.py \
  --from-mini-benchmark-dir outputs/hep_pilot \
  --output outputs/hep_pilot/dataset_manifest.json
```

Run the quality gate:

```bash
python scripts/run_quality_gate.py \
  --manifest outputs/hep_pilot/dataset_manifest.json \
  --output outputs/hep_pilot/quality_gate_report.json
```

## Success criterion

The pilot is successful only if:

1. the formal summary is `PASS`;
2. the manifest is complete;
3. the manual review summary is acceptable;
4. the quality gate returns `ACCEPT` or a clearly actionable `REVISE`.

Do not scale to more papers if the pilot is `REJECT` or if the review reveals
systematic unsupported claims.

## Using a plain-text arXiv ID list

For controlled pilots, a frozen list of IDs can be easier to review than a broad query. The file format is one ID or arXiv URL per line:

```text
# selected HEP pilot papers
2301.00001v1
arXiv:2301.00002v2
https://arxiv.org/abs/2301.00003v1
```

Download metadata from the list:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --output outputs/metadata/selected_ids_metadata.json
```

Run the pilot from the frozen metadata file:

```bash
python scripts/run_hep_pilot.py \
  --metadata-list outputs/metadata/selected_ids_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

Or run the pilot directly from the ID-list file:

```bash
python scripts/run_hep_pilot.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --backend ollama \
  --model qwen3:0.6b
```

The direct form still freezes metadata into `outputs/hep_pilot/pilot_metadata.json` before model interpretation.
