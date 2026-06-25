# KamiKnows model mini benchmark

This note defines the Fase 0 mini benchmark after dropping the fake-vs-Ollama comparison.

## Goal

The mini benchmark checks whether one model/backend can process two small arXiv query groups and produce formally valid KamiKnows JSONL records.

It does **not** judge scientific correctness yet.

It checks only:

- metadata can be downloaded or loaded;
- each metadata record has the required source fields;
- the model output can be parsed as JSON;
- extraction fields are present;
- `confidence` is one of `low`, `medium`, `high`;
- run metadata is present;
- JSONL and summary JSON files are produced.

## Two steps must stay separate

KamiKnows separates ingestion from interpretation.

### Step 1: metadata ingestion

This step talks to arXiv and saves metadata only:

```text
arXiv query/IDs
-> metadata records
-> validated JSON list
```

Command example:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 3 \
  --output outputs/metadata/calorimeter_metadata.json
```

This does not call Ollama and does not interpret the abstract.

### Step 2: model interpretation

This step reads already available metadata and calls the selected model/backend:

```text
metadata JSON list
-> title + abstract
-> Ollama/Qwen or another backend
-> extraction JSON
-> formal summary
```

Command example:

```bash
python scripts/run_batch_arxiv_extraction.py \
  --metadata-list outputs/metadata/calorimeter_metadata.json \
  --backend ollama \
  --model qwen3:0.6b \
  --output outputs/qwen_calorimeter_extractions.jsonl
```

This does not need to call arXiv if the metadata file already exists.

## Run both steps together

For convenience, the mini benchmark can download metadata and run extraction in one command:

```bash
python scripts/run_model_mini_benchmark.py \
  --query-a "cat:hep-ex AND calorimeter" \
  --query-b "cat:hep-ex AND Higgs" \
  --max-results 2 \
  --backend ollama \
  --model qwen3:0.6b
```

The script still keeps the phases internally separate:

```text
query A -> metadata_query_a.json -> query_a JSONL -> query_a summary
query B -> metadata_query_b.json -> query_b JSONL -> query_b summary
```

## Run with pre-downloaded metadata

This is more reproducible:

```bash
python scripts/run_model_mini_benchmark.py \
  --metadata-a outputs/metadata/calorimeter_metadata.json \
  --metadata-b outputs/metadata/higgs_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

In this mode, arXiv is not contacted at all.

## Suggested query pairs

Start with small query groups:

```text
cat:hep-ex AND calorimeter
cat:hep-ex AND Higgs
```

Other useful pairs:

```text
cat:hep-ex AND simulation AND calorimeter
cat:physics.ins-det AND detector
```

```text
cat:hep-ex AND Higgs AND LHC
cat:hep-ph AND Higgs
```

## Interpretation of results

The benchmark output is formal.

`PASS` means:

```text
records were produced
required fields exist
confidence labels are valid
summary JSON was generated
```

`PASS` does not mean:

```text
the extraction is scientifically correct
the model understood the paper
the model is better than another model
```

Scientific evaluation comes later with manual review, judge models, and explicit rubrics.

## Prompt identity in benchmark records

Each JSONL record and each benchmark report includes prompt identity fields:

```json
{
  "prompt_version": "abstract_to_json_v0",
  "prompt_template_sha256": "...64 hex characters...",
  "extraction_schema_version": "extraction_schema_v0"
}
```

This makes the benchmark auditable: if the prompt template changes, the hash changes. If the extraction schema semantics change, the schema version should change.

## One-command benchmark plus manual checklist

After the formal benchmark is working, use:

```bash
python scripts/run_benchmark_quality_workflow.py \
  --metadata-a outputs/metadata/calorimeter_metadata.json \
  --metadata-b outputs/metadata/higgs_metadata.json \
  --backend ollama \
  --model qwen3:0.6b \
  --review-limit 3
```

This runs the same formal benchmark and then creates:

```text
query_a_manual_quality_checklist.md
query_b_manual_quality_checklist.md
benchmark_quality_workflow_report.json
```

The workflow is still split conceptually:

```text
metadata ingestion -> model interpretation -> formal checks -> manual review sheet
```

The manual review sheet is not a score. It is a compact place to check whether
`main_claim`, `method` and `limitations` are supported by the abstract.

## ID-list files and benchmark discipline

For longer or more curated paper selections, prefer this two-step flow:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --ids-file data/examples/arxiv_ids_example.txt \
  --output outputs/metadata/selected_ids_metadata.json
```

Then pass the frozen metadata file to the benchmark or pilot scripts. This keeps arXiv download separate from model interpretation and makes reruns reproducible.
