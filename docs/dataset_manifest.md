# KamiKnows dataset manifest

The dataset manifest is a small JSON inventory of files produced by a Fase 0 run.
It does **not** evaluate scientific correctness. It records where the files are,
how large they are, their SHA-256 hashes, JSONL record counts, and the run
context used to produce them.

## Why it exists

A KamiKnows run can produce several files:

```text
metadata JSON
extraction JSONL
formal summary JSON
mini benchmark report JSON
manual quality checklist Markdown
manual review summary JSON
dataset manifest JSON
```

Without a manifest, it becomes hard to know which metadata file, JSONL file,
summary, checklist, manual review summary, prompt hash, backend, and model belong together.

The manifest answers this formal question:

```text
Which files belong to this dataset/run, and are they present and traceable?
```

It does not answer:

```text
Are the extracted scientific claims correct?
```

## Create a manifest from a mini benchmark directory

After running:

```bash
python scripts/run_model_mini_benchmark.py \
  --metadata-a outputs/metadata/calorimeter_metadata.json \
  --metadata-b outputs/metadata/higgs_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

create or recreate the manifest with:

```bash
python scripts/create_dataset_manifest.py \
  --from-mini-benchmark-dir outputs/model_mini_benchmark \
  --output outputs/model_mini_benchmark/dataset_manifest.json
```

`run_model_mini_benchmark.py` also writes this manifest automatically unless
`--no-manifest` is used.

## Create a manifest from explicit files

```bash
python scripts/create_dataset_manifest.py \
  --metadata-file outputs/metadata/calorimeter_metadata.json \
  --jsonl-file outputs/qwen_calorimeter_extractions.jsonl \
  --summary-file outputs/qwen_calorimeter_summary.json \
  --benchmark-report outputs/model_mini_benchmark/mini_benchmark_report.json \
  --backend ollama \
  --model qwen3:0.6b \
  --output outputs/dataset_manifest.json
```


## Add manual review files to a manifest

After creating and completing manual checklist files, summarize them first:

```bash
python scripts/summarize_manual_quality_review.py \
  outputs/model_mini_benchmark/query_a_manual_quality_checklist.md \
  --json-output outputs/model_mini_benchmark/query_a_manual_review_summary.json
```

Then recreate the manifest from the benchmark directory. The command now
auto-registers files such as:

```text
query_a_manual_quality_checklist.md
query_b_manual_quality_checklist.md
query_a_manual_review_summary.json
query_b_manual_review_summary.json
benchmark_quality_workflow_report.json
```

```bash
python scripts/create_dataset_manifest.py \
  --from-mini-benchmark-dir outputs/model_mini_benchmark \
  --output outputs/model_mini_benchmark/dataset_manifest.json
```

You can also register them explicitly:

```bash
python scripts/create_dataset_manifest.py \
  --checklist-file outputs/model_mini_benchmark/query_a_manual_quality_checklist.md \
  --manual-review-summary-file outputs/model_mini_benchmark/query_a_manual_review_summary.json \
  --output outputs/dataset_manifest.json
```

## Manifest structure

The current manifest schema is intentionally small:

```json
{
  "manifest_version": "dataset_manifest_v0",
  "created_at": "...Z",
  "name": "kamiknows_model_mini_benchmark",
  "status": "PASS",
  "scope": "file inventory and formal traceability only; no scientific correctness judgment",
  "run_context": {
    "backend": "ollama",
    "model": "qwen3:0.6b",
    "prompt_version": "abstract_to_json_v0",
    "prompt_template_sha256": "...",
    "extraction_schema_version": "extraction_schema_v0"
  },
  "files": [
    {
      "path": "outputs/model_mini_benchmark/query_a_ollama_qwen3_0_6b.jsonl",
      "relative_path": "outputs/model_mini_benchmark/query_a_ollama_qwen3_0_6b.jsonl",
      "role": "jsonl_extractions",
      "exists": true,
      "size_bytes": 1234,
      "sha256": "...",
      "record_count": 2,
      "json_top_level_type": "jsonl"
    }
  ],
  "missing_paths": [],
  "notes": "..."
}
```

## Status meaning

`PASS` means all registered files exist and can be inventoried.

`WARN` means at least one registered path is missing. It does not necessarily
mean the data content is invalid, but the dataset bundle is incomplete.

## Current limitation

The manifest checks file presence, hashes and simple record counts. It can
register manual checklist and manual review summary files, but it does not
validate extraction semantics or scientific fidelity by itself. The actual
fidelity judgment remains a human review or later judge-model evaluation step.

## Relation to the quality gate

The dataset manifest is one of the inputs to the quality gate:

```bash
python scripts/run_quality_gate.py \
  --manifest outputs/model_mini_benchmark/dataset_manifest.json
```

The quality gate reads the manifest, discovers files with role
`manual_review_summary`, and produces an `ACCEPT`, `REVISE` or `REJECT` decision.
