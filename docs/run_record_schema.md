# KamiKnows run record schema - Fase 0

This document describes the JSONL records produced by the current Fase 0 scripts.
It is a working schema note, not a stable production contract.

The goal is traceability: every saved extraction should answer three questions:

```text
What source did the extraction come from?
What did the model extract?
How, when, and with which backend was it produced?
```

## 1. JSONL convention

KamiKnows uses JSONL for early tutorial outputs:

```text
one line = one JSON object = one extraction record
```

Example file:

```text
outputs/arxiv_extractions.jsonl
```

Each line must be a valid JSON object. Do not write pretty-printed multi-line JSON inside a JSONL file.

Correct JSONL shape:

```jsonl
{"source":{"arxiv_id":"0000.00000v1"},"extraction":{"confidence":"medium"},"run":{"backend":"fake"}}
{"source":{"arxiv_id":"0000.00001v1"},"extraction":{"confidence":"low"},"run":{"backend":"ollama"}}
```

Incorrect JSONL shape:

```json
{
  "source": {
    "arxiv_id": "0000.00000v1"
  }
}
```

That is valid JSON, but it is not one-line JSONL.

## 2. Record type A: simple extraction record

Produced by:

```bash
python scripts/run_fake_extraction.py
```

Top-level structure:

```json
{
  "extraction": {
    "title": "...",
    "field": "...",
    "main_claim": "...",
    "method": "...",
    "limitations": "...",
    "confidence": "medium"
  },
  "run": {
    "run_id": "...",
    "created_at": "...Z",
    "backend": "fake",
    "model": "fake",
    "prompt_version": "abstract_to_json_v0",
    "prompt_template_sha256": "...64 hex characters...",
    "extraction_schema_version": "extraction_schema_v0"
  }
}
```

Use this record type for demos where the input title and abstract are provided directly through CLI arguments.
It has no `source` block because it is not tied to an arXiv metadata record.

## 3. Record type B: arXiv-style extraction record

Produced by:

```bash
python scripts/run_arxiv_extraction.py \
  --metadata-file data/examples/arxiv_metadata_example.json \
  --backend fake
```

Also produced by the offline smoke test:

```bash
python scripts/smoke_test_offline.py
```

Top-level structure:

```json
{
  "source": {
    "arxiv_id": "0000.00000v1",
    "title": "Fast calorimeter simulation for high energy physics",
    "authors": ["Ada Example", "Bruno Example"],
    "categories": ["hep-ex", "physics.ins-det"],
    "published": "2026-01-01T00:00:00Z",
    "url": "https://arxiv.org/abs/0000.00000v1"
  },
  "extraction": {
    "title": "Fast calorimeter simulation for high energy physics",
    "field": "High Energy Physics / detector simulation",
    "main_claim": "...",
    "method": "...",
    "limitations": "...",
    "confidence": "medium"
  },
  "run": {
    "run_id": "...",
    "created_at": "...Z",
    "backend": "fake",
    "model": "fake",
    "prompt_version": "abstract_to_json_v0",
    "prompt_template_sha256": "...64 hex characters...",
    "extraction_schema_version": "extraction_schema_v0"
  }
}
```

Use this record type when the extraction comes from validated arXiv-style metadata.

## 4. `source` block

The `source` block records where the input text came from.

Required keys for the current Fase 0 arXiv-style record:

| Key | Type | Meaning |
|---|---:|---|
| `arxiv_id` | string | arXiv identifier, preferably with version when available. |
| `title` | string | Paper title used as extraction input. |
| `authors` | list[string] | Author names from arXiv metadata. |
| `categories` | list[string] | arXiv categories such as `hep-ex`. |
| `published` | string | Publication timestamp from metadata. |
| `url` | string | arXiv abs URL. |

Current validator:

```text
kamiknows/ingestion/arxiv_metadata.py
```

Important: `abstract` is validated in the metadata object before extraction, but it is not saved inside the compact `source` block yet. This is intentional for Fase 0. We may add an explicit `input` block later if we need full auditability of the exact text sent to the model.

## 5. `extraction` block

The `extraction` block records what the model produced after parsing and schema validation.

Required keys:

| Key | Type | Meaning |
|---|---:|---|
| `title` | string | Title copied or normalized by the extraction step. |
| `field` | string | Scientific field inferred from title/abstract. |
| `main_claim` | string | Main claim or core contribution stated by the abstract. |
| `method` | string | Method or approach described by the abstract. |
| `limitations` | string | Limitations stated or clearly implied by the abstract. |
| `confidence` | enum | One of `low`, `medium`, `high`. |

Current validator:

```text
kamiknows/extraction/abstract_to_json.py
```

Important: this is not a claim-level scientific schema yet. It is only a small tutorial extraction schema for Fase 0.

## 6. `run` block

The `run` block records how the extraction was produced.

Required keys:

| Key | Type | Meaning |
|---|---:|---|
| `run_id` | string | Unique ID for this extraction run. Currently a UUID hex string. |
| `created_at` | string | UTC timestamp ending with `Z`. |
| `backend` | string | Backend family, for example `fake` or `ollama`. |
| `model` | string | Concrete model identifier, for example `fake` or `qwen3:0.6b`. |
| `prompt_version` | string | Prompt/schema version label. Current default: `abstract_to_json_v0`. |

Current helper:

```text
kamiknows/run_metadata.py
```

Why this matters:

```text
same source + same prompt + different model = different record
same source + same model + different prompt_version = different record
same output text + different run_id = separate execution
```

## 7. Current prompt version

Current default:

```text
abstract_to_json_v0
```

This label refers to the current small prompt and schema in:

```text
kamiknows/extraction/abstract_to_json.py
```

Change `prompt_version` when the instruction, schema, field definitions, or validation behavior changes in a way that can affect outputs.

Do not silently change extraction behavior while keeping the same prompt version once comparisons matter.

## 8. What is stable now

Stable enough for Fase 0 tutorial use:

```text
source/extraction/run top-level separation
run_id
created_at
backend
model
prompt_version
small extraction schema
JSONL as one-record-per-line format
```

Not stable yet:

```text
full paper schema
claim-level schema
equation/table schemas
chunk schema
RAG citation schema
benchmark result schema
fine-tuning dataset schema
```

## 9. Minimal inspection commands

Run the offline smoke test:

```bash
python scripts/smoke_test_offline.py
```

Inspect the generated JSONL:

```bash
cat outputs/smoke_arxiv_extractions.jsonl
```

Pretty-print the last JSONL record:

```bash
tail -n 1 outputs/smoke_arxiv_extractions.jsonl | python -m json.tool
```

Check only run metadata manually:

```bash
tail -n 1 outputs/smoke_arxiv_extractions.jsonl | python -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.read())["run"], indent=2))'
```

## 10. Design rule

Do not save raw LLM output as a final dataset record.

Use this order:

```text
model output
-> parse JSON
-> validate extraction schema
-> add source metadata when available
-> add run metadata
-> append one JSONL record
```

This keeps Fase 0 small but already aligned with the future KamiKnows requirement: outputs must be traceable, inspectable, and reproducible enough to compare models and prompts.

## Human-readable inspection

Use the inspection CLI to summarize a small JSONL output file:

```bash
python scripts/inspect_jsonl.py outputs/smoke_arxiv_extractions.jsonl
```

This command is read-only. It reports record count, title, source identifier, confidence, backend/model, timestamp, and a short main claim.


## Batch JSONL records

The batch runner writes the same arXiv-style traceable record shape, one record per metadata item. The source can be a local metadata list, a remote arXiv query, or a list of remote arXiv IDs:

```bash
python scripts/run_batch_arxiv_extraction.py
```

Default input:

```text
data/examples/arxiv_metadata_batch_example.json
```

Default output:

```text
outputs/batch_arxiv_extractions.jsonl
```

Each line still follows the same top-level contract:

```text
source + extraction + run
```

The main difference is cardinality: the batch script appends multiple JSONL lines in one execution. The safe default is still `--backend fake`, which validates software flow and record shape. With `--backend ollama`, the same record shape is used for real model output, but quality must be reviewed separately.

## Offline summary/evaluation command

Use the summary CLI to check a JSONL file at record-shape level:

```bash
python scripts/summarize_jsonl.py outputs/batch_arxiv_extractions.jsonl
```

It reports:

```text
Records
Records with source block
Confidence labels
Backends
Models
Prompt versions
Prompt template SHA-256
Extraction schema versions
Missing fields
Invalid confidence labels
Evaluation status: PASS | WARN
```

The same information can be saved as JSON:

```bash
python scripts/summarize_jsonl.py \
  outputs/batch_arxiv_extractions.jsonl \
  --json-output outputs/batch_summary.json
```

The JSON report has this shape:

```json
{
  "total_records": 3,
  "records_with_source": 3,
  "require_source": true,
  "confidence_counts": {"medium": 3},
  "backend_counts": {"fake": 3},
  "model_counts": {"fake": 3},
  "prompt_version_counts": {"abstract_to_json_v0": 3},
  "prompt_template_sha256_counts": {"...": 3},
  "extraction_schema_version_counts": {"extraction_schema_v0": 3},
  "missing_fields": {},
  "invalid_confidence_records": [],
  "evaluation_status": "PASS"
}
```

Default behavior expects arXiv-style records with:

```text
source + extraction + run
```

For simple records produced by `scripts/run_fake_extraction.py`, where the `source` block is intentionally absent, use:

```bash
python scripts/summarize_jsonl.py outputs/extractions.jsonl --allow-simple-records
```

This command is an engineering check, not a scientific judge. A `PASS` status means the record shape is complete enough for Fase 0. It does not mean the extracted scientific content is correct.

## Model mini benchmark outputs

`python scripts/run_model_mini_benchmark.py` writes outputs for two query groups.

Default location:

```text
outputs/model_mini_benchmark/
```

Main files:

```text
metadata_query_a.json
metadata_query_b.json
query_a_<backend>_<model>.jsonl
query_b_<backend>_<model>.jsonl
query_a_<backend>_<model>_summary.json
query_b_<backend>_<model>_summary.json
mini_benchmark_report.json
```

`mini_benchmark_report.json` records metadata source, record counts, extraction timing, formal summary status, backend/model, and prompt identity.

The benchmark checks formal validity only. It does not compare fake-vs-Ollama and it does not judge scientific correctness.


## Source abstract for manual review

Current batch extraction records include `source.abstract` when the source
metadata contains it. This makes manual quality review self-contained:

```text
source.abstract -> compare against extraction.main_claim, extraction.method, extraction.limitations
```

Older JSONL files may not contain `source.abstract`; in that case use
`scripts/create_manual_quality_checklist.py --metadata-list ...` to recover the
abstracts from saved arXiv metadata.
