# KamiKnows Fase 0 - Learning notes

These notes explain the current minimal repository in the order you should read it.
The goal is not to memorize files, but to understand the design pattern:

```text
small contract
-> small implementation
-> deterministic test
-> optional real model test
-> traceable output
```

KamiKnows must remain modular, testable, and corpus-centric. At this stage it is not a full scientific pipeline.
It is a learning repo for the first working pieces.

## 1. The current mental model

The current code has three small branches:

```text
Model branch:
  ModelPlugin
  -> FakeExtractionModel or OllamaPlugin

Extraction branch:
  title + abstract
  -> prompt
  -> model.generate(...)
  -> raw text
  -> parsed JSON
  -> validated extraction object

Ingestion branch:
  arXiv-style metadata
  -> metadata validation
  -> title + abstract available for extraction
```

The first connected path is:

```text
offline metadata JSON or arXiv metadata
-> validate_arxiv_metadata(...)
-> title + abstract
-> ModelPlugin backend
-> abstract_to_json(...)
-> append traceable JSONL record with run metadata
```

The current JSONL record shapes are documented separately in:

```text
docs/run_record_schema.md
```

The important idea is that the model is replaceable. The extraction code does not know if the backend is fake, Qwen, Mistral, DeepSeek, Ollama, Transformers, vLLM, or a future fine-tuned adapter.

## 2. Read this file first: `kamiknows/models/base.py`

This file defines the first model contract:

```python
class ModelPlugin:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        ...
```

This is deliberately small. It says:

```text
Give me a prompt.
Return generated text.
```

Why this matters:

- extraction code stays independent from the model provider;
- Qwen, Mistral, DeepSeek, and future adapters can share the same interface;
- tests can use fake models without Ollama or GPUs;
- benchmarking later becomes easier because the task code does not need to change.

Bad design would be:

```text
abstract_to_json_qwen_ollama_only(...)
```

Good design is:

```text
abstract_to_json(model: ModelPlugin, title: str, abstract: str)
```

## 3. Then read: `kamiknows/models/fake.py`

This file contains a deterministic fake backend.
It is not an LLM. It returns a known JSON string.

Purpose:

```text
test the software path without testing model quality
```

The fake backend is useful because LLM outputs can vary. If a test fails, you want to know whether the failure comes from:

```text
software bug
```

or from:

```text
model behavior / local Ollama / missing model / hardware / network
```

The fake backend isolates the software.

## 4. Then read: `kamiknows/models/ollama_plugin.py`

This file adapts local Ollama to the `ModelPlugin` interface.

The plugin sends an HTTP request to:

```text
http://localhost:11434/api/generate
```

with a payload containing:

```text
model name
prompt
stream = false
temperature
```

Then it returns the model response text.

Important point:

```text
Ollama details stay inside OllamaPlugin.
The rest of KamiKnows only sees model.generate(...).
```

This is what makes Qwen replaceable later.

## 5. Then read: `kamiknows/extraction/abstract_to_json.py`

This is the first real extraction module.

It contains four main steps:

```text
build_extraction_prompt(...)
parse_model_json(...)
validate_extraction(...)
abstract_to_json(...)
```

### 5.1 `build_extraction_prompt(...)`

This builds the instruction sent to the LLM.
The prompt asks for valid JSON only and tells the model not to invent scientific results beyond the input.

The schema is intentionally small:

```json
{
  "title": "string",
  "field": "string",
  "main_claim": "string",
  "method": "string",
  "limitations": "string",
  "confidence": "low | medium | high"
}
```

Why the schema is small:

```text
first observe model failures
then extend the schema
```

Do not start with 30 fields. A large schema makes early debugging harder.

### 5.2 `parse_model_json(...)`

LLMs often return extra text even when instructed not to.
For example:

```text
Here is the JSON:
{ ... }
```

or:

```text
```json
{ ... }
```
```

The parser tries to extract the first JSON object from the model output and parse it.
If it cannot find valid JSON, it raises an `ExtractionError`.

### 5.3 `validate_extraction(...)`

Parsing JSON is not enough.
This function checks that the parsed object has exactly the fields we expect.

For example, this is invalid:

```json
{
  "title": "Example",
  "confidence": "certain"
}
```

Reasons:

- missing required fields;
- `confidence` must be one of `low`, `medium`, `high`;
- extra fields are rejected.

This matters because KamiKnows must not silently store malformed model outputs.

### 5.4 `abstract_to_json(...)`

This is the orchestrator:

```text
prompt = build_extraction_prompt(...)
raw = model.generate(prompt, temperature=...)
parsed = parse_model_json(raw)
validated = validate_extraction(parsed)
return validated
```

It is the first minimal model-powered scientific extraction function.

## 6. Then read: `kamiknows/run_metadata.py`

This module creates the small `run` object saved with CLI-produced JSONL records.

It separates two questions:

```text
What did the model extract?
-> extraction object

How was that extraction produced?
-> run metadata
```

The current run metadata fields are:

```json
{
  "run_id": "...",
  "created_at": "...Z",
  "backend": "fake | ollama",
  "model": "fake | qwen3:0.6b | ...",
  "prompt_version": "abstract_to_json_v0"
}
```

Why this matters:

- two records from the same abstract can be compared;
- fake and real-model outputs are not confused;
- future prompt changes can be tracked;
- bad outputs can be traced back to the run that produced them.

## 7. Then read: `kamiknows/storage/jsonl.py`

This module persists validated records.

Main functions:

```text
append_jsonl_record(...)
append_extraction_jsonl(...)
read_jsonl_records(...)
```

JSONL means one JSON object per line:

```json
{"paper": "A"}
{"paper": "B"}
```

Why JSONL is useful now:

- easy to append;
- easy to inspect with `cat`;
- good for future datasets;
- does not require a database in Fase 0;
- compatible with many ML data workflows.

Important rule:

```text
validate before saving
```

The code should not store malformed LLM output and hope to fix it later.

## 8. Then read: `kamiknows/ingestion/arxiv_metadata.py`

This module validates arXiv-style metadata records.

Required fields:

```text
arxiv_id
title
authors
abstract
categories
published
url
```

It checks that key fields are present and have the expected type.

Why this exists:

```text
bad metadata
-> reject before model call

good metadata
-> use title + abstract for extraction
```

This keeps ingestion quality separate from model quality.

## 9. Then read: `kamiknows/ingestion/arxiv_downloader.py`

This module retrieves and parses minimal metadata from the arXiv Atom API.

Main functions:

```text
normalize_arxiv_id(...)
build_arxiv_query_url(...)
parse_arxiv_atom_feed(...)
fetch_arxiv_metadata_by_id(...)
search_arxiv_metadata(...)
```

Tests for this module use a local XML sample. They do not call the live network.

Why:

```text
unit tests should be deterministic
```

A separate script can call live arXiv when you explicitly choose to run it.

## 10. Understand the difference between tests and scripts

### Tests

Run:

```bash
pytest -q
```

Tests usually do not print useful output when they pass.
They answer:

```text
Did the code behave as expected?
```

A passing test looks like:

```text
42 passed, 2 skipped
```

The dots are good. Silence is normal.

### Scripts

Run:

```bash
python scripts/smoke_test_offline.py
```

or:

```bash
python scripts/run_fake_extraction.py
```

Scripts are for visible learning and manual inspection.
They answer:

```text
What did the mini-pipeline produce?
```

## 11. The main scripts

### `scripts/smoke_test_offline.py`

This is the recommended first command.
It does not require network or Ollama.

Run:

```bash
python scripts/smoke_test_offline.py
```

It executes:

```text
data/examples/arxiv_metadata_example.json
-> validate metadata
-> fake model
-> extract JSON
-> validate extraction
-> save JSONL
-> read back output
```

Expected final line:

```text
SMOKE TEST PASSED
```

### `scripts/run_fake_extraction.py`

This demonstrates extraction from a title and abstract.

Run fake backend:

```bash
python scripts/run_fake_extraction.py
```

Run Qwen via Ollama:

```bash
python scripts/run_fake_extraction.py --backend ollama --model qwen3:0.6b
```

Use Qwen only after:

```bash
ollama pull qwen3:0.6b
ollama serve
```

### `scripts/run_arxiv_lookup.py`

This retrieves metadata only.
It does not call an LLM.

Run:

```bash
python scripts/run_arxiv_lookup.py --query "cat:hep-ex AND calorimeter" --max-results 1
```

Use this to inspect what arXiv metadata looks like.

### `scripts/run_arxiv_extraction.py`

This connects metadata to extraction.

Offline run:

```bash
python scripts/run_arxiv_extraction.py \
  --metadata-file data/examples/arxiv_metadata_example.json \
  --backend fake
```

Live arXiv + Qwen run:

```bash
python scripts/run_arxiv_extraction.py \
  --query "cat:hep-ex AND calorimeter" \
  --backend ollama \
  --model qwen3:0.6b
```

Treat the Qwen output as experimental. Valid JSON does not mean scientifically correct extraction.

## 12. Read the tests in this order

Recommended order:

```text
1. tests/test_abstract_to_json.py
2. tests/test_jsonl_storage.py
3. tests/test_run_metadata.py
4. tests/test_integrated_save_extraction_basic.py
5. tests/test_arxiv_metadata.py
6. tests/test_arxiv_downloader.py
7. tests/test_run_fake_extraction_script.py
8. tests/test_run_arxiv_extraction_script.py
9. tests/test_smoke_test_offline_script.py
```

The first tests check small functions.
The later tests check connected behavior.

The Qwen/Ollama tests are optional:

```text
tests/test_qwen_ollama_basic.py
tests/test_qwen_abstract_to_json_ollama.py
```

Run them only when Ollama is installed and the model is pulled:

```bash
KAMIKNOWS_RUN_OLLAMA_TEST=1 pytest -s tests/test_qwen_ollama_basic.py
KAMIKNOWS_RUN_OLLAMA_TEST=1 pytest -s tests/test_qwen_abstract_to_json_ollama.py
```

The previous `KNOWKAMI_RUN_OLLAMA_TEST=1` flag is still accepted temporarily as
a deprecated alias.

## 13. How to inspect generated files

After the offline smoke test:

```bash
cat outputs/smoke_arxiv_extractions.jsonl
```

After the fake extraction demo:

```bash
cat outputs/extractions.jsonl
```

After the arXiv extraction script:

```bash
cat outputs/arxiv_extractions.jsonl
```

Each line is one JSON object. CLI-generated records now include a `run` object with `run_id`, `created_at`, `backend`, `model`, and `prompt_version`.
For easier reading, you can use Python:

```bash
python -m json.tool outputs/smoke_arxiv_extractions.jsonl
```

This works only when the file has one JSON line. For multi-line JSONL, inspect line by line.

## 14. What is mature and what is not

Mature enough for Fase 0:

```text
plugin interface
fake backend
Ollama backend smoke path
abstract-to-JSON schema validation
JSONL persistence
arXiv-style metadata validation
offline end-to-end smoke test
run metadata for CLI-generated JSONL records
```

Still early / not mature:

```text
quality of scientific extraction
model comparison
prompt robustness
real arXiv large-scale ingestion
PDF parsing
LaTeX parsing
chunking
RAG
fine-tuning
benchmarking
```

Do not interpret this repo as a scientific result yet.
It is a controlled software scaffold.

## 15. Common mistakes to avoid

### Mistake 1: expecting pytest to print the JSON

`pytest` is quiet when tests pass.
Use scripts when you want visible output.

### Mistake 2: treating the fake backend as a model

The fake backend is not a model.
It is deterministic scaffolding for software tests.

### Mistake 3: trusting valid JSON as scientific truth

Valid JSON means the output follows the schema.
It does not mean the extraction is scientifically correct.

### Mistake 4: adding many dependencies too early

Do not add `torch`, `transformers`, `vLLM`, vector databases, or PEFT until a micro-task needs them.

### Mistake 5: coupling extraction to one model

Keep all models behind `ModelPlugin`.
KamiKnows should benchmark models before choosing any default.

### Mistake 6: saving outputs without run metadata

A JSON extraction without `backend`, `model`, and `prompt_version` is hard to audit later.
The CLI scripts now save run metadata with generated records.

## 16. What to do next

The next useful micro-task is to add a small offline evaluation summary over the batch JSONL file. It should count records, confidence labels, backend/model usage, and obvious missing fields.

## Inspecting saved JSONL records

After you run a script that writes JSONL, you can inspect the result without opening the raw file manually:

```bash
python scripts/inspect_jsonl.py outputs/smoke_arxiv_extractions.jsonl
```

This prints a compact summary of each saved record:

```text
source -> arxiv_id or -
extraction -> title, confidence, main_claim
run -> backend, model, created_at
```

Use this as a learning/debugging tool. It does not judge whether the scientific extraction is correct; it only shows what was saved and how it was produced.


## Batch extraction demo

Use this command to process a tiny local list of arXiv-style metadata records without network access and without Ollama:

```bash
python scripts/run_batch_arxiv_extraction.py
```

It reads:

```text
data/examples/arxiv_metadata_batch_example.json
```

and writes traceable JSONL records to:

```text
outputs/batch_arxiv_extractions.jsonl
```

Then inspect the output:

```bash
python scripts/inspect_jsonl.py outputs/batch_arxiv_extractions.jsonl
```

This step proves that the same validated extraction path can handle multiple metadata records. With default settings it uses `FakeExtractionModel`, so it is a software-path check, not a scientific-quality check.

The same script can now also use remote arXiv:

```bash
python scripts/run_batch_arxiv_extraction.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 3 \
  --backend fake
```

or multiple explicit IDs:

```bash
python scripts/run_batch_arxiv_extraction.py \
  --ids 2301.00001 2301.00002 \
  --backend fake
```

And it can call Qwen through Ollama when you are ready for real model output:

```bash
ollama pull qwen3:0.6b
python scripts/run_batch_arxiv_extraction.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 2 \
  --backend ollama \
  --model qwen3:0.6b
```

Keep the distinction clear: remote arXiv tests ingestion; `--backend ollama` tests real LLM extraction; neither automatically proves scientific correctness.

## Offline JSONL summary/evaluation

After you create a batch output file:

```bash
python scripts/run_batch_arxiv_extraction.py
```

summarize it with:

```bash
python scripts/summarize_jsonl.py outputs/batch_arxiv_extractions.jsonl
```

This script answers basic engineering questions:

```text
How many records were saved?
Which confidence labels appear?
Which backend/model produced the records?
Which prompt_version was used?
Are obvious required fields missing?
Are confidence labels outside low/medium/high?
Can I save the summary as JSON for future automation?
```

To save a machine-readable report:

```bash
python scripts/summarize_jsonl.py \
  outputs/batch_arxiv_extractions.jsonl \
  --json-output outputs/batch_summary.json
```

It does **not** answer the scientific question:

```text
Is the extraction correct?
```

That distinction matters. In Fase 0, the evaluation is mostly about record shape and traceability. Scientific correctness comes later, after real model outputs, human review, and benchmark design.

Use this mental model:

```text
inspect_jsonl.py    -> show me the records in human-readable form
summarize_jsonl.py  -> count labels/backends, detect obvious schema issues, optionally write summary JSON
pytest              -> check code behavior automatically
```

## Model mini benchmark

The script `scripts/run_model_mini_benchmark.py` is the current formal mini-benchmark entry point.

It does not compare fake-vs-Ollama. It compares one selected backend/model across two query groups and checks only formal validity.

Conceptual flow:

```text
query group A -> metadata A -> extraction JSONL A -> summary A
query group B -> metadata B -> extraction JSONL B -> summary B
-> mini_benchmark_report.json
```

Useful command with Qwen via Ollama:

```bash
python scripts/run_model_mini_benchmark.py \
  --query-a "cat:hep-ex AND calorimeter" \
  --query-b "cat:hep-ex AND Higgs" \
  --max-results 2 \
  --backend ollama \
  --model qwen3:0.6b
```

Interpretation rule:

```text
PASS = records are complete, valid, traceable JSONL.
PASS does not mean the extraction is scientifically correct.
```

Prompt traceability is part of this check: each record stores `prompt_version`, `prompt_template_sha256`, and `extraction_schema_version`.

## Metadata ingestion vs model interpretation

From this point, keep two ideas separate.

Metadata ingestion is the arXiv-facing step:

```text
query or IDs -> arXiv API -> metadata JSON list
```

Model interpretation is the LLM-facing step:

```text
metadata JSON list -> title + abstract -> model backend -> extraction JSONL
```

They can run separately:

```bash
python scripts/download_arxiv_metadata_batch.py \
  --query "cat:hep-ex AND calorimeter" \
  --max-results 3 \
  --output outputs/metadata/calorimeter_metadata.json

python scripts/run_batch_arxiv_extraction.py \
  --metadata-list outputs/metadata/calorimeter_metadata.json \
  --backend ollama \
  --model qwen3:0.6b
```

Or together in the formal mini benchmark:

```bash
python scripts/run_model_mini_benchmark.py \
  --query-a "cat:hep-ex AND calorimeter" \
  --query-b "cat:hep-ex AND Higgs" \
  --backend ollama \
  --model qwen3:0.6b
```

The mini benchmark is not a scientific quality benchmark yet. It only checks that the selected backend produces complete, valid, traceable JSONL records over two query groups.


## Dataset manifest

After producing metadata, JSONL and summary files, create a manifest so the run
can be audited later:

```bash
python scripts/create_dataset_manifest.py   --from-mini-benchmark-dir outputs/model_mini_benchmark   --output outputs/model_mini_benchmark/dataset_manifest.json
```

The manifest is formal traceability only. It records paths, hashes, record
counts, backend/model and prompt identity. It does not validate scientific
correctness.

Relevant files:

```text
kamiknows/dataset_manifest.py
scripts/create_dataset_manifest.py
docs/dataset_manifest.md
```

## Manual quality review

After the formal mini benchmark, use `scripts/create_manual_quality_checklist.py`
to inspect a few records manually. This step compares the original abstract with
`main_claim`, `method` and `limitations`. It is deliberately human-reviewed:
formal JSON validity is not the same as scientific fidelity.

See `docs/manual_quality_checklist.md`.

## Manual review summary

Once you have generated and manually edited the Markdown checklist, summarize it
with:

```bash
python scripts/summarize_manual_quality_review.py \
  outputs/model_mini_benchmark/query_a_manual_quality_checklist.md \
  --json-output outputs/model_mini_benchmark/query_a_manual_review_summary.json
```

This is the first bridge from human scientific review back into a structured
artifact. It records review outcomes and checked fidelity criteria, but it still
does not automate scientific judgment.

## Dataset manifest after manual review

Formal JSONL summaries only say whether records are structurally complete. After
manual review, recreate `dataset_manifest.json` so the bundle also tracks:

```text
manual_quality_checklist
manual_review_summary
workflow_report
```

This keeps the chain explicit:

```text
metadata -> extraction JSONL -> formal summary -> checklist -> human review summary -> manifest
```


## Quality gate

After the formal benchmark and manual review summaries, run the quality gate:

```bash
python scripts/run_quality_gate.py \
  --manifest outputs/model_mini_benchmark/dataset_manifest.json \
  --output outputs/model_mini_benchmark/quality_gate_report.json
```

The quality gate does not re-evaluate the science automatically. It combines:

```text
dataset_manifest.json
manual_review_summary.json files
```

and returns:

```text
ACCEPT | REVISE | REJECT
```

Use `ACCEPT` as the clean condition before expanding to the 10-20 paper pilot.
Use `REVISE` to fix extraction, prompts or manual annotations. Use `REJECT` when
records or traceability are not usable.


## HEP pilot runner

`scripts/run_hep_pilot.py` is the first Fase 1 runner. It should be used when
moving from tutorial examples to a controlled 10-20 paper HEP pilot.

It keeps metadata ingestion and model interpretation conceptually separate:
remote arXiv query or frozen metadata file first, then Qwen/Ollama extraction
on the saved title+abstract records. The pilot still evaluates only formal
completeness automatically; scientific fidelity remains a manual checklist.

## Post-pilot analysis

After `run_hep_pilot.py`, the manual checklist, review summary, manifest and quality gate,
use `scripts/summarize_hep_pilot_run.py` to create `post_pilot_analysis.json`.

This is not a scientific evaluator. It is an operational status report that says whether
the current pilot is ready for the next controlled cycle or whether it needs revision.
