# KamiKnows manual quality checklist

This document describes the first manual quality-review step in KamiKnows Fase 0.

The automatic checks in the mini benchmark are formal only. They verify that JSON
is parseable, required fields exist, run metadata is present and confidence
labels are valid. They do not verify scientific fidelity.

The manual checklist is the next controlled step:

```text
extraction JSONL
-> select 2-3 records
-> show abstract + extracted fields
-> human checks fidelity
-> notes are written in Markdown
```

## Why this is separate from formal validation

Formal validation answers:

```text
Is the record complete and machine-readable?
```

Manual quality review answers:

```text
Are main_claim, method and limitations supported by the abstract?
```

These are different tasks. A record can be formally valid but scientifically
weak or over-interpreted.

## Create a checklist from a JSONL file

```bash
python scripts/create_manual_quality_checklist.py \
  outputs/model_mini_benchmark/query_a_ollama_qwen3_0_6b.jsonl \
  --limit 3 \
  --output outputs/manual_quality_checklist.md
```

Then open:

```text
outputs/manual_quality_checklist.md
```

and review each record manually.

## If abstracts are missing from older JSONL files

Current batch records include `source.abstract` so the checklist is self-contained.
Older records may not. In that case, pass the metadata JSON used to create them:

```bash
python scripts/create_manual_quality_checklist.py \
  outputs/qwen_calorimeter_extractions.jsonl \
  --metadata-list outputs/metadata/calorimeter_metadata.json \
  --output outputs/manual_quality_checklist.md
```

## What to check

For each record, check:

```text
main_claim  -> directly supported by the abstract?
method      -> directly supported by the abstract?
limitations -> directly supported by the abstract?
```

Also check whether the model introduced unsupported scientific claims or used a
confidence label that looks too strong.

Recommended outcome labels:

```text
pass
revise
reject
unclear
```

## Current scope

This is still not a full benchmark. It is a small manual review step before
processing more papers or comparing more models.

## Generate checklists directly after a mini-benchmark

For convenience, this workflow runs the formal mini-benchmark and immediately
creates checklist files for both query groups:

```bash
python scripts/run_benchmark_quality_workflow.py \
  --metadata-a outputs/metadata/calorimeter_metadata.json \
  --metadata-b outputs/metadata/higgs_metadata.json \
  --backend ollama \
  --model qwen3:0.6b \
  --review-limit 3
```

Outputs:

```text
outputs/model_mini_benchmark/query_a_manual_quality_checklist.md
outputs/model_mini_benchmark/query_b_manual_quality_checklist.md
```

Open those Markdown files and fill the checkboxes manually. Do not treat the
formal benchmark `PASS` as a scientific quality score.
