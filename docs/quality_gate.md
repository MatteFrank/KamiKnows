# KamiKnows quality gate

The quality gate is the final control step for the current Fase 0 workflow. It
turns already-produced formal checks and human review summaries into one
operational decision:

```text
ACCEPT | REVISE | REJECT
```

It does **not** judge scientific correctness automatically. It only combines:

1. the formal dataset manifest;
2. the manual quality review summaries compiled by a human reviewer.

## Inputs

Typical inputs are produced by the benchmark quality workflow:

```text
outputs/model_mini_benchmark/dataset_manifest.json
outputs/model_mini_benchmark/query_a_manual_review_summary.json
outputs/model_mini_benchmark/query_b_manual_review_summary.json
```

The manifest registers metadata files, extraction JSONL files, formal summaries,
benchmark reports, checklist Markdown files and manual review summaries.

## Command

After generating and summarizing the manual checklists, run:

```bash
python scripts/run_quality_gate.py \
  --manifest outputs/model_mini_benchmark/dataset_manifest.json \
  --output outputs/model_mini_benchmark/quality_gate_report.json
```

The script discovers files with role `manual_review_summary` from the manifest.
You can also pass them explicitly:

```bash
python scripts/run_quality_gate.py \
  --manifest outputs/model_mini_benchmark/dataset_manifest.json \
  --manual-review-summary outputs/model_mini_benchmark/query_a_manual_review_summary.json \
  --manual-review-summary outputs/model_mini_benchmark/query_b_manual_review_summary.json \
  --output outputs/model_mini_benchmark/quality_gate_report.json
```

## Decision logic

`ACCEPT` means:

```text
manifest status is PASS
no missing files are reported
at least one manual review summary is present
all manual review summaries are PASS
all reviewed records are fully checked
```

`REVISE` means:

```text
manual review is missing
or at least one reviewed record needs revision
or review is incomplete/unclear
or manifest has a non-PASS warning without missing files
```

`REJECT` means:

```text
manifest reports missing files
or at least one manual review summary contains rejected records
```

This is intentionally conservative. Before moving to a 10-20 paper pilot, the
run should normally be `ACCEPT`.

## Strict mode

For CI-style usage:

```bash
python scripts/run_quality_gate.py \
  --manifest outputs/model_mini_benchmark/dataset_manifest.json \
  --fail-on-non-accept
```

The command returns exit code `1` unless the decision is `ACCEPT`.

## Formal-only check

For debugging only, you can disable the manual review requirement:

```bash
python scripts/run_quality_gate.py \
  --manifest outputs/model_mini_benchmark/dataset_manifest.json \
  --allow-unreviewed
```

Do not use this as a scientific approval gate.
