# KamiKnows manual quality review summary

This document describes the step after generating and filling manual quality
checklists.

The workflow is:

```text
manual_quality_checklist.md
-> human reviewer edits checkboxes and outcome labels
-> summarize_manual_quality_review.py
-> manual_quality_review_summary.json
```

## Why this exists

The formal benchmark only checks structure: JSON validity, required fields,
run metadata and confidence label format.

The manual review checks scientific fidelity:

```text
Is the extracted main_claim supported by the abstract?
Is the extracted method supported by the abstract?
Are the limitations supported by the abstract?
Did the model introduce unsupported claims?
Is the confidence label plausible?
```

## How to fill the checklist

For each record, change the checkboxes from:

```markdown
- [ ] `main_claim` is directly supported by the abstract.
```

to:

```markdown
- [x] `main_claim` is directly supported by the abstract.
```

Then replace the outcome template:

```markdown
Review outcome: `pass | revise | reject | unclear`
```

with one value:

```markdown
Review outcome: pass
```

Allowed outcomes:

```text
pass
revise
reject
unclear
```

Use:

```markdown
- [x] Notes: short human note here
```

or leave notes unchecked/empty.

## Summarize the completed checklist

```bash
python scripts/summarize_manual_quality_review.py \
  outputs/model_mini_benchmark/query_a_manual_quality_checklist.md \
  --json-output outputs/model_mini_benchmark/query_a_manual_review_summary.json
```

Strict mode:

```bash
python scripts/summarize_manual_quality_review.py \
  outputs/model_mini_benchmark/query_a_manual_quality_checklist.md \
  --json-output outputs/model_mini_benchmark/query_a_manual_review_summary.json \
  --fail-on-non-pass
```

The strict mode returns exit code `1` unless every reviewed record has outcome
`pass` and all required checks are checked.

## Output JSON

The summary has this form:

```json
{
  "review_summary_version": "manual_review_summary_v0",
  "total_records": 3,
  "status": "PASS",
  "outcome_counts": {
    "pass": 3,
    "reject": 0,
    "revise": 0,
    "unclear": 0
  },
  "check_pass_counts": {
    "main_claim_supported": 3,
    "method_supported": 3,
    "limitations_supported": 3,
    "no_unsupported_claim": 3,
    "confidence_plausible": 3
  },
  "records": []
}
```

## Current scope

This is still a small Fase 0 quality gate. It does not replace expert review of
full papers. It only records whether abstract-level extractions look faithful on
a small sample.

## Add the review summary to the dataset manifest

After producing `*_manual_review_summary.json`, recreate the manifest so the
review artifact is tracked with hashes and file roles:

```bash
python scripts/create_dataset_manifest.py \
  --from-mini-benchmark-dir outputs/model_mini_benchmark \
  --output outputs/model_mini_benchmark/dataset_manifest.json
```

The manifest records the checklist Markdown with role
`manual_quality_checklist` and the parsed review JSON with role
`manual_review_summary`.

