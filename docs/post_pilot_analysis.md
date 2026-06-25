# KamiKnows post-pilot analysis

`post_pilot_analysis.json` is an operational summary created after a controlled
HEP pilot run.

It does **not** judge scientific truth automatically. It reads artifacts that
already exist:

- `dataset_manifest.json`
- formal JSONL summaries
- manual review summaries, when present
- `quality_gate_report.json`, when present
- pilot report and registered JSONL files

The goal is to answer one practical question:

```text
What should we do after this pilot run?
```

## Command

From the repository root:

```bash
python scripts/summarize_hep_pilot_run.py \
  --manifest outputs/hep_pilot/dataset_manifest.json \
  --output outputs/hep_pilot/post_pilot_analysis.json
```

If the quality gate report is not next to the manifest, pass it explicitly:

```bash
python scripts/summarize_hep_pilot_run.py \
  --manifest outputs/hep_pilot/dataset_manifest.json \
  --quality-gate-report outputs/hep_pilot/quality_gate_report.json \
  --output outputs/hep_pilot/post_pilot_analysis.json
```

## Recommendations

The report can return:

- `READY_FOR_NEXT_PILOT_CYCLE`: the manifest is complete and the quality gate accepted the run.
- `REVISE_BEFORE_SCALING`: the run needs review completion or revision before scaling.
- `STOP_AND_FIX`: the quality gate rejected at least one reviewed record.
- `FIX_TRACEABILITY`: the manifest has missing files or broken paths.
- `RUN_QUALITY_GATE`: the quality gate report is missing.
- `INVESTIGATE_GATE_STATUS`: unexpected quality gate state.

## Typical post-pilot flow

```text
run_hep_pilot.py
→ complete pilot_manual_quality_checklist.md
→ summarize_manual_quality_review.py
→ create_dataset_manifest.py
→ run_quality_gate.py
→ summarize_hep_pilot_run.py
```

## Difference from the manifest

`dataset_manifest.json` is an inventory of files.

`post_pilot_analysis.json` interprets that inventory together with formal summaries,
manual review summaries and the quality gate report to suggest the next operational step.
