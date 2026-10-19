# Phase 2 Task 07b - Benchmark Citation Failure Debug v1

Date: 2026-07-02

## Objective

Diagnose P2T7 benchmark citation-validation failures and harden the reduced benchmark controls and future prompt/validator behavior.

The full P2T7 benchmark was not rerun in this task; only reduced validation or static diagnostics were performed.

## Inputs Used

- `outputs/rag_v1_fastcalosim_generator_benchmark_v1/benchmark_answers_v1.jsonl`
- existing P2T7 output dir: `outputs/rag_v1_fastcalosim_generator_benchmark_v1`

## Outputs Produced

- `outputs/rag_v1_fastcalosim_generator_benchmark_debug_v1/citation_failure_audit_v1.jsonl`
- `outputs/rag_v1_fastcalosim_generator_benchmark_debug_v1/citation_failure_summary_v1.json`
- `outputs/rag_v1_fastcalosim_generator_benchmark_debug_v1/prompt_profile_failure_summary_v1.json`
- `outputs/rag_v1_fastcalosim_generator_benchmark_debug_v1/context_label_mapping_summary_v1.json`
- `outputs/rag_v1_fastcalosim_generator_benchmark_debug_v1/benchmark_debug_manifest_v1.json`
- `docs/task_reports/phase2_task07b_benchmark_citation_debug_v1.md`

## Commands Run

- Static diagnostics over existing `benchmark_answers_v1.jsonl`
- `python3 scripts/run_rag_generator_benchmark_v1.py --config configs/phase2_p2t7_rag_generator_benchmark_v1.json --dry-run-matrix --limit-configs 2 --limit-questions 2`
- `pytest -q`

## Key Metrics

- Source answer records: 156
- Invalid citation records: 34
- Answered without valid citation records: 34
- Invented citation labels: 3
- Main suspected failure cause: `answer_status_answered_but_no_citation`
- Context label mapping status: `valid`
- Context label mapping issues: 0

## Failure Cause Counts

- `answer_status_answered_but_no_citation`: 14
- `citation_label_not_in_context`: 1
- `json_parse_fallback_lost_citations`: 14
- `model_returned_empty_citations_array`: 5

## Citation Validation Status Counts

- `answered_without_valid_citation`: 33
- `invented_citation`: 1

## Axis Highlights

### prompt_profile
- `grounded_json_v1`: invalid=19/78, causes={"answer_status_answered_but_no_citation": 8, "citation_label_not_in_context": 1, "json_parse_fallback_lost_citations": 6, "model_returned_empty_citations_array": 4}
- `strict_abstention_v2`: invalid=15/78, causes={"answer_status_answered_but_no_citation": 6, "json_parse_fallback_lost_citations": 8, "model_returned_empty_citations_array": 1}

### model_name
- `mistral:7b`: invalid=13/52, causes={"answer_status_answered_but_no_citation": 3, "citation_label_not_in_context": 1, "json_parse_fallback_lost_citations": 9}
- `qwen3:0.6b`: invalid=9/52, causes={"answer_status_answered_but_no_citation": 5, "json_parse_fallback_lost_citations": 3, "model_returned_empty_citations_array": 1}
- `qwen3:4b-instruct`: invalid=12/52, causes={"answer_status_answered_but_no_citation": 6, "json_parse_fallback_lost_citations": 2, "model_returned_empty_citations_array": 4}

## Code Hardening

- Future P2T7 prompts now state that `answer_status="answered"` requires a non-empty `citations` array containing at least one valid retrieved `citation_id`.
- Citation extraction now uses safer label boundaries while still accepting `[C1]` and `C1` forms.
- The benchmark CLI now supports reduced question/config selection and dry-run matrix inspection.

## Limits And Non-Claims

- No training, LoRA, QLoRA, discovery generation, corpus scaling, parser changes to source full text, or chunks_v1 changes were performed.
- Expected source IDs were not added to generation prompts.
- These diagnostics classify structural citation failures only; they do not establish factual entailment or scientific answer correctness.
- Manual grounding review remains required.

## Next Recommended Step

Use the debug audit to inspect the top invalid configs, then run at most a small reduced benchmark before choosing any manual-review queue.

Pytest status: pytest -q passed (225 passed, 2 skipped)
