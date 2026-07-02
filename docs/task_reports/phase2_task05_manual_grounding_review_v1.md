# Phase 2 Task 05 - Manual Grounding Review and Eval Set Hardening v1

Date: 2026-07-02

## Objective

Create manual review artifacts for P2T4 generated answers, classify candidate grounding and answer-quality failure modes for human review, harden the eval set draft, and prepare static v1 eval questions as data.

Manual review artifacts support groundedness assessment, but pending human review means answer quality is not yet validated.

## Inputs P2T4 Used

QA output directory: `outputs/rag_v1_fastcalosim_qa_grounded_config_v1`

- `outputs/rag_v1_fastcalosim_qa_grounded_config_v1/rag_eval_set_v1.jsonl`
- `outputs/rag_v1_fastcalosim_qa_grounded_config_v1/retrieved_contexts_v1.jsonl`
- `outputs/rag_v1_fastcalosim_qa_grounded_config_v1/citation_audit_v1.jsonl`
- `outputs/rag_v1_fastcalosim_qa_grounded_config_v1/citation_validation_summary_v1.json`
- `outputs/rag_v1_fastcalosim_qa_grounded_config_v1/rag_eval_summary_v1.json`
- `outputs/rag_ready_fastcalo_v1/chunks_v1.jsonl`
- `outputs/rag_v1_fastcalosim_qa_grounded_config_v1/rag_answers_qwen3_0_6b_v1.jsonl`
- `outputs/rag_v1_fastcalosim_qa_grounded_config_v1/rag_answers_qwen3_4b_instruct_v1.jsonl`
- `outputs/rag_v1_fastcalosim_qa_grounded_config_v1/rag_answers_mistral_7b_v1.jsonl`

## Outputs Produced

- `outputs/rag_v1_fastcalosim_manual_review/manual_review_queue_v1.jsonl`
- `outputs/rag_v1_fastcalosim_manual_review/manual_review_sheet_v1.csv`
- `outputs/rag_v1_fastcalosim_manual_review/manual_review_guidelines_v1.md`
- `outputs/rag_v1_fastcalosim_manual_review/eval_question_audit_v1.jsonl`
- `outputs/rag_v1_fastcalosim_manual_review/eval_question_audit_summary_v1.json`
- `outputs/rag_v1_fastcalosim_manual_review/rag_eval_set_v1_hardened_draft.jsonl`
- `outputs/rag_v1_fastcalosim_manual_review/rag_eval_set_hardening_notes_v1.md`
- `outputs/rag_v1_fastcalosim_manual_review/manual_review_summary_v1.json`
- `data/eval/rag_eval_set_v1_hardened.jsonl`
- `configs/phase2_p2t5_manual_review_v1.json`
- `docs/task_reports/phase2_task05_manual_grounding_review_v1.md`

## Review Records

- Responses to review: 39
- Questions: 13
- Models involved: qwen3:0.6b, qwen3:4b-instruct, mistral:7b
- Manual review status: {"pending": 39}

## Citation Status

- Citation validation clean: True
- Invalid citation records: 0
- Invented citation labels: 0
- Answered without valid citation: 0
- Status counts: `{"insufficient_evidence_no_citation": 3, "valid": 36}`

## Problematic Questions

- Questions needing rewrite or expected-source review: eval_v0_q001, eval_v0_q002, eval_v0_q003, eval_v0_q004, eval_v0_q005, eval_v0_q006, eval_v0_q007, eval_v0_q008, eval_v1_q012, eval_v1_q013
- Retrieval misses: eval_v0_q002, eval_v0_q005
- Retrieval misses highlighted for P2T5: eval_v0_q002, eval_v0_q005

## What Was Automated

- Built a pending review queue with one record per model/question answer.
- Built a CSV sheet for human completion.
- Joined answer records with retrieved chunks, retrieval eval, citation validation, and citation audit data.
- Flagged deterministic candidate failure modes such as retrieval misses, source-coverage gaps, generation warnings, and context quality flags.
- Audited eval questions for ambiguous wording, multi-source requirements, no-answer behavior, and rewrite needs.
- Produced a hardened eval-set draft without changing ground truth automatically.

## What Remains Manual

- Scientific groundedness assessment.
- Answer correctness assessment.
- Citation support assessment beyond formal label validity.
- Severity and reviewer notes.
- Acceptance of any eval-set rewrite or expected-source change.

## Key Metrics

- P2T4 expected-source hit rate: 0.8333333333333334
- P2T4 expected-source hits: 10 / 12
- Chunks available for traceability: 108
- Known candidate failure modes: `{"generation_or_audit_warning": 9, "multi_source_retrieval_incomplete": 15, "possible_incomplete_answer_or_over_abstention": 2, "possible_overclaim_or_failed_abstention": 2, "retrieval_miss": 6, "retrieved_context_quality_flags": 30}`

## Config Notes

- Review schema options: `{"manual_abstention_behavior": ["pass", "partial", "fail", "pending"], "manual_answer_correctness": ["pass", "partial", "fail", "pending"], "manual_citation_support": ["pass", "partial", "fail", "pending"], "manual_grounding": ["pass", "partial", "fail", "pending"], "severity": ["none", "minor", "major", "critical", "pending"]}`
- Hardening settings: `{"canonical_hardened_eval_set_file": "data/eval/rag_eval_set_v1_hardened.jsonl", "do_not_change_ground_truth_automatically": true, "flag_ambiguous_wording": true, "flag_multi_source_questions": true, "flag_no_answer_questions": true, "highlight_retrieval_misses": ["eval_v0_q002", "eval_v0_q005"], "llm_as_final_judge_enabled": false, "llm_question_generation_enabled": false, "preserve_original_question_ids": true}`

## Limits

- No training, LoRA, QLoRA, discovery, corpus scaling, parser changes, or chunks_v1 changes were performed.
- P2T4 outputs and RAG-ready v0/v1 outputs were read, not overwritten.
- Automated citation validation checks labels only, not factual entailment.
- Suggested failure modes are triage hints for a human reviewer, not final judgments.
- All scientific judgments remain pending until a human reviewer fills the sheet.

## Recommended Next Step

Complete `manual_review_sheet_v1.csv`, then summarize human labels before claiming any answer-quality or groundedness improvement.
