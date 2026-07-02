# Phase 2 Task 05b - Eval Externalization and Hardening

Date: 2026-07-02

## Objective

Externalize the complete RAG eval set into one canonical data file and remove
operational question text from the QA runtime.

P2T5b was necessary because P2T5 showed that several questions needed clearer
source constraints, multi-source rules, and insufficient-evidence behavior. A
file-backed eval set is easier to audit, diff, reuse, and rerun than Python
constants.

The hardened eval set improves test clarity and reproducibility, but it does not replace human grounding review.

## What Moved Out Of Code

- The v1 extension questions are no longer stored as operational Python data.
- `kamiknows/rag/qa_grounded_v1.py` now loads eval questions from JSONL when
  `eval_questions_file` is provided.
- The deprecated `_additional_eval_questions_v1()` helper contains no question
  text and is not used by runtime loading.

## Canonical File

```text
data/eval/rag_eval_set_v1_hardened.jsonl
```

Human-readable companion:

```text
data/eval/rag_eval_set_v1_hardened.md
```

## Eval Set Counts

- Total questions: 13
- Hardened v0 questions: 8 (`eval_v0_q001` through `eval_v0_q008`)
- Hardened v1 questions: 5 (`eval_v1_q009` through `eval_v1_q013`)
- Insufficient-evidence questions: `eval_v0_q008`, `eval_v1_q013`
- Multi-source questions requiring all expected sources: `eval_v0_q005`,
  `eval_v0_q006`, `eval_v1_q012`
- Questions still flagged for manual question review: `eval_v0_q002`,
  `eval_v0_q005`, `eval_v0_q006`, `eval_v0_q007`, `eval_v0_q008`,
  `eval_v1_q012`
- Questions still flagged for expected-source review: `eval_v0_q002`,
  `eval_v0_q005`, `eval_v0_q006`, `eval_v0_q007`, `eval_v0_q008`,
  `eval_v1_q012`

## Runtime Changes

- Added `load_eval_questions_from_file(path)` with validation for file
  existence, JSONL records, required fields, and duplicate `question_id`.
- `build_eval_set_v1()` uses only `eval_questions_file` when provided.
- `eval_set.mode=from_file` requires `questions_file`.
- Legacy v0-plus-static composition remains available only with an explicit
  static file path.
- Retrieval evaluation now respects `required_all_expected_sources` for
  multi-source questions.
- `qa_run_config_v1.json` and `qa_manifest_v1.json` now record the eval
  question file.

## Config

The P2T4 config now points to:

```json
{
  "mode": "from_file",
  "questions_file": "data/eval/rag_eval_set_v1_hardened.jsonl",
  "static_v1_questions_enabled": false,
  "llm_question_generation_enabled": false
}
```

## Rerun Command

To rerun QA grounded v1 without overwriting existing P2T4 outputs, use a new
output directory:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --config configs/phase2_p2t4_qa_grounded_v1.json \
  --output-dir outputs/rag_v1_fastcalosim_qa_grounded_hardened_v1 \
  --report-path docs/task_reports/phase2_task04_qa_grounded_hardened_v1.md
```

Equivalent explicit eval-file override:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_qa_grounded_hardened_v1 \
  --report-path docs/task_reports/phase2_task04_qa_grounded_hardened_v1.md \
  --top-k 5 \
  --eval-questions-file data/eval/rag_eval_set_v1_hardened.jsonl \
  --models qwen3:0.6b qwen3:4b-instruct mistral:7b
```

## Limits

- No training, LoRA, QLoRA, discovery, corpus scaling, parser changes, or
  `chunks_v1` changes were performed.
- Existing P2T4 and P2T5 output directories were not regenerated or
  overwritten.
- Hardened wording and deterministic retrieval checks do not prove answer
  correctness.
- Manual grounding review remains required before any scientific reliability
  claim.

## Next Step

Run QA grounded v1 into a new hardened output directory, then use the P2T5
manual review workflow to evaluate groundedness, answer correctness, citation
support, and abstention behavior by human review.
