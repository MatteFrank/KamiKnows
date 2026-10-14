# Preparation of the v2 question-level review

## Objective

Prepare a human-editable comparison of all 13 questions across four QA runs:
word/TF-IDF, sentence/TF-IDF, semantic/TF-IDF and semantic/MiniLM.

## Inputs

Canonical `data/eval/rag_eval_set_v2.jsonl` and saved answers/contexts in:
- `outputs/qa_grounded_qwen06b_eval_v2_top5_20261012`
- `outputs/qa_grounded_qwen06b_sentence_eval_v2_top5`
- `outputs/qa_grounded_qwen06b_semantic_eval_v2_top5`
- `outputs/qa_grounded_qwen06b_semantic_minilm_eval_v2_top5_retry1`

## Outputs

- `outputs/review_chunking_retrieval_eval_v2/review_13_questions_v1.md`:
  13-row overview, question text, original criteria, draft expected content,
  per-run diagnostics, full saved answer text and rank-to-source mapping.
- `outputs/review_chunking_retrieval_eval_v2/review_52_answers_v1.csv`:
  one row per question/configuration with original response, raw response,
  automatic diagnostics, provisional assistant assessment and blank human fields.

## Commands and validation

Read-only inspection of saved JSONL answers and contexts; local Python artifact
assembly. Checked 13 distinct question IDs and 52 pending review rows. No pipeline,
model or benchmark was run. Existing benchmark, outputs and grounding fields were
not changed. No software tests are needed for these review-only artifacts.

## Metrics, warnings and limits

Source hits, rank, parser warnings and citation-label status are copied from the
saved runs. Scientific answer assessments and expected-content summaries are
provisional proposals, not verified gold labels. The sentence q008 raw response
attempts abstention despite an answered parser status. Answer correctness,
evidence sufficiency and citation support remain separate human judgments.
No aggregate scientific correctness score is calculated before human validation.

## Next step

The user fills the reviewer fields and corrects provisional judgments. Use the
validated review to choose the next controlled intervention.
