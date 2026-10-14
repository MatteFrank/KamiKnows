# QA from saved retrieval v1

## Objective

Reuse the saved MiniLM ranking for grounded QA, without rerunning retrieval.
Default QA behavior remains TF-IDF. No generation job was launched for this task.

## Changes and inputs

- New `kamiknows/rag/saved_retrieval.py` validates comparison artifacts and
  hydrates full text from the selected dataset, preserving rank and score.
- `scripts/run_qa_grounded_v1.py` adds `--retrieval-results-file`.
- `kamiknows/rag/qa_grounded_v1.py` accepts the optional file, validates before
  output writes or Ollama access, and records actual retriever and provenance.
- `tests/test_saved_retrieval.py` checks hydration and rejection of hash, chunk ID,
  question-text and rank mismatches.

The comparison manifest must be beside the saved JSONL. Dataset and canonical
question hashes, question IDs/text, top-k, rank order, chunk uniqueness and source
IDs must match. Citation labels are rebuilt from rank; evaluation is recomputed
with existing QA metrics. Prompt and generation settings are unchanged.

## Proposed command (requires a running Ollama service; not executed)

```bash
python scripts/run_qa_grounded_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_semantic_v1 \
  --eval-questions-file data/eval/rag_eval_set_v2.jsonl \
  --retrieval-results-file outputs/retrieval_tfidf_minilm_semantic_eval_v2_top5/minilm_retrieval_v1.jsonl \
  --output-dir outputs/qa_grounded_qwen06b_semantic_minilm_eval_v2_top5 \
  --report-path docs/task_reports/qa_grounded_qwen06b_semantic_minilm_eval_v2_top5.md \
  --top-k 5 --models qwen3:0.6b --temperature 0 --top-p 0.9 --num-predict 512 \
  --timeout-seconds 120 --base-url http://127.0.0.1:11434
```

## Validation and commands run

`python -m pytest -q`: 240 passed, 2 skipped. `git diff --check` passed.
Read-only validation of actual saved MiniLM results successfully hydrated 13
questions and 65 complete chunks, without model calls or output artifacts.

## Outputs, metrics, warnings and limits

Only code, tests and this report were changed. No new QA metrics exist yet.
Config/manifest record the retrieval artifact hash and original comparison manifest,
including token-window and expanded-query policies. Summary and answer metadata
identify MiniLM, rather than TF-IDF. Existing scientific-review limits still apply:
valid citation labels do not prove factual support. QA output paths still require
care: use fresh paths because the existing QA writer can overwrite artifacts.

## Next step

Run the command inside the established Slurm/Ollama setup with fresh output paths,
then compare Qwen answers against the semantic TF-IDF QA run.
