# Explicit source constraints: implementation

## Objective and inputs

Connect source constraints extracted from question text to TF-IDF, MiniLM,
direct QA and saved-context QA. Preserve the current dataset, 13 questions and
previous outputs. This task implements and tests the code; it does not run a
new real-model benchmark or submit a GPU job.

## Changes

- `rag/source_constraints.py`: shared planning and diagnostics. Missing requested
  sources or too-small top-k produce an explicit status and empty context.
- `rag/retriever_v1.py`: optional candidate indices; global TF-IDF statistics
  remain fixed. Invalid/repeated indices are rejected.
- `scripts/compare_retrievers_v1.py`: `--source-policy none|explicit_arxiv`.
  Filter BEFORE ranking; slice chunk/vector arrays with identical indices. Rank
  all eligible candidates for constrained questions, then reserve the best chunk
  per requested source and fill remaining places. Reassign ranks and labels.
- `rag/qa_grounded_v1.py`: same rules for direct TF-IDF QA, source attribution
  prompt and traceability in contexts, run config and manifest.
- `scripts/run_qa_grounded_v1.py`: CLI/config source policy. When loading saved
  retrieval and no policy override is given, inherit the saved manifest policy.
- `rag/saved_retrieval.py`: validate policy/version, recompute diagnostics from
  question and original dataset, accept fewer-than-k or empty valid contexts,
  reject disallowed sources or incomplete source coverage. Legacy manifests
  without policy continue to mean unrestricted retrieval.
- `scripts/run_qa_explicit_sources_v1.slurm`: prepared launch script with new
  retrieval/output paths and dedicated Ollama port 11445; refuses existing QA
  output directory. Slurm log is written directly under outputs.
  Updated to use the native shared executable
  `/storage-hpc/kamiknows/software/ollama/bin/ollama` and models under
  `/storage-hpc/kamiknows/models/ollama`, without Apptainer. Paths were recovered
  from the user's shell history and verified on disk. The root can be overridden
  with `KK_ROOT`; executable/model directory checks run before creating outputs.
- Tests: extraction, versions, low-ranked requested sources, vector alignment,
  balancing, global IDF invariance, missing sources, limited budgets, saved-context
  validation and tampered diagnostics.

## Commands run and validation

- Full suite: 256 passed, 2 skipped.
- Both CLI help commands passed; Slurm script `bash -n` passed.
- `git diff --check` passed.
- No benchmark or generation metrics yet; no model/GPU job launched.

## Commands for the next experiment (not executed)

Run from repository root in the existing kamiknows environment:

```bash
python scripts/compare_retrievers_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1 \
  --eval-questions-file data/eval/rag_eval_set_v2.jsonl \
  --output-dir outputs/retrieval_explicit_sources_semantic_eval_v2_top5 \
  --top-k 5 --source-policy explicit_arxiv

# Inspect contexts/diagnostics, then launch QA using the saved MiniLM results.
sbatch scripts/run_qa_explicit_sources_v1.slurm
```

Direct TF-IDF QA can also use `--source-policy explicit_arxiv` without a saved
retrieval file. For saved results an explicit mismatching policy is rejected.

## Limits and next step

IDs are recognized syntactically in question text; named models alone do not
become hidden source constraints. Versioned IDs require that exact version;
unversioned IDs accept available versions. No expected source IDs enter selection.
Existing question-type/section-keyword query expansion remains unchanged.

For unrestricted questions all papers remain eligible, but top-k may still come
from one paper. No automatic diversity rule is added to unrestricted retrieval.
For constrained questions, source hit@k may become guaranteed by selection when
the request and eval labels agree: it is not comparable as a pure semantic
retrieval-quality claim. Evidence sufficiency and factual support remain manual.
Missing evidence is exposed in the prompt; the LLM is still capable of ignoring
an abstention instruction. No scientific accuracy claim follows from test passes.

Next: run the prepared retrieval comparison, inspect passages, then QA. Add new
evaluation questions only after verifying single-source, multi-source and
unrestricted behavior on the existing set.
