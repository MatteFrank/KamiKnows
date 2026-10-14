# Sentence grouping v1: implementation and user-run experiment

## Objective

Prepare a sentence-group chunking experiment on the exact existing five-paper
selection. Dataset construction and benchmarks are intentionally left to the user.

## Inputs

- Reference dataset: `outputs/rag_ready_fastcalo_v1` (paper selection, domain,
  equations, legacy questions and paths to parsed paper directories).
- Text: each selected paper's existing `sections.json` in
  `outputs/fulltext_fastcalo_5_v0/processed/papers/`.
- Evaluation, selected explicitly at retrieval time: `data/eval/rag_eval_set_v2.jsonl`.

## Implementation

- `kamiknows/fulltext/sentence_chunking.py`: deterministic punctuation-based
  sentence splitting, common abbreviation protection and groups of consecutive
  sentences within a section. No external model or new dependency.
- `kamiknows/rag_ready/sentence_dataset.py`: reads the reference paper list,
  generates groups and reuses `build_chunks_v1` for existing enriched metadata.
- `scripts/build_sentence_dataset_v1.py`: CLI for construction only.
- `tests/test_sentence_chunking.py`: segmentation, coverage, overlap, section
  boundaries, long sentences, parameter validation and overwrite protection.

Target size is 400 whitespace words, matching the original window construction.
The next chunk retains the largest whole-sentence suffix fitting within 60 words;
the overlap can be smaller or zero. A single sentence above target stays intact
and is flagged `oversize_sentence`. Small final groups are retained; the old
80-word tail-merging rule is not applied. These size differences must be inspected.

New chunk IDs use `<paper_id>::sentence_v1::<index>`. Sentence spans are zero-based
and section-local, with an exclusive end. Source section files are hashed in the
manifest. Coverage is checked before output creation. Existing output directories
are refused. Bibliography and parser artifacts are retained, without reparsing.

## Outputs when the user runs construction

`outputs/rag_ready_fastcalo_sentence_v1/` will contain `chunks_v1.jsonl`,
`rag_manifest_v1.json`, `chunks.jsonl`, `papers.jsonl`, `equations.jsonl` and
`eval_questions_v0.jsonl`. The v1 files are compatible with QA and reranking
loaders. No v0 manifest is generated, so this is not an input for the legacy
retrieval-baseline loader. Legacy questions are copied only for compatibility;
the v2 file must be specified explicitly for the experiment.

## User commands

From the repository root, in the existing Conda environment:

```bash
conda activate kamiknows
python scripts/build_sentence_dataset_v1.py \
  --reference-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_ready_fastcalo_sentence_v1 \
  --target-words 400 \
  --overlap-words 60
```

Inspect the printed counts and manifest length statistics, then inspect several
chunk boundaries, especially formulas, abbreviations and bibliography. No GPU,
Ollama, retrieval or question generation is invoked by this command.

For retrieval, reuse the existing evaluator after construction and inspection:

```bash
python scripts/run_light_reranking_v1.py \
  --config configs/phase2_p2t6_light_reranking_v1.json \
  --rag-ready-dir outputs/rag_ready_fastcalo_sentence_v1 \
  --eval-questions-file data/eval/rag_eval_set_v2.jsonl \
  --output-dir outputs/retrieval_sentence_eval_v2_top5 \
  --report-path docs/task_reports/retrieval_sentence_eval_v2_top5.md \
  --candidate-top-n 20 \
  --final-top-k 5
```

This CPU command produces both baseline TF-IDF and lexical reranking outputs.
Use only `baseline_*` results for the chunking comparison. It does not run Qwen.
It has no overwrite protection: use fresh output and report paths on every run.
No new retrieval implementation or metric definitions were introduced.

## Commands run and validation

Only software tests were executed, including construction on temporary synthetic
fixtures. No real dataset was built and no benchmark or GPU job was launched.
Validation results are recorded below after the test run.

- Targeted tests: 13 passed.
- Final full suite: `python -m pytest -q` — 226 passed, 2 skipped.
- CLI `--help` completed successfully; `git diff --check` passed.
- Tests ran with `/home/e/eapicell/.conda/envs/kamiknows/bin/python`.
- No benchmark metrics are available for sentence chunks yet.

## Limits and non-claims

- Heuristic punctuation splitting is not a guarantee of linguistic sentence
  correctness, particularly in noisy scientific LaTeX extraction.
- Partial CaloGAN parsing and bibliographic retrieval distractors remain.
- Chunk counts and sizes are unknown until the user builds the real dataset.
- No claim of better retrieval or scientifically correct answers is made.
- Keep benchmark, retriever, parser and generation settings fixed. Inspect
  relevant evidence as well as expected-source metrics in the comparison.

## Next step

The user builds the sentence dataset, checks its manifest and sample boundaries,
then runs retrieval. Compare against the saved word-based v2 contexts before
running Qwen again.
