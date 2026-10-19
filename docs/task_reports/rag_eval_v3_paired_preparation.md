# Paired evaluation v3: twenty questions on the same five papers

## Objective and inputs

Prepare 20 English questions as 10 matched pairs, with and without explicit
arXiv IDs, keeping the five-paper corpus and existing v2 questions unchanged.
Inputs: v2 question wording and locally parsed sections under
`outputs/fulltext_fastcalo_5_v0/processed/papers/` for all five papers.
No original PDF rereading, network acquisition, corpus change or model run.

## Outputs

- `data/eval/rag_eval_set_v3_paired.jsonl`: operational question set.
- `data/eval/rag_eval_set_v3_paired.md`: readable questions, draft reference
  answers, acceptance points and exact evidence excerpts.
- `data/eval/rag_eval_set_v3_reference_answers_draft.jsonl`: reference material
  stored separately, keyed by question ID, with source paths, section IDs, exact
  character offsets and file hashes. Human validation pending.
- `data/eval/rag_eval_set_v3_reference_review.csv`: ten pair-level review rows.
- `data/eval/rag_eval_set_v3_manifest.json`: counts and artifact/source hashes.

## Design

Pairs cover CaloGAN identity, layer representation, U-net architecture for jet
images, WGAN speed and energy scaling, CaloFlow two-step factorization, GAN/flow
likelihood comparison, jet/shower objects, energy-loss constraints across the two
CaloGAN papers, and two operational deployment questions without adequate evidence.

There are 10 explicit and 10 implicit variants; 16 answerable and 4 insufficient-
evidence records. Three answerable pairs require both papers. Negative explicit
variants name five and two papers respectively; this allows testing abstention
even when all requested sources are available. k=5 covers the largest request.

Within each pair only the arXiv prefix changes (plus initial capitalization).
Implicit variants retain model/task descriptors to identify the same information
need without giving an arXiv ID or paper title. Named models do not activate the
source constraint extractor. Section keyword hints are empty in both variants;
existing type-based query expansion remains unchanged.

## Commands and validation

Local Python artifact assembly, followed by checks using the existing
`load_eval_questions_from_file`, `extract_requested_arxiv_ids` and
`plan_source_constraints` functions. All checks passed: schema, unique IDs,
20/10/10 counts, matched bodies, source availability at k=5, behavior counts,
answer/reference separation, exact quote offsets and source hashes.
`git diff --check` passed. No software implementation changed; no full suite
rerun needed for these dataset-only artifacts. No benchmark metrics yet.

For the next retrieval run, use a fresh output directory:

```bash
python scripts/compare_retrievers_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1 \
  --eval-questions-file data/eval/rag_eval_set_v3_paired.jsonl \
  --output-dir outputs/retrieval_explicit_sources_semantic_eval_v3_paired_top5 \
  --top-k 5 --source-policy explicit_arxiv
```

Saved v2 retrieval files cannot be reused with the new twenty-question file:
question IDs, content and hashes differ. QA needs the new retrieval file and
the same v3 question file. Existing defaults and Slurm scripts are unchanged.

## Warnings, limits and next step

These are draft references authored from extracted source text, not expert-
validated gold labels. Parser artifacts remain in exact quotes, including
missing mathematical symbols and 'energy conversation' in the CaloFlow text.
No unseen mathematical detail should be inferred from those artifacts.

Canonical expected papers do not exhaust all valid sources for implicit
questions: for example both CaloGAN papers can support some identity or geometry
answers. Review alternate support before treating a source-metric miss as a wrong
answer. Scope-limitation excerpts do not by themselves prove absence of a date
or deployment prescription; the negative labels require human corpus review.

Source hit@k is favored by explicit filtering and cannot be treated as an unbiased
retrieval comparison across variants. Inspect paired differences in actual answer
evidence, attribution, completeness and abstention. Answer text and reference
evidence are not passed to generation. Next: review draft answer/source labels,
then run retrieval and QA, grouping analysis by pair and explicit/implicit variant.
