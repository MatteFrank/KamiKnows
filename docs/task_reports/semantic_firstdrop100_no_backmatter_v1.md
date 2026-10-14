# Semantic first-drop chunking, filtered corpus, MiniLM retrieval

## Objective and inputs

Explore earlier semantic cuts using the same five parsed papers, excluding
acknowledgments/bibliography, with 100–400 whitespace words and MiniLM retrieval.
Questions remain `data/eval/rag_eval_set_v2.jsonl` (13, k=5).
The user explicitly chose to keep short sections separate as flagged exceptions.
No parser changes, training, corpus expansion or previous-output overwrites.

## File-by-file changes

1. `kamiknows/fulltext/semantic_chunking.py`: add opt-in `first_drop` policy.
   A boundary qualifies when cosine is below the per-section 25th percentile AND
   decreases by at least 0.10 from the previous boundary. At the first boundary,
   use the section mean as reference. Cut at the earliest qualifying boundary
   after 100 words, rather than the lowest cosine within the size window. Unlike
   the old policy, semantic cuts are considered even when the rest fits in 400.
   Reject cuts leaving fewer than 100 new words in a terminal tail. If the maximum
   forces a cut, balance the tail when whole-sentence boundaries permit. Keep
   sections separate and complete; retain oversize single sentences as exceptions.
   Record score decreases, selection policy and decision reasons in diagnostics.
2. `kamiknows/rag_ready/semantic_dataset.py`: pass new parameters to the chunker;
   record settings and length exceptions; flag chunks below/above size bounds.
   Regenerate PNG/PDF plots with cosine decreases, minimum drop and selected cuts.
3. `scripts/build_semantic_dataset_v1.py`: expose `--cut-policy first_drop` and
   `--min-score-drop 0.10`; use existing `--min-words 100 --target-words 400`
   and `--exclude-backmatter`. Legacy defaults remain available.
4. `tests/test_semantic_chunking.py`: verify earliest qualifying cut, cutting
   inside a section fitting the maximum, rejection of small decreases, tail
   balance and complete coverage.
5. Existing `fulltext/section_filter.py` and `rag_ready/sentence_dataset.py`:
   reused for filtering and source/coverage audit; no new parser/filter policy.
6. Existing `scripts/compare_retrievers_v1.py`: reused unchanged to compare real
   locally cached MiniLM against TF-IDF on the new dataset. MiniLM uses exact
   token windows for long passages and normalized length-weighted pooling.
7. New output-local `run.slurm`: run Qwen with saved MiniLM contexts, existing
   questions and parameters; source hashes validated by `rag/saved_retrieval.py`.

## Commands

```bash
python scripts/build_semantic_dataset_v1.py \
  --reference-dir outputs/rag_ready_fastcalo_v1 \
  --sentence-comparison-dir outputs/rag_ready_fastcalo_sentence_no_backmatter_v1 \
  --output-dir outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1 \
  --min-words 100 --target-words 400 --overlap-words 60 \
  --window-sentences 2 --threshold-quantile 0.25 \
  --cut-policy first_drop --min-score-drop 0.10 --exclude-backmatter

python scripts/compare_retrievers_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1 \
  --eval-questions-file data/eval/rag_eval_set_v2.jsonl \
  --output-dir outputs/retrieval_minilm_semantic_firstdrop100_no_backmatter_eval_v2_top5 \
  --top-k 5

sbatch outputs/qa_grounded_qwen06b_semantic_firstdrop100_no_backmatter_minilm_eval_v2_top5/run.slurm
```

## Outputs, metrics and validation

- Dataset: 169 chunks, mean 211.64 words, maximum 397; source sentence coverage PASS.
- Length exceptions: three entire short sections (79, 66 and 16 words); no chunks above 400.
- Cuts: 93 semantic drops, 11 length limits, 65 section ends.
- Plots: 132 nonempty PNG/PDF files; plot status PASS. Overview and Method section inspected visually.
- Tests: 18 targeted passed; full suite 244 passed, 2 skipped.
- Only warning: existing embedding dimension method deprecation; inference completed.

| Retrieval | Source hit@5 | MRR | Multi-source complete |
|---|---:|---:|---:|
| Previous MiniLM, unfiltered old semantic policy | 9/11 | 0.4530 | 2/3 |
| New MiniLM, filtered first-drop 100–400 | 9/11 | 0.5212 | 2/3 |
| New TF-IDF on same new chunks | 8/11 | 0.5030 | 1/3 |

MiniLM misses remain q006 and q010. Question-level before/after ranks and sections
are saved in `before_after_minilm_retrieval_v1.json` in the retrieval directory.
Qwen job 12230 submitted with saved MiniLM contexts; generation parameters remain
qwen3:0.6b, temperature=0, top_p=0.9, num_predict=512, timeout=120. Dedicated port
11444. QA completed on 2026-10-14. Output:
`outputs/qa_grounded_qwen06b_semantic_firstdrop100_no_backmatter_minilm_eval_v2_top5`.
The score improvement concerns source ordering, not evidence support or answer accuracy.
Original source-section hashes and removed-text audit are saved with the dataset.
CLI help and `git diff --check` passed.

## Limits and next step

The threshold 0.10 is an explicit initial experimental choice, not optimized
against answer labels. Low cosine and a steep decrease are distinct conditions;
this policy requires both. Similarity measures representation changes, not factual
support. Short sections and indivisible long sentences are retained and flagged;
100–400 words is not claimed as a universal hard bound on every source section.

Both backmatter filtering and chunk policy/minimum change versus the previous
unfiltered semantic run. A before/after comparison measures their combined effect,
not an isolated causal contribution. A separate filtered run with the old policy
would be necessary to isolate the chunking change.

Keep the same MiniLM query expansion and token-window policies for comparability.
Inspect plots, retrieved passages and answer citation support; source-hit metrics
and automatic answer statuses do not prove scientific correctness. Human review
of the generated answers remains pending.


## Completed QA and review artifacts

- 13 answer records; 0 backend errors.
- New normalized statuses: 12 answered, 1 insufficient; previous: 11 and 2.
- Invalid citations: 3 vs 0 previously; invented labels: 0 in both.
- JSON parse warnings: 5 vs 6 previously.
- Non-JSON/truncated outputs remain preserved. Automatic answered counts are not
  scientific correctness scores.
- `before_after_review_v1.csv` (13 unique pending review rows) and
  `before_after_review_v1.md` were generated in the new QA directory. CSV retains
  both raw responses and full before/after retrieved contexts; Markdown includes
  answers, automatic diagnostics and full new evidence.

Preliminary inspection, not human-validated labels: q001/q002 still attribute
WGAN to the CaloGAN questions; q012 still confuses GANs and normalizing flows.
q005 and q011 have severely incomplete output. q008 now proposes a Geant4
replacement configuration/procedure instead of abstaining, despite this being
an insufficient-evidence question. q010 changes its named loss component, but
its required source is still missing from retrieval; do not accept this as a
scientifically verified correction. The pipeline's no-error completion and
improved MRR do not establish improved answer quality.

Recommended next step: review the retrieved evidence independently of Qwen;
then isolate generation/format problems using the same saved contexts. To
attribute gains to first-drop chunking alone, add a filtered old-policy control.
