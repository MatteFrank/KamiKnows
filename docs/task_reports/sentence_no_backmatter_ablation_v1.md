# Sentence chunking without acknowledgments/bibliography: controlled ablation

## Objective

Test whether acknowledgments and bibliography distract retrieval and Qwen, using
the existing five-paper corpus and canonical v2 questions. The filter is optional
and defaults to disabled. Original datasets, parser and runs are preserved.

## Inputs and implementation

- Reference selection: `outputs/rag_ready_fastcalo_v1`, existing parsed sections.
- Baseline comparison: `outputs/rag_ready_fastcalo_sentence_v1` and
  `outputs/retrieval_sentence_eval_v2_top5`.
- Questions: `data/eval/rag_eval_set_v2.jsonl` (13).
- New `fulltext/section_filter.py`: exclude sections whose normalized heading is
  acknowledgments/acknowledgements, bibliography, references or equivalent.
- Bibliography embedded in WGAN's Acknowledgments section is removed with that
  section. Known trailing unexpanded markers `refs` and `biblio spphys` are removed.
- Keep scientific sections, appendices and prose mentioning references. Do not
  guess bibliography boundaries from ordinary prose. No parser changes.
- Shared sentence/semantic builder supports `--exclude-backmatter`; sentence
  coverage is checked on retained sections. The manifest and full removed-text
  audit document the intervention. Source section hashes are preserved.
- Equation artifacts remain copied from the reference corpus for provenance;
  the experiment's retrieval input is the filtered chunk file.

## Commands run

```bash
python scripts/build_sentence_dataset_v1.py \
  --reference-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_ready_fastcalo_sentence_no_backmatter_v1 \
  --target-words 400 --overlap-words 60 --exclude-backmatter

python scripts/run_light_reranking_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_sentence_no_backmatter_v1 \
  --eval-questions-file data/eval/rag_eval_set_v2.jsonl \
  --output-dir outputs/retrieval_sentence_no_backmatter_eval_v2_top5 \
  --report-path docs/task_reports/retrieval_sentence_no_backmatter_eval_v2_top5.md \
  --candidate-top-n 20 --final-top-k 5

sbatch outputs/qa_grounded_qwen06b_sentence_no_backmatter_eval_v2_top5/run.slurm
```

The retrieval runner also produces reranked results, but the controlled chunking
comparison and QA use baseline TF-IDF, without reranking.

## Dataset and retrieval results

- 5 papers retained; 114 baseline chunks -> 110 filtered chunks.
- Removed 3 acknowledgments sections and 1 trailing bibliography marker: 730
  whitespace words. Full audit: `section_exclusions_v1.jsonl` in the new dataset.
- Coverage PASS; 0 oversize sentences.

| Baseline TF-IDF metric | Before | Filtered |
|---|---:|---:|
| Expected-source hit@5 | 8/11 | 9/11 |
| Hit@1 | 3/11 | 3/11 |
| MRR | 0.4424 | 0.4409 |
| Complete multi-source coverage | 2/3 | 2/3 |

Filtered misses: `eval_v0_q002`, `eval_v1_q012`; q003 changes from miss to hit.
Question-level ranks and retrieved section labels are saved in
`outputs/retrieval_sentence_no_backmatter_eval_v2_top5/backmatter_ablation_comparison_v1.json`.

## QA and validation

Qwen job 12207 uses the same 13 questions, k=5, qwen3:0.6b, temperature=0,
top_p=0.9, num_predict=512 and request timeout=120 seconds. Dedicated server
port 11443. Outputs are in
`outputs/qa_grounded_qwen06b_sentence_no_backmatter_eval_v2_top5`.

- Targeted tests: 16 passed.
- Full suite: 242 passed, 2 skipped.
- QA job 12207 completed: 13 records, 0 backend errors.
- 12 normalized `answered`, 1 `insufficient_evidence`, 1 invalid-citation record,
  0 invented citation labels. These status counts match the original sentence run.
- JSON parse warnings decreased from 7 to 3. The malformed q008 raw response
  attempts abstention but is normalized to `answered`; do not treat the automatic
  status totals as correct-answer or correct-abstention counts.
- Before/after review artifacts (13 rows, all pending):
  `before_after_review_v1.csv` and `before_after_review_v1.md` in the new QA folder.
  The CSV retains original raw responses and complete retrieved contexts; the
  Markdown shows both answers and filtered evidence passages.
- `git diff --check`: passed.
- Initial sandbox sbatch failed because Slurm user/configuration was unavailable;
  submission outside sandbox succeeded using the existing approved sbatch rule.

## Limits and next step

This is the sentence-chunking ablation only; semantic support is implemented but
no filtered semantic dataset/benchmark has been executed. Corpus, questions and
generator settings are held fixed. TF-IDF IDF weights necessarily change when
documents are removed. Chunk IDs can shift after excluded sections; compare
paper, section and text rather than assuming old IDs denote the same passage.

Unlabelled bibliography embedded inside scientific sections is not automatically
identified. The inspected current corpus's long explicit bibliography was inside
the excluded WGAN acknowledgment section. No claim that bibliography explains
most answer errors is established. Source hits are not evidence-support labels.

Compare the new Qwen answers against the original sentence run, checking factual
correctness, citation support, evidence sufficiency and abstention manually.

## Preliminary answer inspection (not human-validated labels)

- q001 changes from `CaloFlow` to `CaloGAN`, correcting the named model in the
  draft expected-content comparison.
- q002 changes from WGAN to GAN, but the requested source is still not retrieved,
  and layer representation remains vague.
- q012 still misses complete expected-source coverage and incorrectly describes
  both CaloGAN and CaloFlow as normalizing flows.
- Removing distractor sections helps some cases but does not establish that most
  errors came from bibliography or acknowledgments. No aggregate scientific
  accuracy score is assigned before human review.
