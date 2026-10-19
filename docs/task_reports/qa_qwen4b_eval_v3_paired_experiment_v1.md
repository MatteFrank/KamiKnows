# Qwen 4B / MiniLM on paired v3 questions

## Objective and inputs

Run the 20-question v3 set on the same five-paper corpus. Compare ten matched
explicit/implicit pairs using native shared Ollama qwen3:4b-instruct, real cached
MiniLM retrieval and existing citation validation. Reference answers remain draft,
human review pending; they are not supplied to retrieval/generation.

- Dataset: `outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1`.
- Questions: `data/eval/rag_eval_set_v3_paired.jsonl`.
- Reference drafts: `data/eval/rag_eval_set_v3_reference_answers_draft.jsonl`.
- Retrieval: `outputs/retrieval_explicit_sources_semantic_eval_v3_paired_top5`.
- QA: `outputs/qa_grounded_qwen4b_minilm_eval_v3_paired_top5`.

## Commands and settings

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/compare_retrievers_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1 \
  --eval-questions-file data/eval/rag_eval_set_v3_paired.jsonl \
  --output-dir outputs/retrieval_explicit_sources_semantic_eval_v3_paired_top5 \
  --top-k 5 --source-policy explicit_arxiv

sbatch scripts/run_qa_qwen4b_minilm_eval_v3_paired_v1.slurm
```

The interpreter is `/home/e/eapicell/.conda/envs/kamiknows/bin/python`. QA uses
temperature=0, top_p=0.9, num_predict=512, request timeout=300 seconds, top-k=5,
dedicated Ollama port 11450, one GPU and 16 GB requested host memory. Shared
executable/model paths are unchanged; no Apptainer or model downloads.
The comparison runner also saves TF-IDF results; generation uses only MiniLM.

## Outputs, validation and metrics

Job 12332 submitted. MiniLM retrieval completed: explicit answerable questions
8/8 canonical-source hits; implicit answerable questions 6/8. Missing complete
source coverage: pair_06_implicit and pair_08_implicit. TF-IDF results are saved
for comparison (15/16 overall vs MiniLM 14/16); QA uses MiniLM only.
Only retrieval warning: existing embedding dimension method deprecation.
Slurm script syntax checked with bash -n. No
application code changes; no full software test rerun needed. Original outputs,
datasets and eval v2 remain intact.

## Limits and next step

Reference labels and evidence require human review. Source hits in explicit
questions are favored by the filter; do not interpret their rate as unbiased
retrieval quality. Implicit canonical source labels may omit alternative valid
supporting papers. Status counts and structural citation validity do not measure
scientific accuracy. Compare evidence sufficiency, answer content and abstention
pair by pair, rather than declaring a winner from source-hit or answered counts.


## Completed QA, job 12332

- All 20 questions processed; zero backend errors.
- Normalized statuses: 12 answered, 8 insufficient_evidence.
- Three JSON parse warnings; one answered-without-valid-citation record;
  zero invented citation labels.
- Explicit: 6 answered / 4 insufficient; 1 JSON warning, 0 invalid citations.
- Implicit: 6 answered / 4 insufficient; 2 JSON warnings, 1 invalid citation.
- All four intended insufficient-evidence questions were marked insufficient.
  Four answerable questions were also marked insufficient: pair_06_explicit,
  pair_07_explicit, pair_01_implicit, pair_07_implicit.
- First pair: explicit answer names CaloGAN/GAN; implicit answer field is empty
  with insufficient_evidence status. Equal aggregate status counts therefore
  hide differences in which questions receive an answer.

Review outputs:
`outputs/review_qwen4b_eval_v3_paired_v1/review_10_pairs_v1.md`,
`review_20_answers_v1.csv`, `paired_status_summary_v1.json`,
`review_manifest_v1.json`.
Each pair includes the draft expected answer, exact reference excerpts, both
actual model outputs and complete retrieved contexts. Raw JSON is retained,
including malformed responses and empty answer fields. No scientific correctness
labels were automatically assigned. All 20 human review rows remain pending.
Validation: 20 unique question IDs, 10 pairs, 5 retrieved contexts per answer,
reference/input hashes recorded, shell syntax and git diff --check passed.
No software implementation was changed for this experiment.

Next: review reference drafts and assess answer completeness, attribution and
citation entailment, especially the architecture question and the comparison
questions, before interpreting explicit/implicit differences scientifically.
