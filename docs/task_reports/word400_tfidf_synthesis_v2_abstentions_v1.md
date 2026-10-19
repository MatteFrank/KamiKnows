# Original word chunking / TF-IDF on four abstention cases

## Objective and inputs

Repeat the same four questions as the synthesis_v2 prompt experiment, replacing
semantic first-drop chunking / MiniLM with the original word-window dataset and
TF-IDF retrieval. Keep qwen3:4b-instruct, synthesis_v2, temperature=0, top_p=0.9,
num_predict=512, request timeout=300 seconds, top-k=5 and explicit-source policy.

- Questions: pair_01_implicit, pair_06_explicit, pair_07_explicit, pair_07_implicit
  from `data/eval/rag_eval_set_v3_paired.jsonl`.
- Word dataset: `outputs/rag_ready_fastcalo_v1_enriched/chunks_v1.jsonl` (108).
- Baseline: `outputs/qa_qwen4b_synthesis_v2_abstentions_v1` (semantic/MiniLM).
- New outputs: `outputs/qa_qwen4b_word400_tfidf_synthesis_v2_abstentions_v1`.

## Chunking and comparison limits

The original target is 400 whitespace words, overlap 60, but short terminal tails
can be merged into a window; actual maximum length is 471 words. No new chunker
or parser changes are made. Original acknowledgments/bibliography are retained,
unlike the filtered semantic baseline. This experiment changes chunking, retrieval
and backmatter handling jointly; it does not isolate each causal contribution.

The explicit jet/WGAN question retrieves an Acknowledgments chunk at rank 1.
The implicit CaloGAN question now retrieves its Method section at rank 1. The
likelihood comparison retrieves both CaloGAN Method and CaloFlow Conclusions.

## Commands, artifacts and validation

Prepared contexts locally using `retrieve_contexts_v1` with the existing global
TF-IDF index and source constraints, then submitted:

```bash
sbatch outputs/qa_qwen4b_word400_tfidf_synthesis_v2_abstentions_v1/run.slurm
# Job 12334
```

Native shared Ollama, one GPU, 16 GB host memory, dedicated port 11452.
Artifacts include question subset, exact contexts/prompts, copied semantic/MiniLM
answers, input/prompt hashes, generator settings, runner, Slurm script and logs.
Four contexts and five passages each verified; runner compiles, bash syntax and
git diff --check pass. No application-code edits or full test rerun needed for
this artifact-only experiment. Reference answers are not supplied to generation.

## Results and next step

Job 12334 completed: four records, zero backend errors. Four unique IDs and
identical question text/profile across the two runs were verified. Compare content, completeness, source
attribution, raw JSON and citation support against the semantic/MiniLM results.
An answered status can result from malformed-JSON fallback; it is not an accuracy
score. All scientific judgments remain pending human review. Earlier outputs and
the default prompt remain unchanged.


## Completed results

The word/TF-IDF run reports three answered and one insufficient_evidence, versus
two and two for semantic/MiniLM. One answered record in each run is an incomplete
JSON abstention (pair_07_explicit) normalized by the existing fallback; it must
not be counted as a quality gain. Both runs have one JSON parse warning.

The substantive change is pair_01_implicit: word/TF-IDF answers CaloGAN/GAN,
where semantic/MiniLM abstained. Both answer the central likelihood comparison,
but additional statements still need evidence checks. Both fail to synthesize
the jet/shower comparison. Word/TF-IDF therefore helps one observed case; this
selected sample does not establish general superiority or isolate the effects
of chunking, retrieval and backmatter.

Review artifacts: comparison_v1.md, comparison_v1.csv and
comparison_summary_v1.json in the new output directory. All four reviewer rows
remain pending; earlier review files were not edited. No automatic scientific
accuracy score is assigned. Next: inspect claim-level support and use matched
filtering/chunking controls before attributing improvements to TF-IDF alone.
