# Explicit-source QA: TF-IDF controls

## Objective

Compare the original word chunker plus TF-IDF against the recent semantic
first-drop/MiniLM run with the same explicit-source policy. Also run TF-IDF on
the exact same semantic dataset to isolate the retriever change.

## Inputs and settings

- Original word chunks: `outputs/rag_ready_fastcalo_v1_enriched` (108 chunks;
  original backmatter retained).
- Semantic chunks: `outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1`
  (169 chunks; acknowledgments/bibliography filtered).
- Existing semantic retrieval comparison:
  `outputs/retrieval_explicit_sources_semantic_eval_v2_top5`.
- Previous QA reference: `outputs/qa_grounded_explicit_sources_minilm_eval_v2_top5`.
- Same `data/eval/rag_eval_set_v2.jsonl`, top-k=5, explicit_arxiv policy,
  native shared Ollama and qwen3:0.6b, temperature=0, top_p=0.9,
  num_predict=512 and request timeout=120 seconds.

## Commands and outputs

```bash
sbatch scripts/run_qa_explicit_sources_word_tfidf_v1.slurm
# Submitted job 12245; direct TF-IDF retrieval with source constraints.

sbatch scripts/run_qa_explicit_sources_semantic_tfidf_v1.slurm
# Submitted job 12246; saved constrained TF-IDF contexts on the semantic dataset.
```

Output directories:
- `outputs/qa_grounded_explicit_sources_word_tfidf_eval_v2_top5`
- `outputs/qa_grounded_explicit_sources_semantic_tfidf_eval_v2_top5`

Both scripts refuse existing output directories. Earlier runs remain intact.

## Validation, metrics and failures

Both Slurm scripts pass `bash -n`; no changes to the retrieval implementation.
Both jobs completed with 13 answer records each and zero backend errors.
Review artifacts validated: 39 rows, 13 unique questions, all human labels pending.

## Limits and next step

Word TF-IDF vs semantic MiniLM changes chunking, backmatter filtering and
retriever together. Semantic TF-IDF vs semantic MiniLM changes only the
retriever/context selection within the current settings. There is no controlled
native-Ollama word run without constraints in this task, so the source-policy
effect alone is not isolated from every historical run difference.

Because sources are explicitly constrained, source hit rate is largely guaranteed
when requested sources are available. It does not show whether selected text
supports the answer. Scientific correctness, citation support and abstention
must be reviewed independently of parsing/status counts. No aggregate scientific
accuracy score is assigned automatically.

Next: compare all three runs question by question with their actual contexts,
particularly q001, and distinguish generation errors from missing evidence.


## Completed automatic metrics

| Configuration | Answered | Insufficient | Invalid citations | JSON parse warnings |
|---|---:|---:|---:|---:|
| Parole + TF-IDF | 11 | 2 | 1 | 4 |
| Semantico + TF-IDF | 12 | 1 | 1 | 5 |
| Semantico + MiniLM | 12 | 1 | 0 | 6 |

Word TF-IDF and semantic MiniLM answer CaloGAN to q001; semantic TF-IDF answers
Generative Adversarial Network (GAN), avoiding the prior WGAN misattribution but
not naming CaloGAN. Thus MiniLM is not necessary for the named model answer
under explicit-source constraints in this pilot. This does not show
that MiniLM provides no benefit on other questions or isolate every causal factor.
Word TF-IDF reports both intended no-evidence questions as insufficient_evidence;
their scientific abstention quality still requires reading the answers.

Artifacts: `outputs/review_explicit_sources_retrievers_v1/review_13_questions_v1.md`,
`review_39_answers_v1.csv`, and `comparison_summaries_v1.json`. Full retrieved
contexts and raw responses are retained in the CSV. No automated accuracy score.
