# TF-IDF vs MiniLM retrieval on semantic chunks, eval v2

## Objective and inputs

Compare retrievers on the same 118 semantic chunks, 13 canonical v2 questions and
k=5. Two insufficient-evidence questions are excluded from source-hit metrics.
Source inputs and hashes are in the run manifest. No reranking or Qwen is used.

## Implementation and command run

New `scripts/compare_retrievers_v1.py` reuses the existing TF-IDF implementation,
real local EmbeddingEncoder, dense ranking, token-window encoding and existing
source evaluation and metric aggregation. New test covers shared aggregation.
No existing retriever or metric definition was modified.

```bash
python scripts/compare_retrievers_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_semantic_v1 \
  --eval-questions-file data/eval/rag_eval_set_v2.jsonl \
  --output-dir outputs/retrieval_tfidf_minilm_semantic_eval_v2_top5 \
  --top-k 5
```

Executed using the kamiknows Python environment, local model cache and CPU.
Existing output directories are refused. No GPU allocation or network downloads.

## Experimental settings

TF-IDF uses its original weighted title/section/text and expanded query.
MiniLM passages contain title, section and text. Dense queries retain natural
question wording and append the same question-type expansion terms and expected
section keywords used by TF-IDF. These are different native representations;
this is not a raw-question-only experiment. Expected source IDs are used only
for evaluation, not passage filtering or ranking.

106 of 118 passages exceed the 256-token limit and use exact, nonoverlapping
content-token windows, reserving special-token space. Normalized window vectors
are averaged with weights equal to content length, then normalized. No queries
needed windows. This is a documented window-pooling variant of MiniLM retrieval,
not the previous truncating embedding benchmark. Full passages are not discarded.
Encoding counts/window lengths are recorded in the output diagnostics.

## Results

| Metric | TF-IDF | MiniLM |
|---|---:|---:|
| Source hit@5 | 8/11 (0.7273) | 9/11 (0.8182) |
| MRR | 0.4576 | 0.4530 |
| Mean hit rank, hits only | 2.125 | 2.667 |
| Complete multi-source coverage | 2/3 | 2/3 |

MRR and source hits use the existing evaluator: multi-source questions require all
sources; hit rank is the maximum matching rank. Source hits do not prove relevant
answer evidence was retrieved.

## Question-level hit ranks

| eval_v0_q001 | 2 | 5 |
| eval_v0_q002 | miss / no expected source | 2 |
| eval_v0_q003 | miss / no expected source | 3 |
| eval_v0_q004 | 1 | 1 |
| eval_v0_q005 | 5 | 5 |
| eval_v0_q006 | 2 | miss / no expected source |
| eval_v0_q007 | 1 | 1 |
| eval_v0_q008 | miss / no expected source | miss / no expected source |
| eval_v1_q009 | 3 | 1 |
| eval_v1_q010 | 1 | miss / no expected source |
| eval_v1_q011 | 2 | 2 |
| eval_v1_q012 | miss / no expected source | 4 |
| eval_v1_q013 | miss / no expected source | miss / no expected source |

MiniLM recovers q002, q003 and q012 but loses q006 and q010. q009 improves to rank 1.
For q006, the jet-image paper is missing; q010 retrieves related energy-conditioning
passages from other papers, missing the specific requested first CaloGAN source.
Several inspected MiniLM results retrieve content sections instead of the
Acknowledgments bibliography, but no completed evidence-level annotation is claimed.

## Outputs and validation

The new output directory contains both retrieval JSONL files, encoding diagnostics,
comparison summary and manifest with settings and source hashes. TF-IDF chunk IDs
and ordering were checked against every prior semantic baseline record: identical.
All 13 questions were processed. Existing suite: 234 passed, 2 skipped; new metric
aggregation test: 1 passed. `git diff --check` passed. A FutureWarning about the
encoder dimension method was emitted; inference completed successfully.

## Limits and next step

The hit improvement is one question on a small pilot; MRR does not improve. Keep
window pooling and query expansion explicit in any reported result. No general
superiority or scientific answer correctness is established. Inspect evidence and
citation support before deciding whether to run Qwen with the MiniLM contexts.
