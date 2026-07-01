# Phase 2 Task 02 - Embedding Model Selection v1

Date: 2026-06-30

## Objective

Compare configurable open/local embedding candidates on the same fixed FastCaloSimulation corpus and the same validated eval set, then choose a provisional embedding retrieval baseline for KnowKami / M-Fond LLM.

The selected embedding model is a provisional retrieval baseline, not a final scientific retrieval solution.

This task did not scale the corpus, train or fine-tune a model, add reranking, modify the full-text parser, introduce discovery generation, or generate free-form citations.

## Dataset Used

Input dataset:

```text
outputs/rag_ready_fastcalo_v0/
  chunks.jsonl
  papers.jsonl
  equations.jsonl
  eval_questions_v0.jsonl
  rag_manifest_v0.json
```

Counts:

```text
papers: 5
chunks: 108
equations: 28
questions: 8
```

Output directory:

```text
outputs/rag_v1_fastcalosim_embedding_selection/
```

## Models Tested

The registry was written to:

```text
outputs/rag_v1_fastcalosim_embedding_selection/embedding_model_registry_v1.json
```

Configured candidates:

| Model | Provider/library | Prefixes | Normalize | Max length | Configured dim |
| --- | --- | --- | --- | ---: | ---: |
| `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers` | none | yes | 256 | 384 |
| `intfloat/e5-small-v2` | `sentence-transformers` | `query: ` / `passage: ` | yes | 512 | 384 |
| `BAAI/bge-m3` | `sentence-transformers` | none | yes | 8192 | 1024 |

Environment note: `sentence_transformers` was not installed in this environment. To keep tests and the run offline, the benchmark used the deterministic hash embedding fallback for all three configured candidates. This preserves comparable pipeline behavior, metric calculation, prefix handling, caching, and citation validation, but it is not a real model-quality comparison.

## Command

```bash
python3 scripts/run_embedding_model_selection_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_v1_fastcalosim_embedding_selection \
  --top-k 5
```

## Output Files

Created:

```text
outputs/rag_v1_fastcalosim_embedding_selection/embedding_model_registry_v1.json
outputs/rag_v1_fastcalosim_embedding_selection/embedding_benchmark_results_v1.jsonl
outputs/rag_v1_fastcalosim_embedding_selection/embedding_eval_summary_v1.json
outputs/rag_v1_fastcalosim_embedding_selection/citation_validation_summary_v1.json
outputs/rag_v1_fastcalosim_embedding_selection/embedding_model_comparison_v1.md
```

Local embedding caches were also written under:

```text
outputs/rag_v1_fastcalosim_embedding_selection/embedding_cache/
```

## Comparative Metrics

All metrics use top_k=5 and expected arXiv/source matching from the existing eval set.

The mean reciprocal rank (MRR) is a statistic measure for evaluating any process that produces a list of possible responses to a sample of queries, ordered by probability of correctness. The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer: 1 for first place, 1⁄2 for second place, 1⁄3 for third place and so on.

| Model | Backend | Hit@5 | Retrieval miss | Mean expected rank | MRR | Avg query sec | Index build sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sentence-transformers/all-MiniLM-L6-v2` | `hash` | 0.625 | 3 | 1.600 | 0.479 | 0.001276 | 0.014705 |
| `intfloat/e5-small-v2` | `hash` | 0.500 | 4 | 1.500 | 0.417 | 0.001213 | 0.014124 |
| `BAAI/bge-m3` | `hash` | 0.250 | 6 | 1.000 | 0.250 | 0.003011 | 0.016488 |

## Comparison With TF-IDF Baseline

Available TF-IDF comparison:

```text
source: outputs/rag_v1_fastcalosim_retrieval/retriever_eval_summary_v1.json
retriever: tfidf_local_v1
top_k: 5
expected-source hit rate: 0.750
```

Best embedding-selection result in this run:

```text
model: sentence-transformers/all-MiniLM-L6-v2
backend: hash fallback
top_k: 5
expected-source hit rate: 0.625
delta vs tfidf_local_v1 top_k=5: -0.125
```

Because this run used hash fallback embeddings, the comparison should be read as a pipeline validation and provisional placeholder, not as evidence that MiniLM is better or worse than real E5/BGE embeddings.

## Selected Provisional Model

Selected provisional embedding baseline:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Reason:

- highest observed hit@5 among the three configured candidates in this offline run;
- highest MRR among the three configured candidates;
- smallest configured model among the candidates, so it is the most practical first real local embedding run once `sentence-transformers` and local model files are available.

Status:

```text
provisional_retrieval_baseline
```

## Citation Compatibility

Citation validation summary:

```text
total question records: 24
valid citation records: 24
invalid citation records: 0
all citations reference retrieved chunks: true
```

Each retrieved chunk receives deterministic labels:

```text
[C1], [C2], [C3], ...
```

Each citation record includes chunk ID, paper ID, arXiv ID, title, section, source type, retrieved rank, retrieval score, and text preview. No citation is generated freely by an embedding model.

## Failure Cases

Misses by configured candidate:

```text
all-MiniLM-L6-v2: eval_v0_q002, eval_v0_q003, eval_v0_q005
e5-small-v2: eval_v0_q001, eval_v0_q003, eval_v0_q004, eval_v0_q006
bge-m3: eval_v0_q001, eval_v0_q002, eval_v0_q003, eval_v0_q004, eval_v0_q005, eval_v0_q006
```

Known interpretation limits:

- hash fallback is lexical and deterministic, not semantic;
- `eval_v0_q002` remains vulnerable to ambiguous wording such as "another selected paper";
- comparison questions can hit a relevant source without retrieving both expected sources;
- unanswerable/source-lookup questions are still retrieval scaffolds, not scientific answer validation.

## Limits

- `sentence_transformers` was unavailable, so no real transformer embedding model was executed.
- No network download was attempted.
- No vector database was introduced.
- No QA generation was run.
- No reranking was added.
- Hit@k against expected source IDs does not prove grounding or scientific correctness.
- Manual review remains required.

## Recommended Next Step

Install or provide local cached `sentence-transformers` models and rerun the same command with real local embeddings. Start with `sentence-transformers/all-MiniLM-L6-v2` because it is small and operationally simple, then compare real `intfloat/e5-small-v2` and `BAAI/bge-m3` under the exact same corpus, eval set, top-k, and citation validation rules.
