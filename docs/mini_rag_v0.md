# KamiKnows Fase 1H Mini-RAG v0

## What RAG Means Here

In this phase, RAG means a minimal local pipeline:

```text
question
-> retrieve relevant chunks from the local RAG-ready corpus
-> optionally pass retrieved chunks to an LLM
-> generate a grounded answer
-> preserve chunk_id/arXiv citations for manual review
```

This is an internal demo over five FastCaloSimulation/calorimetry papers. It is not a reliable scientific QA system.

## Input Dataset From Fase 1G

Mini-RAG v0 reads:

```text
outputs/rag_ready_fastcalo_v0/
  chunks.jsonl
  papers.jsonl
  equations.jsonl
  eval_questions_v0.jsonl
  rag_manifest_v0.json
```

## Retrieval-Only Command

```bash
python3 scripts/run_mini_rag.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_v0_fastcalosim_retrieval_only \
  --retrieval-only \
  --top-k 5
```

Retrieval-only mode does not call Ollama or any LLM. It still writes skipped answer records so downstream output shapes can be inspected.

## Ollama/Qwen Generation Command

```bash
python3 scripts/run_mini_rag.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_v0_fastcalosim \
  --backend ollama \
  --model qwen3:0.6b \
  --top-k 5
```

If Ollama or the model is unavailable, generation records are written with `generation_status: "backend_error"` and reports are still produced.

## Output Layout

```text
outputs/rag_v0_fastcalosim/
  rag_run_config.json
  retrieved_contexts.jsonl
  rag_answers.jsonl
  rag_eval_summary.json
  rag_error_analysis.md
  rag_manual_review_checklist.md
  rag_manifest.json
```

## Retriever Limitations

The default retriever is `tfidf_local_v0`, a small in-memory TF-IDF/cosine retriever implemented with the Python standard library.

Limitations:

- lexical matching only;
- no embeddings;
- no semantic vector search;
- no query expansion;
- no reranking;
- no vector database;
- small-corpus only.

## How Citations Work

Each retrieved context record preserves:

- `chunk_id`
- `paper_id`
- `arxiv_id`
- `title`
- `section_id`
- `section_heading`
- retrieved text

Generation prompts instruct the model to cite both `chunk_id` and `arxiv_id`. Because models may not produce perfectly structured citations, `rag_answers.jsonl` also stores `retrieved_chunk_ids` independently from generated citation text.

Citation correctness is not marked as PASS automatically. Manual review is required.

## Manual Review

Review:

```text
outputs/rag_v0_fastcalosim/rag_manual_review_checklist.md
```

For each question, inspect:

- whether retrieved chunks are useful;
- whether the answer is grounded;
- whether citations are correct;
- whether hallucination is absent;
- whether the model abstained when context was insufficient;
- outcome: `pass | revise | reject | unclear`.

## Error Taxonomy

`rag_error_analysis.md` includes:

- `retrieval_miss`
- `answer_not_grounded`
- `wrong_citation`
- `hallucination`
- `incomplete_answer`
- `failed_abstention`
- `json_or_format_error`
- `backend_error`
- `empty_context`
- `unknown_error`

Only retrieval misses and backend errors are counted automatically in v0. Grounding and citation categories remain not reviewed until manual inspection.

## What This Demonstrates

- The Fase 1G dataset can be loaded.
- Local deterministic retrieval works over `chunks.jsonl`.
- Retrieved contexts are traceable to chunk and arXiv IDs.
- Evaluation-question scaffolds can drive retrieval and optional generation.
- Outputs are reviewable and manifestable.

## What This Does Not Demonstrate

- Scientific correctness.
- Robust RAG.
- Production retrieval quality.
- Complete citation reliability.
- Embedding quality.
- Vector DB performance.
- Discovery generation.
- Fine-tuning or model improvement.

## Why This Is Not Fase 2 Robust RAG Yet

Fase 1H uses lexical retrieval and manual review scaffolding. A robust Fase 2 system would need explicit retrieval evaluation, stronger chunking policy, semantic retrieval or hybrid retrieval, citation verification, model-output validation, broader corpora, and repeated manual QA before scientific use.
