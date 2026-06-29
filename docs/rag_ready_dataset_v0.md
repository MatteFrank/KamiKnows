# KamiKnows Fase 1G RAG-Ready Dataset v0

## Goal

Fase 1G consolidates the Fase 1F full-text parsing outputs into a small, validated, RAG-ready dataset. The result is ready for manual inspection and later retrieval experiments, but it is not a RAG system yet.

This step creates:

- `chunks.jsonl`
- `papers.jsonl`
- `equations.jsonl`
- `eval_questions_v0.jsonl`
- `rag_manifest_v0.json`
- `chunk_quality_report.json`
- `chunk_quality_report.md`

## Input From Fase 1F

The input is the existing full-text parsing pilot directory:

```text
outputs/fulltext_fastcalo_5_v0/
```

The builder reads each paper directory under:

```text
outputs/fulltext_fastcalo_5_v0/processed/papers/<safe_paper_id>/
```

It uses:

- `paper.json`
- `chunks.jsonl`
- `equations.json`

Only papers with `parsing_status` equal to `success` or `partial` are included.

## Command

```bash
python3 scripts/build_rag_ready_dataset.py \
  --fulltext-dir outputs/fulltext_fastcalo_5_v0 \
  --output-dir outputs/rag_ready_fastcalo_v0 \
  --domain "HEP / FastCaloSimulation / calorimetry"
```

## Output Layout

```text
outputs/rag_ready_fastcalo_v0/
  chunks.jsonl
  papers.jsonl
  equations.jsonl
  eval_questions_v0.jsonl
  rag_manifest_v0.json
  chunk_quality_report.json
  chunk_quality_report.md
```

## Chunk Schema

Each `chunks.jsonl` record contains:

```json
{
  "chunk_id": "...",
  "paper_id": "...",
  "arxiv_id": "...",
  "title": "...",
  "source_type": "section",
  "section_id": "...",
  "section_heading": "...",
  "text": "...",
  "word_count": 0,
  "metadata": {
    "domain": "HEP / FastCaloSimulation / calorimetry",
    "source_pilot": "outputs/fulltext_fastcalo_5_v0",
    "paper_dir": "...",
    "source_file": "...",
    "parsing_status": "success",
    "source_type_original": "latex_source"
  }
}
```

Allowed `source_type` values are:

```text
section
abstract
equation_context
```

The current Fase 1F chunks are section chunks.

## Eval Question Scaffold

`eval_questions_v0.jsonl` contains static source-constrained evaluation questions. The scaffold does not include generated answers and does not call an LLM.

Each record contains:

- `question_id`
- `question`
- `question_type`
- `expected_source_arxiv_ids`
- `expected_section_keywords`
- `evaluation_criteria`
- `notes`

The scaffold includes method, limitation, comparison, source-lookup, and intentionally insufficient-evidence questions.

## Validation Checks

The dataset builder checks:

- globally unique chunk IDs;
- required chunk fields;
- required chunk metadata fields;
- non-empty chunk text;
- word-count reasonableness;
- valid chunk `source_type`;
- no orphan chunks;
- papers without chunks.

The validation result is written to:

```text
rag_manifest_v0.json
chunk_quality_report.json
chunk_quality_report.md
```

## How To Inspect Chunks

Show the first few chunks:

```bash
head -n 3 outputs/rag_ready_fastcalo_v0/chunks.jsonl
```

Count chunks:

```bash
wc -l outputs/rag_ready_fastcalo_v0/chunks.jsonl
```

Inspect the readable quality report:

```bash
cat outputs/rag_ready_fastcalo_v0/chunk_quality_report.md
```

Inspect the manifest:

```bash
cat outputs/rag_ready_fastcalo_v0/rag_manifest_v0.json
```

## Limitations

- No embeddings are created.
- No vector database is created.
- No retrieval is implemented.
- No LLM is called.
- No generated answers are produced.
- Chunks inherit the limitations of the Fase 1F LaTeX parser.
- Current chunks are simple section-derived word chunks, not semantically optimized chunks.
- Evaluation questions define source constraints and criteria only.

## Next Step Toward Fase 1H Mini-RAG

Before implementing mini-RAG, inspect `chunk_quality_report.md`, sample `chunks.jsonl`, and `eval_questions_v0.jsonl`. The next step is to choose a tiny retrieval experiment over this validated dataset while keeping embeddings/vector DB/retrieval logic clearly separate from dataset generation.
