# KamiKnows Fase 1F Full-Text Parsing Pilot

## Goal

Fase 1F adds the first controlled full-text parsing pilot for 3-5 HEP/FastCaloSimulation papers. The goal is to produce inspectable per-paper artifacts from arXiv source files and prepare simple chunks that can later be evaluated for RAG readiness.

This phase is still parsing only. It does not call an LLM and does not evaluate scientific correctness.

## Why Prefer LaTeX Source Over PDF

arXiv LaTeX source is preferred because it preserves document structure, section commands, equation environments, labels, and raw mathematical notation better than a PDF text extraction fallback. For this early pilot, PDF fallback is explicitly skipped when it would require heavy dependencies. Skipped PDF fallback is recorded in `source_download.json` and in the quality report.

## Expected Command

```bash
python scripts/run_fulltext_parsing_pilot.py \
  --ids-file data/input/arxiv_ids_fastCalo_5fulltext.txt \
  --output-dir outputs/fulltext_fastcalo_5_v0
```

## Output Layout

```text
outputs/fulltext_fastcalo_5_v0/
  parsing_quality_report.md
  parsing_quality_report.json
  fulltext_manifest.json
  processed/
    papers/
      <safe_paper_id>/
        metadata.json
        source/
        source_download.json
        flat.tex
        plain_text.txt
        paper.json
        sections.json
        equations.json
        chunks.jsonl
```

Some files may be absent for failed papers. The reason is recorded in `source_download.json`, `paper.json`, and the quality report.

## How To Inspect Outputs

Start with the top-level quality report:

```bash
cat outputs/fulltext_fastcalo_5_v0/parsing_quality_report.md
```

Inspect one paper summary:

```bash
cat outputs/fulltext_fastcalo_5_v0/processed/papers/<safe_paper_id>/paper.json
```

Inspect extracted sections:

```bash
cat outputs/fulltext_fastcalo_5_v0/processed/papers/<safe_paper_id>/sections.json
```

Inspect chunks:

```bash
head -n 3 outputs/fulltext_fastcalo_5_v0/processed/papers/<safe_paper_id>/chunks.jsonl
```

## Known Limitations

- LaTeX flattening handles simple `\input{}` and `\include{}` references only.
- Main TeX file detection is heuristic.
- Section extraction is based on `\section`, `\subsection`, and `\subsubsection`.
- Equation extraction is limited to display environments such as `equation`, `align`, `gather`, `\[...\]`, and `$$...$$`.
- Complex macros, custom commands, tables, figures, references, and bibliography entries are not semantically parsed.
- Inline math is kept as plain text where possible, but not normalized.
- PDF fallback is skipped by design in this minimal pilot.
- Chunking is word-based and not yet semantically optimized.

## Not Implemented Yet

- RAG.
- Embeddings.
- Vector database.
- PDF text parsing.
- LaTeX semantic interpretation.
- Claim extraction from full text.
- LLM calls.
- Fine-tuning, LoRA, QLoRA, or training.
- Scientific QA or discovery generation.

## Next Step Toward RAG-Ready Chunks

After the 3-5 paper pilot, manually inspect `plain_text.txt`, `sections.json`, `equations.json`, and `chunks.jsonl`. The next useful step is to improve parsing only where repeated structural errors appear, then define quality criteria for chunks before introducing embeddings or retrieval.
