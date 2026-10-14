# Semantic chunking v1 with MiniLM and diagnostic plots

## Objective

Implement a contiguous semantic chunker and inspectable plots, without running
MiniLM on the real corpus or launching retrieval/Qwen experiments.

## Inputs and algorithm

Reuse the exact baseline paper selection from `outputs/rag_ready_fastcalo_v1`
and its existing `sections.json` sources. Use the same heuristic sentence splitter
as sentence grouping v1. Preserve bibliography and parser artifacts.

The existing `EmbeddingEncoder` is required to use real, locally cached
`sentence-transformers/all-MiniLM-L6-v2`, on CPU, with normalized embeddings.
There is no hash fallback and no automatic model download. Each whole sentence
is encoded once. Inputs fitting the actual `max_seq_length` use the existing encoder unchanged.
Longer inputs are split into nonoverlapping windows of exact content token IDs,
reserving space for special tokens in each window. Each window embedding is
normalized; their content-token-length-weighted average is normalized again. No
decode/re-tokenize round trip or content truncation occurs. The original sentence
stays intact in chunk text. This representation is an approximation and may differ
from encoding the entire sentence with a long-context model.

For boundary i, average up to two sentence vectors on each side and compute the
cosine between these window means. At section edges, windows are shorter. The
threshold is the linearly interpolated 25th percentile of all boundary similarities
within the section; a candidate must be strictly below it. Equal scores therefore
do not create semantic cuts. This rule is fixed without evaluation questions.

Within each chunk, eligible cuts must be after newly covered content and between
250 and 400 whitespace words (including overlap). Choose the lowest eligible
similarity, breaking ties by earlier position. If no candidate qualifies, use the
last complete sentence fitting within 400 words. If the remaining section fits,
retain it as the final group without further semantic splitting. Tails can be
below 250 words. Single oversized sentences remain intact and flagged, with token-window encoding when necessary. Overlap is a whole-sentence suffix within 60
words, reduced if it prevents new content from entering the next chunk.

## Files created or changed

- New `kamiknows/fulltext/semantic_chunking.py`: scores, thresholds, decisions,
  contiguous groups, size limits, overlap and section diagnostics.
- New `kamiknows/rag_ready/semantic_dataset.py`: local encoder, token checks,
  construction via the existing builder and plotting.
- New `scripts/build_semantic_dataset_v1.py`: CLI, no retrieval/generation.
- New `tests/test_semantic_chunking.py`: synthetic topic changes, coverage,
  overlap, uniform similarities, section isolation, invalid vectors and plots.
- Changed `kamiknows/rag_ready/sentence_dataset.py`: optional injected generator
  and strategy; preserve cut reasons. Default sentence algorithm and manifest
  version remain unchanged.

## User commands (not executed on the real corpus)

In `kamiknows`, matplotlib is missing; sentence-transformers is present. Install
plotting support, then build:

```bash
cd /home/e/eapicell/KamiKnows
conda activate kamiknows
python -m pip install 'matplotlib>=3.9'
python scripts/build_semantic_dataset_v1.py \
  --reference-dir outputs/rag_ready_fastcalo_v1 \
  --sentence-comparison-dir outputs/rag_ready_fastcalo_sentence_v1 \
  --output-dir outputs/rag_ready_fastcalo_semantic_v1 \
  --min-words 250 --target-words 400 --overlap-words 60 \
  --window-sentences 2 --threshold-quantile 0.25
```

Existing output directories are refused. If plotting fails after dataset writing,
the summary explicitly records FAIL and the command fails; preserve the partial
run for inspection and use a fresh directory after addressing the error.

## Output artifacts

- Existing compatible dataset files, including `chunks_v1.jsonl` and
  `rag_manifest_v1.json`; new IDs use `<paper_id>::semantic_v1::<index>`.
- `semantic_boundary_scores_v1.jsonl`: section, sentence boundary, word position,
  similarity, threshold, candidate status, selection and decision reason.
- `semantic_chunk_spans_v1.jsonl`: chunk span in word coordinates, actual overlap,
  length and cut reason. Section-final cuts are included here, without invented
  similarity values.
- `semantic_chunking_summary_v1.json`: encoder settings, thresholds, section means,
  global boundary mean, cut counts and plot status/paths.
- `plots/<paper_id>_section_<index>.png` and `.pdf`: similarity curve, section mean,
  threshold, unselected candidates, selected cut types, chunk spans and overlap.
  Even single-sentence sections get a span plot (no similarity measurements).
- `plots/semantic_overview_v1.png` and `.pdf`: score histogram, chunk-length
  comparison (word/sentence/semantic) and cut-reason counts.

Candidate rejection labels deliberately combine size constraints and competing
stronger boundaries; they do not pretend to isolate a unique cause.

## Validation and commands run

- Full suite in kamiknows: 229 passed, 3 skipped (plot test skips because matplotlib
  is absent, in addition to the two pre-existing skipped tests).
- CLI `--help` and `git diff --check` passed.
- Standalone synthetic PNG/PDF rendering checked using existing matplotlib in
  caloinn: six nonempty files verified, including semantic boundaries, overlap
  and a single-sentence section. The plotting function was isolated from unrelated
  package imports and `MKL_THREADING_LAYER=GNU` was set for that test process;
  initial mixed-environment attempts failed on missing jsonschema and an MKL
  symbol error. No persistent environment settings were changed. Pytest itself
  is absent from caloinn.
- No dependencies installed, real dataset constructed, models loaded, benchmark
  executed or GPU job submitted by the agent.

## Metrics, warnings and limits

No experimental retrieval or QA metrics exist yet for semantic chunks. Plot scores
measure representation similarity, not scientific evidence relevance. The model is
pretrained and receives no training on this corpus. Long-input token windows are recorded explicitly; no content tokens are dropped. Source hashes and sentence
coverage are recorded using the shared dataset builder. No question schema or
retrieval metric changes are introduced.

## Next step

The user installs matplotlib and runs construction. Inspect plots, lengths, coverage
and selected boundaries before running the same v2 TF-IDF evaluation, then Qwen.


## Long-input correction

The first blocked input was a figure-file list and caption in 1712.10321v1,
Shower Shapes: 53 whitespace words but 261 tokens including special tokens.
Read-only tokenizer inspection found seven over-limit inputs, maximum 614 tokens.
The builder now preserves these texts and encodes exact token windows instead of
aborting. `semantic_long_inputs_v1.jsonl` records paper, section, sentence index,
full text, original token count and window lengths for each affected input.
The summary/manifest record the policy and number of affected sentences.
This changes only the semantic representation, not the source text, sentence
boundaries or retrieval implementation. Rerun the same construction command.

Tests cover exact token coverage, weighted short tails, normalization, special
tokens, padding attention masks and invalid vectors, without loading MiniLM.
After correction: full suite 233 passed, 2 skipped; plotting tests now also pass
in kamiknows (matplotlib has become available). `git diff --check` passed.
The real dataset builder was not rerun by the agent.

## Tokenizer API compatibility correction

The installed BertTokenizer has no prepare_for_model method. Window inputs now
use exact content IDs with CLS/SEP, after checking the actual empty-input special
token template and special-token budget. Attention masks and token-type IDs are
constructed explicitly, then padded by the tokenizer. A test against the real
locally cached MiniLM tokenizer verifies equivalence to ordinary tokenization and
padding masks; no model weights or corpus encoding are invoked.
