# Plot extraction guide

Plot extraction turns the figures of the parsed papers into numerical evidence
that the normal text pipeline can retrieve and cite. The generator (Qwen,
Mistral) never looks at an image: each plot becomes a text chunk that
describes it (caption, axes, units, a few characteristic numbers) and links
its digitized values, and `generate.py --solver` computes the numbers a
question needs from those values with deterministic operations.

```text
parse ──> inventory ──> classify ──> calibrate ──> digitize ──> chunk --digitized-dir ──> retrieve ──> generate ──> review
                                        │              │
                                   review.html    digitization-review (source vs reconstruction)
```

All modes are subcommands of one CLI, `scripts/extract_plots.py`. Each mode
reads the previous run directory and writes a new one with a `manifest.json`,
like every other step. [The slides](KamiKnows_plot_extraction.pdf) walk through
the same chain with examples; [QA results](qa_results.md) reports what the
evaluation questions currently show.

## 1. Inventory

```bash
python scripts/extract_plots.py inventory \
  --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/figures_fastcalo_inventory_v1
```

Reads `flat.tex` and the source tree of an existing parse run (nothing is
downloaded again). For every `figure` environment it records the caption, the
LaTeX label and each `\includegraphics` file, resolved against `\graphicspath`
and hashed with SHA-256.

- `figures.jsonl`: one record per figure, with its `source_assets`
  (`<paper>::figure::0001::asset::01`, path, format, hash, LaTeX layout options).
- `panels.jsonl`: `subfigure` panels, when the paper declares them.
- `diagnostics.jsonl`: unresolved files, skipped macros, conditionals.

One figure can have several assets (one file per panel) and one asset can
itself contain several panels: these counts differ. For the five fastcalo
papers: 58 figures, 224 asset references, 217 distinct files, all resolved.

## 2. Classify

```bash
sbatch scripts/run_ollama.slurm scripts/extract_plots.py classify \
  --classifier vision --model gemma3:4b --describe-skipped \
  --inventory-dir outputs/figures_fastcalo_inventory_v1 \
  --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/figures_fastcalo_classified_vision_v1
```

Decides what each asset is, so that calibration knows what to look for.

- `--classifier vision` (default) renders every asset to PNG and sends it,
  one asset per request, to a local Ollama vision model (Gemma 3 4B). The model
  returns a JSON prediction: role (`plot`, `diagram`, `legend`,
  `visual_comparison`), representation (`histogram_1d`, `histogram_2d`,
  `line_plot`, `scatter_plot`, `bar_chart`), single or multi-panel, and whether
  each axis shows numbers and units. It needs a GPU, hence the SLURM runner.
- `--classifier text` uses caption and LaTeX keywords only (CPU, no model).
- `--describe-skipped` also asks for a short description of non-numerical
  assets (diagrams); descriptions are stored but not ingested as evidence.

The prompt is saved in the run (`prompt.txt`) and every raw answer in
`asset_predictions.jsonl`. Model predictions are hints for routing: they never
supply numbers and never veto what the PDF itself shows.

## 3. Calibrate

```bash
python scripts/extract_plots.py calibrate \
  --classified-dir outputs/figures_fastcalo_classified_vision_v1 \
  --inventory-dir outputs/figures_fastcalo_inventory_v1 \
  --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/calibrations_fastcalo_auto_v1
```

Calibration finds, for each quantitative asset, the mapping from page
coordinates to physical values. It is fully automatic; `--figure-ids` and
`--asset-ids` restrict it to a subset.

1. **Source inspection.** The PDF content is inspected (not the extension):
   `vector` (paths and text), `hybrid` (vector axes around an embedded image,
   typical for heatmaps) or `raster` (a bitmap).
2. **Frames and ticks.** Plot frames are detected from vector rectangles; tick
   marks along each frame edge are matched to the nearest numeric text labels.
3. **Scale fit.** Tick positions and values are fitted with a linear and a
   log10 model; a fit is accepted only with at least three consistent labels
   (two labels plus minor-tick evidence for log axes). Multipliers such as
   `×10³` and units in brackets are recorded separately.
4. **Colour bar.** 2D maps also need a calibrated colour bar (the Z axis).
5. **OCR fallback.** When the PDF has no usable text (raster images, labels
   drawn as outlines), RapidOCR reads the labels; OCR proposals need the same
   consistent anchors and stay marked as unverified.

Outputs: `calibration_proposals.json` (the mappings digitization consumes),
`calibration_diagnostics.jsonl` (every asset with its status and reasons) and
`review.html`, with the detected labels and fits drawn on each asset. Assets
without a reliable mapping are not guessed: they get statuses such as
`needs_review_axes`, `needs_review_ocr` or `needs_review_classification`.
`calibration-review` re-renders the HTML of an existing run.

## 4. Digitize

```bash
python scripts/extract_plots.py digitize --backend auto \
  --classified-dir outputs/figures_fastcalo_classified_vision_v1 \
  --inventory-dir outputs/figures_fastcalo_inventory_v1 \
  --parsed-dir outputs/parsed_fastcalo \
  --calibration-file outputs/calibrations_fastcalo_auto_v1/calibration_proposals.json \
  --output-dir outputs/plots_fastcalo_auto_v1
```

Applies each calibration to the asset and extracts the values.

| Source | Backend | What is extracted |
|---|---|---|
| Vector curves and step histograms | `pymupdf` | PDF path coordinates mapped through the X/Y axes; step histograms keep their plateaus as displayed intervals |
| Hybrid 2D maps | `pymupdf` | colour blocks of the embedded image mapped through X/Y and the colour bar |
| Raster images | `raster` | coloured traces or constant-colour blocks from pixels, with OCR-based axes (prototype) |

`--backend auto` picks the backend from the calibration and falls back from
native geometry to pixels when the vector route is unsupported; a forced
backend never switches silently. `digitized_data.jsonl` holds one record per
asset with its series, units, axis scales, calibration, source hash and quality
flags. Failed or skipped assets stay as explicit records with a reason.

Inspect the result before using it:

```bash
python scripts/extract_plots.py digitization-review \
  --digitized-dir outputs/plots_fastcalo_auto_v1 \
  --output-dir outputs/plots_fastcalo_visual_v1
```

`review.html` shows each source next to a plot redrawn only from the extracted
numbers, plus an overlay; `report.pdf` and `exports.zip` (CSV/SVG) contain the
same material.

## 5. Ingest into the chunks

```bash
python scripts/chunk.py sentence --parsed-dir outputs/parsed_fastcalo \
  --exclude-backmatter --digitized-dir outputs/plots_fastcalo_auto_v1 \
  --output-dir outputs/chunks_fastcalo_sentence_plots_v1
```

`--digitized-dir` (one digitization run, holding every digitized plot whatever
its kind) adds plot chunks next to the
text chunks of any chunker; the text chunks are unchanged. **One plotted image,
one chunk, one digitized record**: a plot, or one panel of a multi-panel
figure. A **series** is one extracted object of that image: a curve or
histogram of one colour, or the cells of one heatmap. One image can hold
several series (a blue and a red histogram), and one figure can span several
images (six heatmap panels).

The chunk describes the plot so that retrieval can find it, and never holds
the extracted values:

| Line | Example (Figure 1 of arXiv:1805.00850v2) |
|---|---|
| figure and image | `Plot: Figure 1 of arXiv:1805.00850v2, single image.` |
| panel position and figure layout | `Panel position: bottom-left panel (explicit source layout).` (multi-panel figures only) |
| caption and original LaTeX caption | the original keeps symbols the cleaned caption drops (e.g. `$\gamma$`) |
| paper discussion | the sentences of the paper that cite the figure |
| type and axes | `Type: 1D histogram (drawn steps); 2 series.` / `Axes: X in GeV (linear), Y in A.U. (log10).` |
| one line per series | `- blue curve: 246 bins; X from 40 to 2540 GeV; maximum 0.005347 A.U. at X = 45 GeV.` |
| caveat | `Values estimate the displayed figure, not the original measurements; ...` |

The characteristic numbers (ranges, maximum, tied maxima such as
`(2 tied cells, first shown)`) are computed with `kamiknows.compute`, the same
code that answers the questions, so a card and an answer cannot disagree.
Series are named by their legend label, else by their observed colour.

Every plot chunk carries `plot_link`: the digitized record (`asset_id`,
`data_id`, `extraction_status`, the SHA-256 of `digitized_data.jsonl`) and,
for each series, its compute handle (`<asset_id>/<series_id>`), label and
whether its values are usable. For a multi-panel figure, `figure_panels` lists
every panel with its position and handles, so that a retrieved panel leads to
the panel the question means. A plot that could not be digitized gets a chunk
too, saying why (`Data status: plot not digitized (extraction status:
needs_calibration) ...`), so a question about it finds it instead of a wrong
plot that has values.

Panel positions are added only when the LaTeX source establishes the rows, and
are left out of `retrieval_text` so that position words do not steer ranking.
The chunker refuses a run whose data or source hashes no longer match, or
that records the same image twice. On the fastcalo corpus the digitization of
all 224 images gives 224 plot chunks (119 with data, 105 not digitized; median
209 words), next to 110 text chunks.

Why this layout: earlier layouts put the values in the chunks (a summary plus
blocks of 24 rows per series, 451 plot chunks). Retrieval worked, but 4B and
7B models could not look values up in the long tables: questions on values
other than the peak were answered 0 times out of 8. The values now stay in
`digitized_data.jsonl` and are computed on demand (step 6).

## 6. Retrieve, generate, review

Plot chunks compete with text chunks in the same index; ranking uses the
question text only. With `expected_plot_sources` annotations in the questions
file, the retrieval summary reports separately whether the required
asset/series was retrieved, directly or through the `figure_panels` of a
retrieved panel of the same figure.

At most one plot chunk per figure enters the passages
(`retrieve.py --max-plots-per-figure`, default 1). The panels of a
multi-panel figure share caption and discussion and score almost alike: without
the cap, the six panels of one figure could take all five slots and push out
the figure the question names (Figure 12 instead of Figure 1 of 1805.00850).
The best-ranked panel represents its figure, and its `figure_panels` give the
planner every other panel.

`generate.py --solver` turns the plot chunks into numbers: the model writes a
JSON plan over the operations of `kamiknows.compute`, the plan runs on the
data linked to the retrieved chunks, and the results join the passages as one
more citable passage of type `computed` (see README, "Step 4: generate", and
"Compute"). Without `--solver`, the model only has the card: ranges, maxima,
axes, units.

The prompt lists the passages and then restates the question, which long
contexts need. `generate.py --num-ctx` (default 32768 tokens) fixes the model
context so that long prompts are not cut by a smaller server default; an
answer whose prompt has more characters than `--num-ctx` and for which Ollama
evaluated fewer tokens than a quarter of those characters is flagged
`prompt_truncated_at_context_limit`.

## Supported inputs and limits

- Values describe what the figure displays. Original bin edges, event counts,
  hidden points and experimental uncertainties cannot be recovered, and the
  evidence says so.
- Legend-to-series association is not automatic: series are named by their
  colour, and the caption must say what each colour means.
- Broken axes, overlapping traces, 3D plots and ambiguous layouts stop
  calibration or extraction with a reason rather than producing numbers.
- Agreement between source and reconstruction is a visual check, not a
  scientific validation against original data.

## Optional: original data and benchmark

`recover` reads original numerical sources when they exist: PGFPlots tables
in the LaTeX source, or HEPData tables mapped explicitly to a figure
(`--hepdata-map`, offline unless `--enable-hepdata-network`). `benchmark`
compares a digitization run with independent reference values (`--ground-truth-file`
or a `recover` run). Neither is needed for the QA chain above.

## Code

| Module (`kamiknows/plot_extractor/`) | Responsibility |
|---|---|
| `latex.py`, `assets.py`, `inventory.py` | figure environments, captions, labels, asset resolution |
| `classification.py` | text and vision classifiers, Gemma prompt, rendering |
| `calibration.py` | frames, ticks, scale fits, colour bars, OCR fallback |
| `digitization.py` | axis mapping, PyMuPDF and raster extraction |
| `pipeline.py`, `records.py` | run orchestration, provenance, output records |
| `recovery.py` | PGFPlots and HEPData recovery |
| `validation.py` | structural checks, calibration and digitization reports, benchmark |

Chunk creation for plots is in `kamiknows/chunking.py`, figure linking in
`kamiknows/retrieval.py`. Tests are offline (synthetic PDFs, a fake Ollama).

## Models on the cluster

Gemma, Qwen and Mistral share the Ollama store in
`/storage-hpc/kamiknows/models/ollama`; `scripts/run_ollama.slurm` starts a
private server for each job. A new model is installed once with
`ollama pull <model>` against that store (or by submitting any job with
`OLLAMA_PULL_MODEL=<model>`); never copy files into `blobs/` by hand.
