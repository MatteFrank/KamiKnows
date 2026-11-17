# Plot extraction guide

Plot extraction turns the figures of the parsed papers into numerical evidence
that the normal text pipeline can retrieve and cite. The generator (Qwen,
Mistral) never looks at an image: it receives the extracted values as text
passages, with units, axis scales and provenance, exactly like paper text.

```text
parse ──> inventory ──> classify ──> calibrate ──> digitize ──> chunk --digitized-dirs ──> retrieve ──> generate ──> review
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
  --exclude-backmatter --digitized-dirs outputs/plots_fastcalo_auto_v1 \
  --output-dir outputs/chunks_fastcalo_sentence_plots_v1
```

`--digitized-dirs` (one or more digitization runs) adds plot chunks next to the
text chunks of any chunker. Only successful series are added. For each series:

- a **summary** chunk: figure, source image and series (e.g. "Figure 1, source
  image 1 of 1, blue curve"), caption, the paper sentences that cite the figure,
  axis scales and units, peak and range;
- **coordinate table** chunks with the extracted values, split to fit the word
  budget;
- one **figure overview** per figure, listing the peak of every panel, so that
  sibling panels can be compared.

Every plot chunk carries `plot_provenance` (asset, series, hashes) and
`axis_metadata` (unit, unit status, scale, multiplier), which survive retrieval.
Values already include display multipliers; `log10` describes axis spacing, not
a unit. Panel positions ("top-left") are added only when the LaTeX source
establishes the rows. The chunker refuses runs whose data or source hashes no
longer match.

## 6. Retrieve, generate, review

Nothing changes in the commands: use the new chunk directory with
`retrieve.py`, `generate.py` and `review.py` (see README). Plot chunks compete
with text chunks in the same index. When a panel is retrieved, its figure
overview is attached once, so that a question about two panels sees both.
With `expected_plot_sources` annotations in the questions file, the retrieval
summary reports separately whether the required asset/series was retrieved.

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
