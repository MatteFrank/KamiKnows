# KamiPlotExtractor — Task 01

KamiPlotExtractor inventories figures in **existing local parse runs**. It is a
separate importable package at `kamiknows/kamiplotextractor/`. Task 01 does not
classify plots, extract numbers, inspect pixels, render PDFs or change RAG chunks.
A figure may contain a plot, a diagram, several plots, or a mixture: every figure
and panel currently has type `unknown`. A resolved image reference is not evidence
of a successful numerical extraction or of scientific correctness.

## Running

From the repository root, using the existing KamiKnows Python environment:

```bash
python scripts/extract_plots.py \
  --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/figures_fastcalo_inventory_v1 \
  --mode inventory
```

`inventory` is the only implemented mode. No new dependencies, downloads, external
executables, models, TeX compilation or cloud services are needed. Use a new
directory for every run, adding `_retry1`, `_retry2`, etc. when necessary. Output
inside the input parse run is rejected. Existing outputs and original assets are
never modified or copied over.

After a real inventory run, add a note to `docs/outputs_index_notes.json`, identifying
the baseline or stating that this is the first inventory, and run:

```bash
python scripts/index_outputs.py
```

The index recognizes the `extract_plots` manifest step. Tests use temporary
directories and do not create indexed production runs.

Python API:

```python
from pathlib import Path
from kamiknows.kamiplotextractor import run_inventory

summary = run_inventory(
    parsed_dir=Path("outputs/parsed_fastcalo"),
    output_dir=Path("outputs/figures_fastcalo_inventory_v1"),
)
```

`FigureExtractor().extract(paper_id=..., flat_tex=..., source_root=..., sections=...)`
also works without a run or CLI. `source_root` is the paper's `source/` directory;
`sections` is the saved list from `sections.json`. It returns `(figures, panels,
diagnostics, source_inputs)`, with dataclass records exposing `to_dict()` and a
mapping of source files read for manifest hashing. `PanelDetector` is independently
importable and consumes the scanner's document and figure environment objects.

## Inputs and integration audit

The pipeline checks `common.read_manifest(parsed_dir, "parse")` and reads
`papers.jsonl`. Paper IDs are taken verbatim from those records, checked for safe
single-component names and uniqueness. Per paper it reads:

```text
papers/<paper_id>/flat.tex
papers/<paper_id>/sections.json     optional; missing means no section link
papers/<paper_id>/source/           includes source/extracted/ for tar e-prints
```

The arXiv downloader and parser are reused through these artifacts, without
downloading or parsing arXiv again. The existing text-normalization and section
heading conventions are imported from `parsing.py`; its behavior is unchanged.

Two existing contracts constrain provenance:

1. `flat.tex` has no source map. The original flattener also resolves includes
   relative to its root, includes each file once, and turns unresolved inputs into
   comments. Inventory reports those comments and remaining input commands. It
   does not re-expand the document or invent figures missing from `flat.tex`.
2. `sections.json` has IDs and headings, without source offsets. Inventory links
   to these IDs only when the heading sequence and levels match. A mismatch or a
   figure before the first section yields `section_id: null` with a warning.
   A saved `Full text` section can cover a document without section commands.

The original requirements file contains two sentence-transformers constraints;
Task 01 leaves the RAG environment unchanged and adds no dependencies.

## Scanner and supported LaTeX

Commands use balanced brace/bracket scanning; nested environments use a stack.
Comments, `\verb`, common verbatim environments and standard macro/environment
definitions are excluded from discovery. Skipped definitions are reported because
their expansions cannot be inventoried. Conditional branches are not evaluated:
their literal figures are retained with a diagnostic.

Supported inventory constructs:

- `figure` and `figure*`, including nested layout environments such as `center`.
- `\includegraphics` and `\includegraphics*`, multiple or repeated references,
  optional arguments with nested braces, nested paths and omitted extensions.
- `\graphicspath{{directory/}{other/}}`, including ordinary group/environment
  scoping and settings found in included original source files.
- PNG, JPG/JPEG, PDF, EPS and SVG references. Format comes from the extension;
  Task 01 does not decode or validate the image contents. Omitted extensions are
  matched against actual directory entries, with case-insensitive extension checks.
- Captions with optional short captions, starred captions, nested formatting and
  labels (including labels inside captions). Multiple figure captions and labels
  are retained; the primary fields expose the first.
- Explicit panels: `subfigure` and `subcaptionblock` environments,
  `\subfigure[caption]{...}`, `\subfloat[...][caption]{...}` and
  `\subcaptionbox{caption}[width][position]{...}`. `\subcaption` inside an
  explicit panel is read as its caption. Panels are LaTeX regions, not crops.

The scanner keeps half-open **character offsets** into the decoded `flat.tex`,
including its original whitespace and comments. `caption_raw` retains the original
caption argument; `caption` is best-effort plain text using KamiKnows normalization,
not a faithful mathematical rendering. Raw LaTeX remains authoritative.

Layout options are retained as strings/flags, not interpreted: width, height,
scale, trim, viewport, clip, angle, origin and selected PDF page, plus any additional
keys in `options`. `options_raw` and the full `includegraphics_raw` preserve option
order and older two-bracket syntax. No crop, rotation or page selection is applied.

## Asset resolution and safety

Discovery uses `flat.tex`. For each graphics occurrence, the resolver finds matching
original `includegraphics` commands by literal reference, options and star status.
Matching whole-figure text and the occurrence number narrow the source location
when possible. If a figure spans includes, command matching can still recover its
origin. All remaining candidate source locations are retained, with an ambiguity
warning when there is more than one.

Candidates are searched relative to original source-file directories, `source/`,
`source/extracted/` when present, and applicable `graphicspath` entries. Only a
**unique** resolved asset is selected. Different valid extensions or files in
different plausible directories yield `asset_ambiguous`, candidate paths, and null
path/hash fields. This is deliberately conservative; it does not emulate a TeX
engine's extension/search precedence. No recursive basename search or semantic
filename similarity is used. Uncertain origin and successful path resolution are
reported separately.

Every path is checked after resolving symlinks, before reading. Absolute paths,
URLs, `..` components (even when they could stay inside the root), and unresolved
macros are rejected. Symlink directories are not recursively scanned. A linked
file outside the source root cannot be read or hashed. No commands in the document
are executed. Excessive group nesting is stopped at 128 levels with a diagnostic.
Missing files, unsafe paths, unsupported extensions, malformed commands and
unclosed environments remain visible in diagnostics. Graphics outside `figure`
and `figure*` are reported explicitly rather than silently treated as figures.

## Output schema 1.0

The schema version is `1.0` in every record, the summary and manifest parameters.
IDs are based on the existing paper ID and source order:

```text
<paper_id>::figure::0001
<paper_id>::figure::0001::panel::01
<paper_id>::figure::0001::asset::01
```

They are deterministic for unchanged input. Inserting/reordering figures or panels
changes subsequent IDs; these are inventory-version identities, not permanent
cross-revision scientific identifiers. Repeated references have distinct asset
occurrence IDs and share the same path/hash when resolved.

| File | Contents |
|---|---|
| `figures.jsonl` | One `FigureRecord` per discovered figure, including unresolved/empty figures. |
| `panels.jsonl` | One `PanelRecord` per explicit subfigure; empty when none exist. |
| `diagnostics.jsonl` | Structured warnings/errors with paper/figure/asset context when available. |
| `figure_summary.json` | Counts of papers, figures, panels, reference resolution, unique assets, statuses and diagnostic codes; per-paper outcomes. |
| `manifest.json` | Standard KamiKnows manifest, step `extract_plots`, with command, git state, parameters and input SHA256 hashes. |

`FigureRecord` fields:

- `paper_id`, `figure_id`, `schema_version`.
- `caption`, `caption_raw`, `captions` (all raw/normalized/short/starred variants),
  `latex_label`, `latex_labels`, `section_id` (nullable).
- `source_assets`: every graphics occurrence, including unresolved references;
  each has `asset_id`, nullable `panel_id`, literal `reference`, nullable `path`,
  `format`, `sha256`, status `resolved|unresolved`, candidate paths, source command
  origins, offsets, graphics paths, raw options, parsed options, layout and warnings.
- `source_hashes`: resolved source-relative asset paths mapped to SHA256.
- `panels`: panel IDs; `figure_type: "unknown"`,
  `classification_confidence: null`, `classification_method: "not_classified"`.
- `provenance`: paths relative to the parse run, figure offsets and environment,
  `manual_review: "pending"`; `warnings`; status `success|partial`.

`PanelRecord` fields: `paper_id`, `figure_id`, `panel_id`, `schema_version`, nullable
`panel_label` (the explicit LaTeX label, not an inferred printed “(a)”),
`source_region` (LaTeX span and construct, `image_region: null`), `source_asset_ids`,
nullable raw/normalized caption, `latex_labels`, `panel_type: "unknown"`,
`plot_type: null`, and `warnings`. Asset ownership uses the innermost explicit panel.

Diagnostics have `schema_version`, `severity: warning|error`, `code`, `message`,
nullable `offset`, and context such as `paper_id`, `figure_id`, `asset_id`, `reference`,
`source_file` or `line_number`. Offsets refer to `flat.tex` unless source-file context
is supplied. Paper input errors do not prevent processing other papers. A run can
finish with failed papers; inspect `paper_status_counts` and diagnostics, not just
the CLI exit code. Missing top-level inputs/wrong manifest steps fail before
creating an output directory. A corrupt per-paper sections file marks that paper
failed; a missing sections file permits inventory with null section IDs.

The manifest hashes the parse manifest, `papers.jsonl`, every read `flat.tex` and
`sections.json`, all scanned original `.tex` files, and resolved assets. Missing or
rejected files have no invented hashes. Asset paths in records are relative to that
paper's `source/`; the manifest paths locate actual inputs. Preserve the parse run
alongside the inventory for later reuse.

## Deferred work and scientific contract

Task 02 classification is **not implemented**. Neither are automatic graphical
panel segmentation, `\captionof` figures outside floats, arbitrary wrapper macros,
TeX expansion/catcode changes, conditional execution, custom graphic extension
rules, TikZ/PGFPlots data extraction, HEPData, original CSV/DAT/ROOT extraction,
numerical validation, plot reconstruction, compare mode or RAG chunk enrichment.
Bare `\subcaption` without a supported explicit panel does not establish a crop or
panel boundary. Exact source provenance can remain ambiguous because the existing
parse step provides no source map; this is surfaced, not repaired by guessing.

No numerical `PlotDataRecord` is emitted in Task 01. The future numerical schema
will link `paper_id`, `figure_id`, `panel_id` and include `source_method`, `backend`,
`backend_version`, `axes`, `units`, `series`, `points`, `bins`, `uncertainties`,
`digitization_quality`, `validation_status`, and `provenance`. Its source method must
distinguish `original_source_data`, `original_published_data`,
`vector_reconstructed`, `raster_digitized`, and `unavailable`. Unrecoverable values
must be null; experimental uncertainties must remain separate from digitization
errors. An unresolved image reference does not produce fabricated numerical data.

Planned backend choices, **none implemented or required by Task 01**:

| Backend | Intended boundary |
|---|---|
| `auto` | Explicit, recorded selection policy; never a silent fallback. |
| `pymupdf` | Optional Python adapter for vector PDF primitives. |
| `plotdigitizer` | Optional CLI/Python integration after checking supported raster cases and installation. |
| `chartdetective` | Interactive workflow with validated export/import; no assumed automation API. |
| `plotsift` | Interactive image/error-bar workflow with validated export/import. |

Future adapters must expose availability, provenance and version, keep backend
failures separate from extraction quality, and preserve separate results when
comparing backends on one plot. Scientific priority remains explicit original
source data, verified published data, vector reconstruction, raster digitization,
and finally metadata alone. Filenames or semantic similarity alone cannot establish
a link to original numerical data.

## Verification and readiness

`pytest -q` includes offline fixtures for supported formats, nested LaTeX, explicit
panels, source-file origin matching, ambiguous/missing/unsafe references, unchanged
asset bytes, deterministic IDs, manifests, partial failures and CLI execution. No
real paper benchmark or scientific evaluation is implied. **Manual review pending.**

The inventory API and versioned records provide the input boundary for Task 02.
Task 02 requires a separate instruction; this module is not the completed
KamiPlotExtractor project.
