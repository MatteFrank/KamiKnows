# KamiPlotExtractor — Tasks 01–03

KamiPlotExtractor inventories figures in **existing local parse runs**. It is a
separate importable package at `kamiknows/kamiplotextractor/`. Task 01 does not
classify plots, extract numbers, inspect pixels, render PDFs or change RAG chunks.
A figure may contain a plot, a diagram, several plots, or a mixture: inventory
records start with type `unknown`. Task 02 classifies saved metadata in a separate
run, without inspecting images. A resolved image reference is not evidence
of a successful numerical extraction or of scientific correctness.
Task 03 recovers explicitly linked native PGFPlots data and mapped HEPData tables.
It performs no raster/vector digitization and does not modify RAG chunks.

## Running

From the repository root, using the existing KamiKnows Python environment:

```bash
python scripts/extract_plots.py \
  --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/figures_fastcalo_inventory_v1 \
  --mode inventory
```

Task 02 reads that inventory directly:

```bash
python scripts/extract_plots.py \
  --inventory-dir outputs/figures_fastcalo_inventory_v1 \
  --output-dir outputs/figures_fastcalo_classified_v1 \
  --mode classify
```

The inventory remains schema `1.0`; classified output uses schema `1.1`. The
`classified_v1` run tag denotes the first classification rule set, not schema 1.0.
`--parsed-dir` is required for inventory and recovery; `--inventory-dir` is required
for classify and can locate relocated inventory provenance during recovery.
Inventory/classify need no new dependencies, downloads, external executables,
models, TeX compilation or cloud services. Recovery is offline by default; YAML
support uses optional PyYAML and online HEPData downloads require an explicit flag.
Use a new
directory for every run, adding `_retry1`, `_retry2`, etc. when necessary. Output
inside the input run is rejected. Existing outputs and original assets are
never modified or copied over.

After a real inventory, classification or recovery run, add a note to `docs/outputs_index_notes.json`, identifying
the baseline or stating that this is the first inventory, and run:

```bash
python scripts/index_outputs.py
```

The index recognizes the `extract_plots` manifest step. Tests use temporary
directories and do not create indexed production runs.

Python API:

```python
from pathlib import Path
from kamiknows.kamiplotextractor import run_classification, run_inventory

summary = run_inventory(
    parsed_dir=Path("outputs/parsed_fastcalo"),
    output_dir=Path("outputs/figures_fastcalo_inventory_v1"),
)
classification_summary = run_classification(
    inventory_dir=Path("outputs/figures_fastcalo_inventory_v1"),
    output_dir=Path("outputs/figures_fastcalo_classified_v1"),
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

## Inventory output schema 1.0

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

Task 02 implements only the textual classification described below. Automatic graphical
panel segmentation, `\captionof` figures outside floats, arbitrary wrapper macros,
TeX expansion/catcode changes, conditional execution, custom graphic extension
rules, arbitrary TikZ expressions, ROOT parsing, plot reconstruction, scientific
validation and RAG chunk enrichment
are not implemented.
Bare `\subcaption` without a supported explicit panel does not establish a crop or
panel boundary. Exact source provenance can remain ambiguous because the existing
parse step provides no source map; this is surfaced, not repaired by guessing.

No numerical `PlotDataRecord` is emitted by inventory/classify. Recovery emits
schema 1.2 records described below. Vector reconstruction and raster digitization
remain deferred. Unrecoverable values remain null; experimental uncertainties
are never replaced with digitization errors. An unresolved image reference does
not produce fabricated numerical data.

Planned digitization backend choices, **none implemented or required by Tasks 01–03**:

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

## Task 02: conservative classification

`FigureClassifier` lives in the existing `inventory.py`, alongside the extractor
and panel detector. It consumes `FigureRecord` and explicit `PanelRecord` objects
and returns classified copies, without mutating its arguments:

```python
from kamiknows.kamiplotextractor import FigureClassifier, FigureRecord, PanelRecord

# figure_data and panel_data are records read from inventory JSONL files.
classified_figure, classified_panels = FigureClassifier().classify(
    FigureRecord(**figure_data), [PanelRecord(**item) for item in panel_data]
)
```

The run entry point additionally validates the input `extract_plots` manifest,
`mode: inventory`, schema `1.0` in the records/summary/diagnostics, summary agreement
with the manifest, record counts, unique IDs, required field types and figure/panel/
asset links. Invalid inputs fail before output creation. It requires all five
inventory files, but never opens a parse run, LaTeX source or asset. Already
classified runs are not accepted as inventory inputs. The existing inventory API,
`SCHEMA_VERSION = "1.0"`, IDs and serialized inventory format remain unchanged;
`CLASSIFIED_SCHEMA_VERSION = "1.1"` describes the new output.

### Rules and evidence

Method `text_rules_v1` uses explicit patterns listed in `CLASSIFICATION_RULES`:

| Class | Examples of decisive caption evidence |
|---|---|
| `plot` | histogram, scatter plot, ROC curve, heatmap, transverse momentum distribution, invariant mass spectrum, efficiency versus a variable, resolution as a function of a variable, Data/MC ratio, uncertainty band, distribution accompanied by error bars |
| `generic` | detector schematic/diagram, experimental setup, photograph, architecture diagram, processing workflow/flowchart, event display |
| `mixed` | at least one explicitly classified plot panel and one generic panel |
| `unknown` | insufficient caption evidence or unresolved contradictory evidence |

Saved normalized captions and raw caption arguments supply decisive evidence;
additional figure captions are considered as well. Raw arguments undergo only
text/markup normalization, not document parsing or macro execution. Each match
records its rule, field of origin, matched normalized text, target class/type,
polarity and whether it can decide a class. LaTeX labels and asset references/paths
provide **non-decisive** hints: they cannot classify a captionless figure or override
its caption. Extensions and asset-resolution status never decide a class.

The words `data`, `distribution`, `model`, `simulation`, `figure`, `calorimeter`,
colours and scales alone are insufficient. Thus an event display with an energy
scale remains generic unless other explicit evidence creates a conflict.

A local negation heuristic recognizes `no`, `not`, `without`, `rather than` and
`instead of` within up to three intervening words in the same sentence. Negated
matches cannot support a class; positive and negated occurrences of the same rule
are contradictory. For example, a histogram *without an uncertainty band* can
still be a plot. Opposing affirmative plot/generic caption signals yield unknown
when no explicit panel structure resolves the composition. No numeric confidence
is computed: **`classification_confidence` is always null**, not a calibrated
probability, and not an implicit claim of certainty. Every classification has
`classification_review_status: "pending"`.

### Panel interpretation

Only existing explicit panels are classified, using their own captions and owned
asset hints. The global figure caption is never copied into panel evidence. Panel
classes are `plot`, `generic`, `unknown`; plot types are `histogram`, `scatter`,
`curve`, `heatmap`, `ratio`, `other`, `unknown`. Generic/unknown panels have
`plot_type: null`. Plot panels with no unique supported subtype use `unknown`.
Mentioning error bars or an uncertainty band is a textual indicator, not extraction
of their values.

Aggregation preserves panel IDs and applies:

- All panels plot → figure plot; all generic → generic.
- A known plot and a known generic panel → mixed, even if other panels remain
  unknown: both content categories are already evidenced.
- Otherwise, unknown panels prevent a homogeneous classification; the figure is
  unknown. A global caption does not fill missing panel evidence.

Specific panel evidence takes priority over the global caption. A conflicting
global caption is retained in `contradictory_indicators` for review while the figure
keeps the panel-derived class. Panel-local contradictions are also reported at
figure level. Explicit mixed panels resolve a global caption's simultaneous
affirmative plot/generic signals; genuine negations still require review.
A caption mentioning both a plot and a diagram without explicit
panels yields unknown, not an invented mixed layout.

Multiple asset occurrences without explicit panels, textual main/ratio or
upper/lower/left/right panel cues, and conflicting content without explicit panels
are flagged as candidates for future visual panel detection. No new panels,
bounding boxes, image regions or numerical records are created.

### Classified JSON schema 1.1

The classified run uses the same five filenames. Both record types retain their
original identifiers, captions, source regions, asset metadata, warnings, source
hashes and provenance. Inventory `status` continues to describe inventory success,
not classifier quality. Classifications extend the existing records with:

```json
{
  "schema_version": "1.1",
  "classification_method": "text_rules_v1",
  "classification_confidence": null,
  "classification_review_status": "pending",
  "classification_evidence": {
    "assigned_class": "plot",
    "plot_type": "histogram",
    "reason": "explicit_caption_evidence",
    "indicators": [
      {"rule": "histogram", "source": "caption", "match": "histogram",
       "class": "plot", "plot_type": "histogram", "polarity": "positive", "decisive": true}
    ],
    "contradictory_indicators": [],
    "visual_panel_detection_reasons": []
  }
}
```

The assigned class is also stored in `figure_type` or `panel_type`; panels have
their existing top-level `plot_type`. Figure aggregation adds `panel_classes`
(`panel_id`/class pairs) to evidence and sets its aggregate `plot_type` to null.
`visual_panel_detection_reasons` is a figure-level list. Classification metadata is
serialized only for schema 1.1, preserving the Task 01 JSON contract.

The summary includes figure and panel class counts, IDs of unknown/mixed and
contradictory cases, visual-detection candidates, review counts, method and schema.
All records require manual review. Original inventory metrics and paper failures
are retained under `inventory_summary`; they are not classification accuracy
metrics. Inherited diagnostics keep their content with schema 1.1 and
`origin_mode: inventory`. New classification diagnostics have `origin_mode: classify`
and codes `classification_unknown`, `classification_conflict`, or
`visual_panel_detection_candidate`. The new manifest hashes all five inventory
input files and records mode, schema, method and input directory; it does not
revalidate the original asset bytes.

### Limitations and future image analysis

These deterministic English text rules do not understand arbitrary natural
language, complex negation, references to other figures, or hidden graphical
composition. Matches may describe surrounding scientific context instead of the
actual image. Unknown cases are expected, and a textual match can be wrong. Local
negation is a heuristic, not a language parser. Filename hints, colour scales and
caption layout cues cannot establish a visual panel boundary or recover an axis.
Visual inspection is needed for unlabeled panels, main/ratio regions inside one
asset, ambiguous plot subtypes, and contradictions between caption and image.
Classification itself performs no numerical extraction. There is no VLM, image
segmentation or digitization backend integration.

## Task 03: original numerical sources

The one new module, `source_data.py`, separates numerical parsers, HEPData import,
bounded downloading and conservative comparisons from inventory/classification.
Its interfaces are `recover_native`, `parse_hepdata_table`, `recover_hepdata`,
`load_hepdata_mapping`, `fetch_hepdata` and `compare_sources`. `pipeline.py` handles
upstream validation and writing through the existing common manifest helpers.
`records.py` adds `PlotDataRecord` and `NUMERICAL_SCHEMA_VERSION = "1.2"`.

```bash
python scripts/extract_plots.py --mode recover \
  --classified-dir outputs/figures_fastcalo_classified_v1 \
  --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/plots_fastcalo_native_v1 \
  --source-method native
```

```python
from pathlib import Path
from kamiknows.kamiplotextractor import run_recovery

summary = run_recovery(
    classified_dir=Path("outputs/figures_fastcalo_classified_v1"),
    parsed_dir=Path("outputs/parsed_fastcalo"),
    output_dir=Path("outputs/plots_fastcalo_native_v1"),
    source_method="native",
)
```

Modes are `native` (default), `hepdata`, `auto` and `compare`. `auto` tries native
sources first and then explicitly mapped HEPData, retaining independent source
records. `compare` runs both and adds comparison results to the summary. Neither
mode merges datasets or treats publication identity as figure identity.

### Provenance checks and relocation

Recovery requires classified schema 1.1, its five files, and the original inventory
schema 1.0. The inventory is located through the classified manifest; pass
`--inventory-dir` if it has moved. Recovery verifies its five hashes against the
classification manifest and checks that IDs/source metadata survived classification
unchanged. It checks the parse manifest and `papers.jsonl` against the inventory
hashes, then verifies saved flat/sections/source-file hashes per paper.

A relocated parse run is accepted by content hashes and paper identities: original
absolute paths are not required. The historical inventory is still required because
it holds the original source hashes. Missing lineage or global identity mismatches
fail before output creation. Per-paper source mismatches produce unavailable
records and errors while other papers continue. Additional explicitly referenced
data files, which Task 01 did not hash, are hashed as read during recovery; this
records the current source snapshot rather than independently authenticating it.

### Native subset

The focused numerical scanner works on `latex.scan()`'s masked text and uses its
balanced argument parser and environment spans; existing scan results are unchanged.
Literal plot commands inside known figures establish the link. Subfigure ownership
uses the innermost explicit panel span. Plot/mixed figures are eligible; unknown
figures require explicit numerical structure. Generic figures and generic panels
are skipped unless `--include-generic` is supplied. Successful extraction never
changes figure/panel classification.

Supported examples (synthetic values):

```latex
\begin{axis}[xlabel={Energy},ylabel={Counts},x unit={GeV},xmode=log]
  \addplot coordinates {(1,2) (2,4e-3) (3,8)};
  \addlegendentry{Signal}
  \addplot table[x=energy,y=counts]{tables/data.dat};
  \addplot table[col sep=comma,x index=0,y index=2]{tables/data.csv};
\end{axis}
```

The subset includes `axis`, `loglogaxis`, `semilogxaxis`, `semilogyaxis`, literal
decimal/scientific coordinates, `\addplot+`, named/indexed table columns, inline
tables with newline or explicit `row sep=\\`, and external CSV/DAT/TXT files.
Comma-separated input requires `col sep=comma`; filename extensions do not change
PGFPlots' whitespace default. Tab and semicolon separators are also supported.
Header detection accepts a wholly textual header or wholly numeric first row;
mixed first rows are rejected unless header behavior is explicitly specified.

With no x/y selectors, the documented first/second columns are used and recorded
as `columns.default_first_two: true`. If any selector is supplied, both axes must
be explicitly and unambiguously selected. Empty/ragged tables, duplicate headers,
missing columns, macros and invalid numeric tokens produce diagnostics. No values
are guessed and no Poisson errors or bin edges are synthesized.

Multiple commands preserve their source order via `series_index` and share an
`axis_id`. Each command produces a separate numerical record, so a failed command
cannot hide a successful adjacent series. A single `\addlegendentry` between that
command and the next can name the series; an axis `legend entries` list is used
only when its length matches the number of commands. Uncertain legends stay null.
Axis labels and explicit `x unit`/`y unit` are retained literally. Units are not
inferred from physical conventions or parsed out of labels. Unspecified scales
remain null; explicit modes and log-axis environments supply scale metadata.
Non-positive values on log axes are rejected rather than silently discarding rows.

Supported table errors use `y error`/`y error index` or paired `y error plus` and
`y error minus` selectors (including index variants), with explicit PGFPlots
error-bar options such as `[error bars/.cd,y dir=both,y explicit]`. Relative explicit
errors retain their relative flag. Raw column names, signs and tokens are retained;
native lower/upper values are not converted into HEPData signed variations.
Statistical/systematic kinds stay null because table-column names alone do not
establish that meaning. Coordinate error syntax, x errors, error expressions and
unsupported error-bar variants are rejected with raw command/options preserved.

External files are resolved only from explicit plot references, relative to matched
original `.tex` locations and the allowed source roots. Ambiguous source directories
or multiple matching files cannot produce verified numerical data. Standalone
`.tex` plots and commands outside known figures remain candidates without promoted
vectors. An unrelated CSV beside an included image is never linked by filename.
Absolute paths, URLs, traversal and external symlink targets are rejected.

There is no TeX/gnuplot execution, expression evaluation, ROOT parsing, generic
data-file discovery, custom style expansion, 3D native extraction or pixel/PDF
geometry analysis. A conservative visual/metadata option allowlist rejects unknown
styles and transformations. `\pgfplotsset`/`\tikzset` declarations are accepted only
for compat/size settings; source conditionals also prevent promotion. Unsupported
filters/expressions never return untransformed source columns as plotted vectors.
Direct generic CSV references outside supported plot commands are not supported.

### Offline HEPData mapping and import

Use submission-format JSON or YAML with `independent_variables` and
`dependent_variables`, as documented by [HEPData](https://hepdata-submission.readthedocs.io/en/latest/data_yaml.html).
The website's separate visualization JSON format, whole-record archives and
multi-document submission manifests are not table inputs for this version.
No directory-wide YAML discovery is performed. Supply a mapping file, or place
`mapping.json` in `--hepdata-dir`. Relative source files are confined to that directory
(by default, the mapping file's parent). Replace all placeholders below with actual
identifiers; table numbering is never equated with figure numbering:

```json
{
  "schema_version": "1",
  "mappings": [{
    "arxiv_id": "<paper arXiv ID>",
    "figure_id": "<paper_id>::figure::0001",
    "record_id": "<numeric HEPData ID or ins-prefixed INSPIRE ID>",
    "table_name": "<exact published table name>",
    "source_file": "table.json"
  }]
}
```

`paper_id` may replace or accompany `arxiv_id`; both must agree if present. An arXiv
version suffix does not change publication identity. `latex_label` may replace or
accompany `figure_id`; duplicate labels remain ambiguous. Optional `panel_id` must
belong to the selected figure. A verified association requires a publication,
exactly one figure, record ID and table name. It means an auditable **explicit user
mapping**, not automatic expert validation of a caption or observable. Contradictory
identity metadata inside the table invalidates the mapping. Publication-only,
ambiguous or unmatched entries remain reviewable with states `candidate`,
`ambiguous` or `unmatched`; only `verified` associations emit published vectors.
No LLM or semantic matching is used. Duplicate identical mapping entries are rejected.

```bash
python scripts/extract_plots.py --mode recover \
  --classified-dir outputs/figures_fastcalo_classified_v1 \
  --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/plots_fastcalo_hepdata_v1 \
  --source-method hepdata --hepdata-dir /path/to/local/hepdata \
  --hepdata-map /path/to/local/hepdata/mapping.json
```

The importer preserves all independent/dependent columns, headers, units,
qualifiers, row order, central values, bin limits, multiple dependent series and
multiple uncertainty components. Symmetric and asymmetric signed errors, percentage
notation and labels survive in raw and normalized representations. Only explicit
`stat`/`statistical` or `sys`/`systematic` labels assign those kinds. Unlabeled errors
remain unlabeled. Missing, textual and unrepresentable values become null with a
state and the original token; infinite bin limits remain in raw tokens. No midpoint,
covariance matrix, uncertainty estimate, interpolation or unit conversion is inferred.

Multidimensional tables retain their full coordinate columns. They are not projected
to 1D: the `x` and `bins` convenience fields are null when more than one independent
variable exists. Published binned contents are not individual experimental events,
and published HEPData measurements are not detector-level raw event data.

### Optional network and dependencies

Networking is disabled by default. To download a **specified table**, omit
`source_file` in its mapping and pass `--enable-hepdata-network` with `hepdata`,
`auto` or `compare`. There is no online discovery/search or HTML scraping. The small
`requests` adapter uses the official client's documented record `format`/`table`
endpoint, requesting YAML. It permits only HTTPS on `www.hepdata.net`, at most three
redirects, three attempts, a 10-second connect timeout and 20-second read timeout.
Both declared and streamed sizes are bounded to 8 MiB. Local numerical/source/table
reads use the same size limit. Response filenames and archives are never extracted.
Downloaded text, final URL and SHA256 are retained in the record; no per-table files
are created. Backend failures remain separate from association/scientific quality.

Native and JSON support add no dependencies. YAML (including online downloads)
uses optional `PyYAML>=6`, installed separately with `pip install 'PyYAML>=6'`.
It uses a SafeLoader, retains floating-point tokens as strings before numeric
conversion, and rejects alias graphs and executable tags. PyYAML is
[MIT-licensed](https://github.com/yaml/pyyaml/blob/main/LICENSE).
[hepdata-cli](https://github.com/HEPData/hepdata-cli) was inspected but is not imported
or installed: it is GPL-3.0 and brings broader search/upload/archive handling that
is unnecessary here. The adapter uses the existing requests dependency and does
not copy client implementation code. Preserve any publisher/table-specific data
license metadata supplied in the raw source; software licenses do not license the
published measurements themselves.

### Numerical output schema 1.2 and comparisons

Recovery writes exactly `plot_data.jsonl`, `recovery_summary.json`,
`diagnostics.jsonl`, and `manifest.json` (`step: extract_plots`, `mode: recover`).
Old inventory/classification files remain unchanged. A `PlotDataRecord` contains:

- Existing `paper_id`, `figure_id`, optional `panel_id`, original classification
  snapshot, deterministic `data_id` and schema 1.2. Unassociated candidates may have
  null paper/figure IDs. IDs depend on associated identities and extraction-source
  locators, not output directory or mapping entry order.
- `source_method`: `original_source_data`, `original_published_data`,
  `candidate_data` or `unavailable`; `association` with state and explicit evidence;
  `extraction_status`: `success`, `partial` or `unavailable`.
- `backend`, adapter/decoder `backend_version`, `provenance.sources` with paths/URLs
  and SHA256. Native paths are source-root-relative except the explicitly marked
  parse-relative flat file. HEPData paths are mapping-root-relative. Network-client
  version is also recorded. The manifest locates and hashes all read local inputs.
- `axes`, `series`, `raw_source` and structured `warnings`. Native axes use `x`/`y`
  objects; published axes retain independent/dependent headers without inventing
  visual scales. Each series retains its order, identity, label, qualifiers, x/y
  convenience values, bins and uncertainty components in the source's semantics.
- Native `raw_values` preserve decimal tokens alongside numeric JSON values. HEPData
  `value_states`, raw cells, raw error components and the original table retain
  missing values, signs, percentages and precision. No unnecessary rounding is
  applied; numeric convenience values can still have normal floating-point limits.
- `validation_status: structural_checks_passed` only after supported parsing and
  shape checks, otherwise `not_validated`; `review_status: pending` always.

Candidate/ambiguous associations carry references/raw commands and warnings but no
promoted vectors. Partial published tables retain explicit nulls. Native parsing
fails a whole command if a required row/selector/transformation is unsupported,
while adjacent commands can succeed. Merely writing JSON is not scientific validation.

`--source-method compare` retains independent native and published records. Comparison
requires a unique nonempty series label, identical explicit axis names/units,
matching qualifiers, a single independent variable with point coordinates, no
published bin limits, and exact decimal coordinate equality. No series is paired
solely by figure ID. Compatible pairs report signed y differences and maximum
absolute difference using decimal arithmetic; incompatible pairs report
`not_comparable` and a reason. Uncertainty comparison is explicitly not performed.
These differences are not accuracy scores, agreement certificates or significance
tests. The summary separates extraction counts, association states, diagnostics,
per-paper source verification and comparisons. **Manual review pending.**

The native syntax is grounded in the [PGFPlots coordinate/table documentation](https://tikz.dev/pgfplots/reference-addplot).
The supported subset intentionally excludes the general transformation and TeX
capabilities described there.

## Verification and readiness

`pytest -q` includes offline fixtures for supported formats, nested LaTeX, explicit
panels, source-file origin matching, ambiguous/missing/unsafe references, unchanged
asset bytes, deterministic IDs, manifests, partial failures and CLI execution. No
real paper benchmark or scientific evaluation is implied. **Manual review pending.**

Classification tests add synthetic plot/generic/mixed/unknown cases, explicit-panel
aggregation, negation and contradiction handling, v1.0 compatibility, v1.1 output,
immutable inputs, standalone classification without the parse tree, and determinism.
They verify code behavior, not scientific accuracy on real figures.

Task 03 tests use exact synthetic coordinates and tables, including source linking,
negative/scientific values, errors, missing values, multidimensional published data,
path escapes, mocked network failures, relocation and source-hash mismatches.
No live network calls or real-paper numerical benchmarks are part of these tests.

Numerical records provide a provenance-preserving input boundary for Task 04,
subject to the supported subset and manual review. Task 04 has not been started;
there is no raster/vector digitization and no numerical RAG integration.
