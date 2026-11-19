# KamiKnows

KamiKnows is a traceable RAG (retrieval-augmented generation) pipeline for
scientific papers. It downloads the LaTeX sources of a set of arXiv
papers, splits them into chunks, retrieves the chunks relevant to each
question, and asks a local LLM (through Ollama) to answer **only from those
chunks, citing them**. Every run is saved with a manifest, so any answer can
be traced back to the passages, the chunks and the papers it came from.

Optional **plot extraction** also reads the numbers out of the papers'
figures (axis calibration, then digitization) and adds them to the chunks as
text with units, scales and provenance, so questions about plotted values can
be answered and cited by the same text-only models.

The included examples use five fast-calorimeter simulation papers (GAN, WGAN
and normalizing-flow generators), with one question set on paper text and one
on plot values.

**Slides:** [KamiKnows_pipeline.pdf](KamiKnows_pipeline.pdf) walks through the
repository, the commands, the output files and the review workflow.
[KamiKnows_plot_extraction.pdf](docs/plot_extractor/KamiKnows_plot_extraction.pdf)
explains plot extraction step by step, with the plot questions and how the
models answer them.

> Automatic checks validate formats and citation labels. They do not prove
> that an answer is scientifically correct: that is what the manual review
> step is for.

## Contents

1. [Repository layout](#repository-layout)
2. [Setup](#setup)
3. [The pipeline](#the-pipeline)
4. [Step 1: parse](#step-1-parse)
5. [Step 2: chunk](#step-2-chunk)
6. [Step 3: retrieve](#step-3-retrieve)
7. [Step 4: generate](#step-4-generate)
8. [Step 5: review](#step-5-review)
9. [Interactive chat](#interactive-chat)
10. [Running on SLURM](#running-on-slurm)
11. [Run directories and naming](#run-directories-and-naming)
12. [Outputs index](#outputs-index)
13. [Questions files](#questions-files)
14. [Tests](#tests)
15. [Models and licenses](#models-and-licenses)
16. [Plot extraction](#plot-extraction)
17. [Compute](#compute)
18. [Rebuild the slides](#rebuild-the-slides)

## Repository layout

```text
kamiknows/            library, one module per step
  parsing.py          arXiv download + LaTeX parsing
  chunking.py         word / sentence / semantic chunkers
  retrieval.py        TF-IDF and dense retrieval, source constraints, hit@k / MRR
  generation.py       prompts, Ollama client, citation validation
  review.py           manual review sheet
  embeddings.py       sentence-transformers encoder (shared by chunking and retrieval)
  common.py           JSON I/O and run manifests (shared by all steps)
  plot_extractor/     plot extraction: inventory, classification, calibration, digitization
  compute/            deterministic numeric operations on extracted data, plan executor
scripts/              one command per step (thin wrappers around kamiknows/)
  parse.py  chunk.py  retrieve.py  generate.py  review.py
  chat.py             interactive terminal chat
  extract_plots.py    plot extraction modes (inventory, classify, calibrate, digitize, ...)
  compute.py          runs the hand-written plans of a questions file on extracted data
  index_outputs.py    writes outputs/INDEX.md, the local list of runs
  run_ollama.slurm    runs any command on a GPU node with its own Ollama server
tests/                offline tests (no network, no Ollama, no models)
data/input/           arXiv id lists (corpora)
data/questions/       questions files: questions + expected answers, one file per evaluation set
docs/plot_extractor/  plot extraction guide, QA results and slides
docs/slides/          source of both slide decks (build_slides.py, fonts, images)
outputs/              run directories (not versioned)
KamiKnows_pipeline.pdf  slides: visual version of this README
```

Run every command from the repository root.

## Setup

```bash
conda create -n kamiknows python=3.11
conda activate kamiknows
pip install -r requirements.txt        # on GPU nodes, install the matching torch build first
```

**Embedding model.** Models are loaded from the local Hugging Face cache only
(nothing is downloaded during a run). Download MiniLM once:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

**Ollama.** Generation uses a local Ollama server. On the cluster a shared
installation lives in `/storage-hpc/kamiknows` (binary in
`software/ollama/bin`, models in `models/ollama`) and `scripts/run_ollama.slurm`
starts it for you. Elsewhere, install Ollama, `ollama pull qwen3:4b-instruct`
and `ollama serve`.

## The pipeline

```text
arXiv ids ──parse──> sections + equations ──────────────┐
                  │                                     ├──chunk──> text + plot chunks
                  └──> figure files                     │                  │
                         │  (optional plot extraction)  │               retrieve
                      inventory                         │                  │
                         ↓                              │          passages per question
                      classify                          │                  │
                         ↓                              │               generate
                      calibrate                         │                  │
                         ↓                              │       answers + citation checks
                      digitize ──> values + units ──────┘                  │
                         └──> visual review                             review
                                                                           ↓
                                                                  review.csv + review.html
```

Each step reads the directory written by the previous step and writes a new
directory. Choices are command-line options, so changing one thing (the
chunker, the retriever, the model, the prompt) means re-running only the steps
from that point on.

| Step | Command | Input | Main output |
|---|---|---|---|
| 1 | `scripts/parse.py` | arXiv id list | `papers.jsonl`, `papers/<id>/sections.json`, `equations.json` |
| 2 | `scripts/chunk.py` | parse run + optional digitization run | `chunks.jsonl`, `equations.jsonl` |
| 3 | `scripts/retrieve.py` | chunk run + questions | `contexts.jsonl` (top-k passages per question) |
| 4 | `scripts/generate.py` | retrieve run | `answers.jsonl` |
| 5 | `scripts/review.py` | one or more generate runs | `review.csv`, `review.html` |

Plot extraction is optional and runs between parsing and chunking, as
subcommands of one CLI, `scripts/extract_plots.py`:
`inventory → classify → calibrate → digitize`. Its output joins step 2 through
`--digitized-dir`; without that option the pipeline is text-only. See
[Plot extraction](#plot-extraction).

Every run directory also contains `manifest.json` (see
[Run directories](#run-directories-and-naming)). Each script prints the run
summary at the end, and `--help` lists all its options.

## Step 1: parse

Downloads metadata and the LaTeX source (e-print) of each paper, inlines
`\input`/`\include`, and extracts sections (`\section`, `\subsection`,
`\subsubsection`) as plain text plus display equations. Citations become
`[REF]`, equations become `[EQUATION_nnn]` placeholders linked to
`equations.json`. PDF-only papers are recorded as `failed`.

```bash
python scripts/parse.py \
  --ids-file data/input/arxiv_ids_fastCalo_5fulltext.txt \
  --output-dir outputs/parsed_fastcalo
```

The id file has one arXiv id per line (`1705.02355v2`, `arXiv:...` or an abs
URL); `#` starts a comment. Output per paper: `paper.json` (metadata, status
`success|partial|failed`, warnings, errors), `sections.json`,
`equations.json`, `flat.tex`, and the downloaded `source/`.

Plot extraction reuses this same `flat.tex` and source tree: parsing does not
need to be repeated to add it.

## Step 2: chunk

Splits each section into chunks. Chunks never cross sections and sizes are
counted in whitespace-separated words. The chunker is a sub-command:

| Chunker | How it cuts |
|---|---|
| `word` | fixed windows of `--target-words`, overlapping by `--overlap-words`; a final tail under 80 words joins the previous window |
| `sentence` | packs whole sentences up to `--target-words`; overlap is made of whole sentences |
| `semantic` | like `sentence`, but cuts where the embedding similarity between neighbouring sentences drops (topic change) |

```bash
python scripts/chunk.py word     --parsed-dir outputs/parsed_fastcalo --output-dir outputs/chunks_fastcalo_word
python scripts/chunk.py sentence --parsed-dir outputs/parsed_fastcalo --output-dir outputs/chunks_fastcalo_sentence

python scripts/chunk.py semantic --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/chunks_fastcalo_semantic_fd100_nobm \
  --exclude-backmatter --cut-policy first_drop --min-words 100 --plots
```

Options shared by all chunkers: `--target-words` (default 400),
`--overlap-words` (60), and `--exclude-backmatter`, which drops
acknowledgment/bibliography sections (logged in `section_exclusions.jsonl`).

**Plot chunks.** `--digitized-dir <digitization run>` adds one chunk per
plotted image (a plot, or one panel of a multi-panel figure) to any chunker:
one plot, one chunk, one digitized record. The chunk describes the plot so
that retrieval can find it: caption, the paper sentences that cite it, panel
position and figure layout, axes with units and scales, and per series a few
characteristic numbers computed by `kamiknows.compute` (ranges, maximum, tied
maxima). It never holds the extracted values. `plot_link` names the digitized
record and the compute handle of each series, and for a multi-panel figure
lists every panel (`figure_panels`), so a retrieved chunk leads to the data
that `scripts/compute.py` and the solver work on. Plots that could not be
digitized get a chunk too, saying why. Panel positions are shown but left out
of the ranking text. The text chunks are unchanged.

**How the semantic chunker decides.** Every sentence is embedded
(`--embedding-model`, default `all-MiniLM-L6-v2`). At each sentence boundary
it computes the cosine similarity between the mean embedding of the
`--window-sentences` (2) sentences before and after. A boundary is a
*candidate* cut if its similarity is in the lowest `--threshold-quantile`
(0.25) of its section. Among candidates that keep the chunk between
`--min-words` (250) and `--target-words`:

- `--cut-policy lowest_similarity` (default) cuts at the lowest candidate;
- `--cut-policy first_drop` cuts at the first candidate whose similarity fell
  by at least `--min-score-drop` (0.10) from the previous boundary.

If no candidate fits, the chunk ends at the size limit or at the section end.

Each chunk in `chunks.jsonl` has `chunk_id` (`<paper_id>::<chunker>::0001`),
paper ids and title, `section_id`/`section_heading`, `text`, `words`,
`cut_reason` (`semantic_drop`, `length_limit` or `section_end`; `plot` for plot
chunks),
`equation_ids` (equations of its section, detailed in `equations.jsonl`),
`sentence_span` for sentence-based chunkers, and `flags`
(`oversize_sentence`, `below_min_words`, `above_max_words`,
`duplicate_text`). The semantic chunker also writes every boundary score to
`semantic_boundaries.jsonl` and, with `--plots`, PNG/PDF figures per section
(similarity, similarity decrease, threshold, cuts, chunk spans) and an
overview comparing word/sentence/semantic chunk lengths.

## Step 3: retrieve

Ranks the chunks for every question of an evaluation set and keeps the top
`--top-k` as numbered passages `C1, C2, ...` with their full text. The query
is the question text only; the annotations of the eval set (expected papers,
expected behaviour) are used to score retrieval, never to rank.

```bash
python scripts/retrieve.py \
  --dataset-dir outputs/chunks_fastcalo_semantic_fd100_nobm \
  --questions data/questions/fastcalo_paired.jsonl \
  --output-dir outputs/retrieval_semantic_fd100_nobm_dense_explicit_paired_top5 \
  --retriever dense --top-k 5
```

Dense retrieval needs the MiniLM model in the local cache (see [Setup](#setup)).

| Option | Values |
|---|---|
| `--retriever` | `tfidf`: lexical cosine, title and section heading weighted ×2; `dense`: cosine of sentence-transformers embeddings |
| `--embedding-model` | `all-MiniLM-L6-v2` (default) or `e5-small-v2` (dense only) |
| `--source-policy` | `explicit_arxiv` (default): when a question names arXiv ids, only chunks of those papers are ranked and the best chunk of each named paper is always kept; a named paper missing from the corpus gives no passages. `none`: always the whole corpus |
| `--top-k` | passages per question (default 5) |
| `--max-plots-per-figure` | at most this many plot chunks of one figure among the passages (default 1) |

`contexts.jsonl` holds, per question, the passages, the source-constraint
status and the evaluation (`hit`, `hit_rank`, `reciprocal_rank`). The
manifest summary reports hit@k, MRR and the missed questions.

Plot chunks are ranked together with paper passages; `contexts.jsonl` keeps
their `plot_link`. With `expected_plot_sources` annotations, the summary also
reports whether the required plot series was retrieved, directly or through
the `figure_panels` of a retrieved panel of the same figure: finding the right
paper is not enough.

The panels of a multi-panel figure share caption and discussion, so their
plot chunks score almost alike and can fill every slot, pushing out the figure
a question is about. `--max-plots-per-figure 1` (the default) keeps only the
best-ranked plot chunk of each figure; the others stay reachable through its
`figure_panels`. On the fastcalo plot questions, with a plot chunk for every
image, this raises the questions whose required series is retrieved from 15 to
19 of 20.

## Step 4: generate

Sends each question with its saved passages to an Ollama model and asks for
JSON: `answer_status` (`answered` or `insufficient_evidence`), `answer`,
`evidence`, `citations` (`["C1", ...]`), `confidence`, `limitations`.
Retrieval is not re-run: the model sees exactly the passages of step 3.
The prompt gives the instructions and the question, then the passages, then
the question again: with long passages, a question stated only at the top was
often ignored.

Plot chunks reach the model as ordinary text passages, so no multimodal model
is needed. They describe the plot and give a few characteristic numbers
(ranges, maxima), not the extracted values: a 4B or 7B model cannot look
values up reliably in long tables. Numbers beyond the card come from the
solver.

**Solver (`--solver`).** For a question whose passages include plot chunks,
the model first acts as a planner. It sees the plot passages with the compute
handle of each series (and of every panel of the same figure), the axis names
and units, and the catalogue of operations of `kamiknows/compute/registry.py`,
and writes a JSON plan (see [Compute](#compute)), or declines when the plot
cannot give the number (original counts, bin edges, uncertainties). The plan
runs on the digitized data linked to the retrieved chunks, never on other
plots. A plan that is malformed or that an operation refuses goes back to the
planner with the exact error (with the operation signature), up to 3 plans;
a correction must be a plan, not a refusal. The results join the passages as
one more labelled passage of type `computed`, with the expression that
produced each number written with readable series names; the model then
writes the answer and cites it. The digitized data is found through the
manifests (retrieval run, then chunk run, then its `--digitized-dir`), and
the hashes in `plot_link` must match.

```bash
python scripts/generate.py \
  --retrieval-dir outputs/retrieval_semantic_fd100_nobm_dense_explicit_paired_top5 \
  --output-dir outputs/qa_semantic_fd100_nobm_dense_explicit_qwen4b_paired_top5 \
  --model qwen3:4b-instruct --base-url http://127.0.0.1:11434
```

| Option | Meaning |
|---|---|
| `--model` | Ollama tag; the run stops immediately if it is not installed |
| `--prompt` | `grounded` (default): answer only if supported, otherwise abstain; `synthesis`: may combine passages and give clearly marked partial answers |
| `--question-ids` | answer only these questions (e.g. re-ask some questions with another prompt) |
| `--num-predict` | max answer tokens (default 1024; 512 truncated some answers) |
| `--num-ctx` | model context in tokens (default 32768), fixed so long contexts are never cut by a server default |
| `--solver` | plan and compute plot numbers with `kamiknows.compute` before answering (needs a retrieval over chunks built with `--digitized-dir`) |
| `--temperature`, `--top-p`, `--timeout-seconds`, `--base-url` | generation and server settings |

Answers are written one by one, so an interrupted job keeps what it finished.
Each answer carries a **citation check**: `valid`, `invented_citation`
(a label that is not among the passages) or `answered_without_citation`;
plus warnings such as `json_parse_error`, `truncated_at_token_limit` or
`prompt_truncated_at_context_limit` (the prompt is longer than `--num-ctx` and
Ollama evaluated too few of its tokens), and
the generation token counts. A failed request is recorded as
`backend_error` and the run continues. The manifest summary counts answer
statuses, also split by the expected behaviour of the question. With
`--solver`, each answer also carries `solver` (status, every plan attempt
with its outcome) and `computed_passage`, and the summary counts
`solver_status` (`computed`, `declined`, `not_computable`, `invalid_plan`,
`no_computation`, `not_applicable`) and `solver_plan_attempts`.

## Step 5: review

Builds one CSV for human review from one or more generation runs (several runs
make side-by-side comparisons easy).

```bash
python scripts/review.py prepare \
  --qa-dirs outputs/qa_semantic_fd100_nobm_dense_explicit_qwen4b_paired_top5 \
  --output-dir outputs/review_qwen4b_paired
```

Each row has the question, expected behaviour, reference answer, required
points and reference evidence (all from the questions file), the model
answer, the citation check, and the text of the cited passages. Fill in
`reviewer_correctness`, `reviewer_citation_support`, `reviewer_abstention`
(allowed values are listed in the review manifest), `reviewer_notes`, and set
`review_status` to anything other than `pending`. Then:

```bash
python scripts/review.py summarize --csv outputs/review_qwen4b_paired/review.csv
```

prints the label counts per run and any value outside the allowed labels.

## Interactive chat

Asks questions from the terminal against a chunk run. Each question is
independent (no conversation memory); every turn is saved in
`outputs/chat/<session>/turns.jsonl`. The chat needs an Ollama server on a GPU
node. Run the two blocks separately.

On the login node, open an interactive GPU session and wait for its prompt:

```bash
srun --account=kamiknows --qos=kamiknows --partition=bare-metal-GPU --gres=gpu:1 --pty bash
```

Then, inside the GPU session (`hostname` must show a GPU node, not the login node):

```bash
conda activate kamiknows
export OLLAMA_MODELS=/storage-hpc/kamiknows/models/ollama OLLAMA_HOST=127.0.0.1:11434
/storage-hpc/kamiknows/software/ollama/bin/ollama serve > /tmp/ollama-$USER.log 2>&1 &
until curl -sf http://127.0.0.1:11434/api/tags > /dev/null; do sleep 2; done
python scripts/chat.py --dataset-dir outputs/chunks_fastcalo_semantic_fd100_nobm \
  --retriever dense --model qwen3:4b-instruct --base-url http://127.0.0.1:11434
```

Type `exit` when done: the session keeps its GPU until you leave it. Open
`srun` from a fresh login shell; inside an existing allocation it tries to add
a step to that job instead of starting a new session. If port 11434 is taken,
use another one in both `OLLAMA_HOST` and `--base-url`. The chat prints the
first 240 characters of each passage; `--show-passages` prints them in full
(the model always receives the full text).

## Running on SLURM

Generation needs a GPU and an Ollama server. `scripts/run_ollama.slurm` runs
any command on a GPU node with a private Ollama server:

```bash
sbatch --job-name=KK-qa-qwen4b scripts/run_ollama.slurm \
  scripts/generate.py \
  --retrieval-dir outputs/retrieval_semantic_fd100_nobm_dense_explicit_paired_top5 \
  --output-dir outputs/qa_semantic_fd100_nobm_dense_explicit_qwen4b_paired_top5 \
  --model qwen3:4b-instruct
```

The script:
1. starts Ollama from `/storage-hpc/kamiknows` on a port derived from the job id;
2. waits until it answers;
3. runs the command with `--base-url` appended (do not pass it yourself);
4. copies `slurm.log` and `ollama.log` into the command's `--output-dir`.

The live log is `outputs/_slurm_logs/<job-name>-<job-id>.log`. The defaults
are 1 GPU, 4 CPUs, 16 GB and 1 hour; override them on the command line
(`sbatch --time=02:00:00 ...`). Activate the Python environment before submission
or export `PYTHON`. `KK_ROOT`, `OLLAMA_BIN`, `OLLAMA_MODELS` and `KK_LOG_DIR`
configure installation paths and logs. Scheduler bootstrap output is
`slurm-<job-id>.out` in the submission checkout (ignored by git).

Comparing several models or prompts means several `sbatch` calls, one output
directory each. Parse, chunk and retrieve run fine on the login node (dense
retrieval and semantic chunking use the cached local encoder; runtime depends
on CPU allocation and passage lengths).

## Run directories and naming

Rules:

- `--output-dir` must be a **new** directory; scripts refuse to overwrite.
  To redo a run, choose a new name (e.g. add `_retry1`).
- Every run directory contains `manifest.json` with the step name, the full
  command, the git commit (and whether the tree had uncommitted changes),
  all parameters, every input file with its sha256, and the summary metrics.
  Each step checks that its input directory comes from the expected step.
- `outputs/` is not versioned, and versioned files never refer to its run
  directories. `outputs_backup/` holds the runs made with the pre-refactor
  code; it is kept for reference only.

Directory names are chosen by hand. Use this pattern so names stay readable
and sortable (the manifest is always the authoritative record):

```text
parsed_<corpus>[_<variant>]
chunks_<corpus>_<chunker>[_<variant>]
retrieval_<chunker>_<retriever>_<sources>_<evalset>_top<k>[_<variant>]
qa_<chunker>_<retriever>_<sources>_<model>_<evalset>_top<k>[_<variant>]
review_<what is reviewed>
figures_<corpus>_inventory[_<variant>]
figures_<corpus>_classified_<classifier>[_<variant>]
calibrations_<corpus>_<variant>
plots_<corpus>_<variant>
compute_<evalset>_<plans>[_<variant>]
```

| Field | Values |
|---|---|
| corpus | `fastcalo` |
| chunker | `word`, `sentence`, `semantic`; suffixes: `nobm` = `--exclude-backmatter`, `fd100` = `--cut-policy first_drop --min-words 100`, `plotcards` = `--digitized-dir` |
| retriever | `tfidf`, `dense` (add the model if not MiniLM, e.g. `dense-e5`) |
| sources | `explicit` = `--source-policy explicit_arxiv`, `allsrc` = `none` |
| model | `qwen06b`, `qwen4b`, `mistral7b`, ... |
| evalset | short tag of the questions file: `paired` = `fastcalo_paired.jsonl`, `plots` = `fastcalo_plots.jsonl`, `plotvalues` = `fastcalo_plots_values.jsonl`, `plotplans` = `fastcalo_plots_values_plans.jsonl` |
| plans | where the executed plans come from: `expected` = the `expected_plan` annotations |
| classifier | `vision` (Gemma) or `text` |
| variant | the one thing that differs from the baseline, e.g. `synthesis`, `np512`; `full_auto` = automatic calibration or digitization of every image of the corpus (no figure filter), `fullauto` = chunks or runs built on that digitization, `onefig` = `--max-plots-per-figure 1` (named so while it was not yet the default) |

Retries append `_retry<N>`.

## Outputs index

`outputs/INDEX.md` lists the local runs, one table per step with date, key
parameters and metrics, generated from the manifests. Like the runs, it is not
versioned. Free-text notes (what changed compared with which baseline, why a
run failed) go in `outputs/index_notes.json`:

```json
{"qa_semantic_fd100_nobm_dense_explicit_qwen4b_paired_top5": {"note": "Baseline after the refactor."}}
```

After each run, add a note and regenerate:

```bash
python scripts/index_outputs.py
```

## Questions files

All the questions of an evaluation, with their expected answers, are in
**one JSONL file** in `data/questions/`, one question per line. It is passed
to `retrieve.py --questions`; generation and review read the questions from
the retrieval run, so nothing else needs to change.

**Never edit a questions file that has already been used.** Runs record its
sha256 and keep its name in their directory name, so editing it in place would
make earlier runs inconsistent with the file. To try new or changed questions,
copy the file to a new descriptive name (e.g. `fastcalo_paired_hard.jsonl`), edit the copy,
use a new `evalset` tag in the run names, and re-run from step 3. Only compare
runs that used the same questions file.

Fields (only `question_id` and `question` are required):

| Field | Used by |
|---|---|
| `question_id` | unique id, joins questions, passages, answers and review rows |
| `question` | the query for retrieval and the question in the prompt |
| `expected_behavior` | `answerable` or `insufficient_evidence`; answer statuses are split by it |
| `expected_arxiv_ids` | retrieval hit@k / MRR (a hit means a passage from an expected paper) |
| `required_all_expected_sources` | if true, a hit needs every expected paper (comparison questions) |
| `reference_answer` | shown to the reviewer next to the model answer |
| `required_answer_points` | list of points a correct answer must contain, shown to the reviewer |
| `reference_evidence` | source passages (or figure assets and series) supporting the reference answer |
| `expected_plot_sources` | plot asset/series pairs that retrieval must find, scored separately from papers |
| `expected_plan`, `expected_results` | a hand-written compute plan and the results it must give (see [Compute](#compute)) |
| `pair_id`, `source_variant`, `question_style`, `question_type` | grouping for analysis and review |

These fields are used for evaluation and review only: they never reach the
ranking or the prompt. Extra fields are allowed and carried along.

`data/questions/fastcalo_paired.jsonl` (run tag `paired`) has 10 pairs on paper
text: each question appears once naming its arXiv id (`explicit`) and once
without (`implicit`). 16 are answerable from the corpus and 4 should be refused.

`data/questions/fastcalo_plots.jsonl` (run tag `plots`) has 6 pairs on plot
values: each question appears once citing the figure (`figure_reference`) and
once as a natural question with no figure or arXiv id (`natural`). 10 are
answerable from the extracted values; 2 ask for original event counts and
should be refused.

`data/questions/fastcalo_plots_values.jsonl` (run tag `plotvalues`) adds 4 pairs
asking for values other than peaks: a given bin, the first bin where one
histogram overtakes the other, a local maximum in a range, a given heatmap
cell. `data/questions/fastcalo_plots_values_plans.jsonl` (run tag `plotplans`)
holds the same 20 questions with an `expected_plan` and `expected_results`
each, taken from the reference answers.

| `question_type` | File | Count | Example topic |
|---|---|---:|---|
| `fact_lookup` | paired | 2 | simulator name and model family |
| `method` | paired | 6 | architecture, image representation, Flow I/II |
| `source_lookup` | paired | 2 | reported speed-up and evaluation time |
| `comparison` | paired | 6 | approach, likelihood or representations across papers |
| `no_answer` | paired | 4 | unsupported deployment dates or procedures |
| `plot_data` | plots | 12 | peaks, ratios, axis scales and units, unavailable counts |

The reference answers are drafts, not yet validated by an expert. Current
results on the plot questions are in [Plot QA results](docs/plot_extractor/qa_results.md).

## Tests

```bash
pytest -q
```

The tests are offline: a synthetic two-paper corpus, a fake encoder and a
fake Ollama cover every step from parsing to the review summary.

## Models and licenses

| Model | Role | License |
|---|---|---|
| `qwen3:4b-instruct`, `qwen3:0.6b` (Qwen) | answer generation | Apache 2.0 |
| `mistral:7b` (Mistral AI) | answer generation | Apache 2.0 |
| `sentence-transformers/all-MiniLM-L6-v2` | semantic chunking, dense retrieval | Apache 2.0 |
| `intfloat/e5-small-v2` (optional) | dense retrieval | MIT |

Check the license of any other model before adding it.

## Plot extraction

`scripts/extract_plots.py` turns the figures of an existing parse run into
numerical evidence. Retrieval, generation and review do not change.

| Mode | What it does | Main output |
|---|---|---|
| `inventory` | finds figure environments, captions, labels and figure files | `figures.jsonl` |
| `classify` | says what each file is (plot type, axes, panels); `--classifier vision` (Gemma, GPU) or `text` | `asset_predictions.jsonl` |
| `calibrate` | matches tick labels to tick marks and fits linear/log10 axes, units, colour bars; OCR fallback | `calibration_proposals.json`, `review.html` |
| `digitize` | reads vector paths or colour cells through the calibrated axes | `digitized_data.jsonl` |
| `digitization-review` | redraws the extracted values next to each source figure | `review.html`, `report.pdf` |
| `calibration-review` | re-renders the HTML report of a calibration run | `review.html` |
| `recover`, `benchmark` | optional: original data (PGFPlots, HEPData) and comparison with reference values | |

```bash
PARSED=outputs/parsed_fastcalo
INVENTORY=outputs/figures_fastcalo_inventory_v1
CLASSIFIED=outputs/figures_fastcalo_classified_vision_v1
CALIBRATED=outputs/calibrations_fastcalo_auto_v1
DIGITIZED=outputs/plots_fastcalo_auto_v1

python scripts/extract_plots.py inventory --parsed-dir $PARSED --output-dir $INVENTORY
sbatch scripts/run_ollama.slurm scripts/extract_plots.py classify \
  --classifier vision --model gemma3:4b --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CLASSIFIED
# when the classification job has finished:
python scripts/extract_plots.py calibrate \
  --classified-dir $CLASSIFIED --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CALIBRATED
python scripts/extract_plots.py digitize --classified-dir $CLASSIFIED --inventory-dir $INVENTORY --parsed-dir $PARSED \
  --calibration-file $CALIBRATED/calibration_proposals.json --output-dir $DIGITIZED
python scripts/extract_plots.py digitization-review --digitized-dir $DIGITIZED \
  --output-dir outputs/plots_fastcalo_visual_v1
python scripts/chunk.py sentence --parsed-dir $PARSED --exclude-backmatter \
  --digitized-dir $DIGITIZED --output-dir outputs/chunks_fastcalo_sentence_plots_v1
```

Then run steps 3-5 on the new chunk directory, for example with
`--questions data/questions/fastcalo_plots.jsonl`. Classification needs a GPU;
the other modes run on CPU. All dependencies, including the OCR engine, are in
`requirements.txt`; Gemma is in the shared Ollama store.

Extracted values are estimates of what the figure displays. Calibration and
digitization report a reason instead of guessing when axes are ambiguous or a
layout is unsupported; original bins, counts and uncertainties are never
inferred. On the fastcalo corpus one automatic calibration and one digitization
of all 224 images give 119 digitized images (322 series: 2D maps, step
histograms and curves); the other 105 report why they were not digitized.
Manual review of the digitized values is pending.

- [Plot extraction guide](docs/plot_extractor/guide.md): how each step works, outputs, limits, code.
- [Slides](docs/plot_extractor/KamiKnows_plot_extraction.pdf): the same chain step by step, with examples.
- [Plot QA results](docs/plot_extractor/qa_results.md): the plot questions and how Qwen and Mistral answer them.

## Compute

`kamiknows/compute/` turns extracted data into numbers with deterministic
operations, so that a language model never has to read raw values. Every
source becomes a `Series`: a discrete function from a 1D or 2D domain to one
value axis, with units and scales (a curve, a histogram, the cells of a
heatmap). Digitized plots are read by `compute/sources/plots.py`, the only
module that knows the digitization schema.

The primitives take and return series, points and quantities, so any
composition of them is again an operation:

| Family | Operations |
|---|---|
| selection | `restrict`, `section`, `coordinate` |
| evaluation | `evaluate`, `solve` |
| algebra | `combine`, `transform`, `component` |
| reduction | `reduce`, `project`, `argextremum` |
| calculus | `differentiate`, `cumulate` |

`derived.py` writes common quantities (`crossing`, `normalize`, `mean`, `std`,
`quantile`, `fwhm`, `linear_fit`, `distance`) only as compositions of the
primitives. Units propagate and are checked; anything the data cannot give
(a point outside the plot, an undigitized figure, incompatible units) raises
`NotComputable` with the reason instead of an approximation. Every number
carries the expression that produced it, e.g. `div(max(A), max(B))`.

A plan is a JSON list of steps over the operations in `registry.py`; later
steps refer to earlier ones (`$a`, `$a[0].x`):

```json
{"steps": [{"id": "blue", "op": "reduce", "args": {"series": "<asset_id>/path_2", "how": "max"}},
           {"id": "red", "op": "reduce", "args": {"series": "<asset_id>/path_3", "how": "max"}},
           {"id": "ratio", "op": "combine", "args": {"left": "$blue", "right": "$red", "op": "div"}}],
 "result": ["$ratio"]}
```

`scripts/compute.py` runs the `expected_plan` of every question and compares
the results with `expected_results` within `--rel-tol` (default 1%). No
language model runs, so this checks the operations and the extracted data
alone:

```bash
python scripts/compute.py --questions data/questions/fastcalo_plots_values_plans.jsonl \
  --digitized-dir $DIGITIZED --output-dir outputs/compute_plotplans_expected
```

Output: `results.jsonl` (per question: plan, outcome, expected results,
differences) and the manifest. An outcome is `computed`, `declined` (the plan
says the data cannot answer), `not_computable` (an operation refused, with the
reason) or `invalid_plan` (a malformed plan).

## Rebuild the slides

```bash
python docs/slides/build_slides.py
```

Rebuilds `KamiKnows_pipeline.pdf` and
`docs/plot_extractor/KamiKnows_plot_extraction.pdf`. The slide content is in
`docs/slides/build_slides.py`, the layout helpers in `docs/slides/deck.py`;
fonts (SIL Open Font License / DejaVu license) and images are in the same folder.
