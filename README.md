# KamiKnows

KamiKnows is a traceable RAG (retrieval-augmented generation) pipeline for
scientific papers. It downloads the LaTeX sources of a set of arXiv
papers, splits them into chunks, retrieves the chunks relevant to each
question, and asks a local LLM (through Ollama) to answer **only from those
chunks, citing them**. Every run is saved with a manifest, so any answer can
be traced back to the passages, the chunks and the papers it came from.

The current corpus is five fast calorimeter simulation papers (FastCaloSim:
GAN, WGAN and normalizing-flow shower generators) with a 20-question
evaluation set.

**Slides:** [KamiKnows_pipeline.pdf](KamiKnows_pipeline.pdf) walks through the
same content visually: repository layout, the five steps with their commands,
SLURM, run naming and a quick start.

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
16. [KamiPlotExtractor: figure inventory](#kamiplotextractor-figure-inventory)

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
  kamiplotextractor/   independent LaTeX figure inventory (Task 01)
scripts/              one command per step (thin wrappers around kamiknows/)
  parse.py  chunk.py  retrieve.py  generate.py  review.py
  chat.py             interactive terminal chat
  index_outputs.py    regenerates docs/outputs_index.md
  extract_plots.py    inventories figures from an existing parse run
  run_ollama.slurm    runs any command on a GPU node with its own Ollama server
tests/                offline tests (no network, no Ollama, no models)
data/input/           arXiv id lists (corpora)
data/questions/       questions files: questions + expected answers, one file per evaluation set
docs/outputs_index.md versioned list of all runs in outputs/
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
arXiv ids ──parse──> sections + equations ──chunk──> chunks ──retrieve──> passages per question
                                                                              │
                         review.csv <──review── answers + citation check <──generate
```

Each step reads the directory written by the previous step and writes a new
directory. Choices are command-line options, so changing one thing (the
chunker, the retriever, the model, the prompt) means re-running only the steps
from that point on.

| Step | Command | Input | Main output |
|---|---|---|---|
| 1 | `scripts/parse.py` | arXiv id list | `papers.jsonl`, `papers/<id>/sections.json`, `equations.json` |
| 2 | `scripts/chunk.py` | parse run | `chunks.jsonl`, `equations.jsonl` |
| 3 | `scripts/retrieve.py` | chunk run + questions | `contexts.jsonl` (top-k passages per question) |
| 4 | `scripts/generate.py` | retrieve run | `answers.jsonl` |
| 5 | `scripts/review.py` | one or more generate runs | `review.csv` |

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
`cut_reason` (`semantic_drop`, `length_limit` or `section_end`),
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
Probably you will need to download the model with
```
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

| Option | Values |
|---|---|
| `--retriever` | `tfidf`: lexical cosine, title and section heading weighted ×2; `dense`: cosine of sentence-transformers embeddings |
| `--embedding-model` | `all-MiniLM-L6-v2` (default) or `e5-small-v2` (dense only) |
| `--source-policy` | `explicit_arxiv` (default): when a question names arXiv ids, only chunks of those papers are ranked and the best chunk of each named paper is always kept; a named paper missing from the corpus gives no passages. `none`: always the whole corpus |
| `--top-k` | passages per question (default 5) |

`contexts.jsonl` holds, per question, the passages, the source-constraint
status and the evaluation (`hit`, `hit_rank`, `reciprocal_rank`). The
manifest summary reports hit@k, MRR and the missed questions.

## Step 4: generate

Sends each question with its saved passages to an Ollama model and asks for
JSON: `answer_status` (`answered` or `insufficient_evidence`), `answer`,
`evidence`, `citations` (`["C1", ...]`), `confidence`, `limitations`.
Retrieval is not re-run: the model sees exactly the passages of step 3.

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
| `--temperature`, `--top-p`, `--timeout-seconds`, `--base-url` | generation and server settings |

Answers are written one by one, so an interrupted job keeps what it finished.
Each answer carries a **citation check**: `valid`, `invented_citation`
(a label that is not among the passages) or `answered_without_citation`;
plus warnings such as `json_parse_error` or `truncated_at_token_limit`, and
the generation token counts. A failed request is recorded as
`backend_error` and the run continues. The manifest summary counts answer
statuses, also split by the expected behaviour of the question.

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
`outputs/chat/<session>/turns.jsonl`.

```bash
python scripts/chat.py --dataset-dir outputs/chunks_fastcalo_semantic_fd100_nobm \
  --retriever dense --model qwen3:4b-instruct --base-url http://127.0.0.1:11434
```


The chat needs a running Ollama server. On the cluster, open an interactive
GPU session and start the shared server first:

```bash
srun --account=kamiknows --qos=kamiknows --partition=bare-metal-GPU --gres=gpu:1 --pty bash
export OLLAMA_MODELS=/storage-hpc/kamiknows/models/ollama OLLAMA_HOST=127.0.0.1:11434
/storage-hpc/kamiknows/software/ollama/bin/ollama serve > /dev/null 2>&1 &
```

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
(`sbatch --time=02:00:00 ...`). Set `PYTHON` to use another interpreter.

Comparing several models or prompts means several `sbatch` calls, one output
directory each. Parse, chunk and retrieve run fine on the login node (dense
retrieval and semantic chunking on CPU take about a minute for this corpus).

## Run directories and naming

Rules:

- `--output-dir` must be a **new** directory; scripts refuse to overwrite.
  To redo a run, choose a new name (e.g. add `_retry1`).
- Every run directory contains `manifest.json` with the step name, the full
  command, the git commit (and whether the tree had uncommitted changes),
  all parameters, every input file with its sha256, and the summary metrics.
  Each step checks that its input directory comes from the expected step.
- `outputs/` is not versioned. `outputs_backup/` holds the runs made with the
  pre-refactor code; it is kept for reference only.

Directory names are chosen by hand. Use this pattern so names stay readable
and sortable (the manifest is always the authoritative record):

```text
parsed_<corpus>[_<variant>]
chunks_<corpus>_<chunker>[_<variant>]
retrieval_<chunker>_<retriever>_<sources>_<evalset>_top<k>[_<variant>]
qa_<chunker>_<retriever>_<sources>_<model>_<evalset>_top<k>[_<variant>]
review_<what is reviewed>
figures_<corpus>_inventory_v1
```

| Field | Values |
|---|---|
| corpus | `fastcalo` |
| chunker | `word`, `sentence`, `semantic`; suffixes: `nobm` = `--exclude-backmatter`, `fd100` = `--cut-policy first_drop --min-words 100` |
| retriever | `tfidf`, `dense` (add the model if not MiniLM, e.g. `dense-e5`) |
| sources | `explicit` = `--source-policy explicit_arxiv`, `allsrc` = `none` |
| model | `qwen06b`, `qwen4b`, `mistral7b`, ... |
| evalset | short tag of the questions file: `paired` = `data/questions/fastcalo_paired.jsonl` |
| variant | the one thing that differs from the baseline, e.g. `synthesis`, `np512` |

Figure inventories use `inventory` for the mode and `v1` for inventory schema 1.0;
the corpus tag follows the same vocabulary (`fastcalo`). Retries append `_retry<N>`.

## Outputs index

`docs/outputs_index.md` is the versioned list of runs, since `outputs/` is
not in git. It is generated from the manifests: one table per step with date,
key parameters and metrics. Free-text notes (what changed compared with which
baseline, why a run failed) go in `docs/outputs_index_notes.json`:

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
| `reference_evidence` | list of `{arxiv_id, section_heading, quote}` supporting the reference answer |
| `pair_id`, `source_variant`, `question_type` | grouping for analysis and review |

These fields are used for evaluation and review only: they never reach the
ranking or the prompt. Extra fields are allowed and carried along.

`data/questions/fastcalo_paired.jsonl` (run tag `paired`) has 10 pairs:
each question appears once naming its arXiv id (`explicit`) and once without
(`implicit`). 16 are answerable from the corpus and 4 should be refused. The
reference answers are drafts, not yet validated by an expert.

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

## KamiPlotExtractor: figure inventory

Task 01 inventories LaTeX `figure`/`figure*` environments from an existing parse
run, resolves original local graphics, and records captions, labels, verified
section IDs, explicit subfigures, layout options and SHA256 hashes. It needs no new
dependencies or downloads and does not change the RAG pipeline.

```bash
python scripts/extract_plots.py \
  --parsed-dir outputs/parsed_fastcalo \
  --output-dir outputs/figures_fastcalo_inventory_v1 \
  --mode inventory
```

The new run contains `figures.jsonl`, `panels.jsonl`, `diagnostics.jsonl`,
`figure_summary.json` and the standard `manifest.json` (`step: extract_plots`).
Records use schema 1.0 and deterministic IDs linked to existing paper IDs. Missing,
unsafe or ambiguous asset references remain in the inventory with structured
warnings. Source assets and previous run directories are never modified.
After a run, add an outputs-index note and run `python scripts/index_outputs.py`.

A **figure is not necessarily a plot**. Every figure is provisionally `unknown`;
classification, numerical extraction and digitization are not implemented.
The planned `auto`, `pymupdf`, `plotdigitizer`, `chartdetective` and `plotsift`
backends are all deferred; the interactive tools will need validated import/export
workflows, not assumed automation APIs. Automatic inventory checks do not establish
scientific correctness: manual review pending.

See [the module documentation](docs/kamiplotextractor.md) for the Python API,
versioned output fields, supported LaTeX, safety rules and known limitations.
