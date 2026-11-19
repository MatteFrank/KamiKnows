"""Build the three KamiKnows slide decks from this file.

    python docs/slides/build_slides.py

Writes KamiKnows_pipeline.pdf (repository guide, at the repository root),
docs/plot_extractor/KamiKnows_plot_extraction.pdf (plot extraction walkthrough)
and docs/compute/KamiKnows_compute.pdf (compute and solver walkthrough).
Content lives here; layout helpers are in deck.py; fonts and images are in
fonts/ and assets/. Numbers quoted on the slides come from the fastcalo runs
described in docs/plot_extractor/qa_results.md and must be updated by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck import ACCENT, HERE, LEFT, MUTED, RIGHT, Deck, esc  # noqa: E402

ROOT = HERE.parents[1]
ASSETS = HERE / "assets"
DATE = "19/11/2026"
AUTHORS = "E. Apicella, F. Corchia, M. Franchini"

# Plot questions (fastcalo_plots_values.jsonl, 18 answerable + 2 to refuse), by inspection; review pending.
RESULTS_ROWS = [
    ["<b>current: every image of the corpus (224 plot chunks), one plot chunk per figure, <code>--solver</code></b>",
     "<b>7 / 18 (+1 partial); new 2 / 8</b>", "<b>2 / 18; new 2 / 8</b>"],
    ["same, without the one-per-figure cap", "5 / 18; new 1 / 8", "2 / 18; new 1 / 8"],
    ["earlier: 77 plot chunks (2D maps and one 1D figure only), <code>--solver</code>", "9 / 18; new 2 / 8",
     "5 / 18 (+1 partial); new 1 / 8"]]
RESULTS_NOTE = """<ul class="small">
<li>"New" = pairs 7-10, values other than peaks. Both count questions are refused correctly in every run.</li>
<li>With every plot in the index the task is fairer and harder: the planner chooses among many similar
plots of the same paper. Retrieval reaches the required series for 19 / 20 questions, so the misses are now
the planner's: a decline although the right plot is there, the wrong panel, a computed number the answer
does not use, the wrong axis.</li>
<li>Qwen never invents a value: without the number it abstains. Mistral 7B rarely writes a valid plan and
invents values when its plans fail.</li>
<li>Near-identical runs differ by one or two questions: with 20 questions, read the trend, not the last digit.
By inspection; expert review pending.</li></ul>"""


def overview() -> Deck:
    deck = Deck(f"KamiKnows - pipeline guide | {DATE}")
    deck.cover("Kami", "Knows",
               "How the pipeline works: one command per step,<br>every answer traceable back to the paper it came from.",
               "Text and numerical evidence from scientific papers.", AUTHORS,
               f"{DATE}, Pipeline guide", ASSETS / "kami_logo.png")

    s = deck.slide("How the repository is organized")
    s.code((LEFT, 112, 500, 365), """
kamiknows/        library: one module per step
  parsing.py  chunking.py  retrieval.py
  generation.py  review.py
  embeddings.py  common.py   (shared)
  plot_extractor/  plot extraction route
  compute/         numeric operations, plans
scripts/          one command per step
  parse.py  chunk.py  retrieve.py
  generate.py  review.py  chat.py
  extract_plots.py  compute.py
  run_ollama.slurm  index_outputs.py
tests/            offline tests
data/input/       arXiv id lists
data/questions/   questions + expected answers
docs/             guides, plot extraction, slides
outputs/          one directory per run (local)""", size=10)
    s.html((540, 112, RIGHT, 460), """
<div class="h">scripts/ = how you run it</div>
<p class="muted">Reads the command-line options, calls one function of the library, prints the
summary. Every script has <code>--help</code>.</p><br>
<div class="h">kamiknows/ = how it works</div>
<p class="muted">All the logic: parser, chunkers, retrievers, prompts, citation checks, plot
extraction, numeric operations. The chat reuses the same functions as the benchmark.</p><br>
<div class="h">outputs/ = what each run produced</div>
<p class="muted">Never versioned, never overwritten. <code>outputs/INDEX.md</code> lists the local
runs; versioned files never refer to them.</p>""")

    s = deck.slide("The pipeline: five steps, five commands")
    steps = [("1. Parse", "parse.py", "Download LaTeX; extract sections, equations and figure files.", "sections.json"),
             ("2. Chunk", "chunk.py", "Split text into passages; optionally one chunk per plot.", "chunks.jsonl"),
             ("3. Retrieve", "retrieve.py", "Top-k passages per question: TF-IDF or dense.", "contexts.jsonl"),
             ("4. Generate", "generate.py", "The model answers from the saved passages and cites them; "
              "<code>--solver</code> computes plot numbers first.", "answers.jsonl"),
             ("5. Review", "review.py", "Check answers, citation support and abstentions.", "review.csv")]
    width, gap = 155, 24
    for i, (title, script, body, out) in enumerate(steps):
        x = LEFT + i * (width + gap)
        s.card((x, 115, x + width, 290), title, f'<code>{script}</code><br><br>{body}', "-> " + out)
        if i:
            s.arrow((x - gap + 4, 202), (x - 4, 202))
    s.box((LEFT, 325, RIGHT, 385), "#f7f4ef")
    s.html((LEFT + 14, 333, RIGHT - 10, 382),
           '<div class="h" style="font-size:13pt">Optional: plot extraction (<code>extract_plots.py</code>)</div>'
           '<div class="pre">figure files -> inventory -> classify -> calibrate -> digitize -> chunk.py --digitized-dir</div>')
    s.arrow((LEFT + 78, 290), (LEFT + 78, 325))
    s.arrow((LEFT + width + gap + 78, 325), (LEFT + width + gap + 78, 290))
    for i, (title, body) in enumerate([
        ("One step, one command", "Variants are options, not new scripts."),
        ("Every run is traceable", "Parameters, inputs, hashes and metrics in manifest.json."),
        ("Re-run from the change", "Reuse upstream runs; write a new output directory.")]):
        x = LEFT + i * 295
        s.html((x, 405, x + 280, 470), f'<div class="h" style="font-size:13pt">{title}</div><div class="muted small">{body}</div>')

    s = deck.slide("Step 1 · Parse")
    s.html((LEFT, 112, 470, 330), """<ul>
<li>Downloads metadata and the LaTeX source (e-print) of each paper.</li>
<li>Inlines <code>\\input</code> / <code>\\include</code> and splits the text at <code>\\section</code>,
<code>\\subsection</code>, <code>\\subsubsection</code>.</li>
<li>Citations become <code>[REF]</code>; display equations become <code>[EQUATION_001]</code>
placeholders, saved with their LaTeX and context.</li>
<li>PDF-only papers are recorded as <code>failed</code>, never guessed.</li>
<li>The figure files stay in <code>source/</code>: plot extraction reuses this same run.</li></ul>""")
    s.html((LEFT, 330, 470, 390), '<p class="muted">Current corpus: <b>5 papers, 70 sections, 28 equations</b> '
           '(CaloGAN, WGAN, CaloFlow, ...).</p>')
    s.code((500, 112, RIGHT, 200), """
python scripts/parse.py \\
  --ids-file data/input/arxiv_ids_fastCalo_5fulltext.txt \\
  --output-dir outputs/parsed_fastcalo""", "COMMAND")
    s.label(500, 215, RIGHT, "ID FILE")
    s.html((500, 235, RIGHT, 280), '<p class="muted">One arXiv id per line (<code>1705.02355v2</code>, '
           '<code>arXiv:...</code> or an abs URL); <code>#</code> starts a comment.</p>')
    s.code((500, 285, RIGHT, 400), """
papers.jsonl                 status of every paper
papers/<id>/paper.json       metadata, warnings, errors
papers/<id>/sections.json    [{section_id, heading, text}]
papers/<id>/equations.json   [{equation_id, raw_latex, ...}]
papers/<id>/flat.tex, source/""", "OUTPUT")

    s = deck.slide("Plot extraction: numbers from the figures")
    for i, (title, body, foot) in enumerate([
        ("Inventory", "Captions, labels and figure files from the parse run.", "figures.jsonl"),
        ("Classify", "Gemma 3 4B says what each image is: plot type, axes, panels.", "asset_predictions.jsonl"),
        ("Calibrate", "Match tick labels to tick marks; fit linear or log10 axes.", "calibration_proposals.json"),
        ("Digitize", "Read vector paths or colour cells through the axes.", "digitized_data.jsonl")]):
        x = LEFT + i * 222
        s.card((x, 112, x + 206, 245), title, body, foot)
        if i:
            s.arrow((x - 14, 178), (x - 3, 178))
    s.code((LEFT, 265, RIGHT, 385), """
python scripts/extract_plots.py inventory --parsed-dir $PARSED --output-dir $INVENTORY
sbatch scripts/run_ollama.slurm scripts/extract_plots.py classify --classifier vision \\
  --model gemma3:4b --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CLASSIFIED
python scripts/extract_plots.py calibrate --classified-dir $CLASSIFIED --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CALIBRATED
python scripts/extract_plots.py digitize --classified-dir $CLASSIFIED --inventory-dir $INVENTORY --parsed-dir $PARSED \\
  --calibration-file $CALIBRATED/calibration_proposals.json --output-dir $DIGITIZED
python scripts/chunk.py sentence --parsed-dir $PARSED --exclude-backmatter --digitized-dir $DIGITIZED --output-dir $CHUNKS""",
           size=9.4)
    s.note("Each plot becomes one chunk that describes it and links its values; the solver computes the numbers "
           "a question needs. Full walkthrough: <code>docs/plot_extractor/KamiKnows_plot_extraction.pdf</code>.", 400)

    s = deck.slide("Step 2 · Chunk")
    s.table((LEFT, 112, 470, 300), ["Chunker", "How it cuts"], [
        ["<code>word</code>", "Fixed windows of <code>--target-words</code> (400), overlap <code>--overlap-words</code> (60). "
                              "A final tail under 80 words joins the previous window."],
        ["<code>sentence</code>", "Packs whole sentences up to the target; the overlap is made of whole sentences."],
        ["<code>semantic</code>", "Like sentence, but cuts where the embedding similarity between neighbouring "
                                  "sentences drops (a topic change). See next slide."]], [0.22, 0.78])
    s.html((LEFT, 315, 470, 460), '<p class="muted">Chunks never cross sections. <code>--exclude-backmatter</code> '
           'drops acknowledgments and bibliography (logged in <code>section_exclusions.jsonl</code>).</p>'
           '<p class="muted"><code>--digitized-dir</code> adds one chunk per plotted image to any chunker: caption, '
           'citing sentences, axes, ranges and maxima, and a <code>plot_link</code> to its digitized values.</p>')
    s.code((500, 112, RIGHT, 280), """
python scripts/chunk.py word \\
  --parsed-dir outputs/parsed_fastcalo \\
  --output-dir outputs/chunks_fastcalo_word

python scripts/chunk.py semantic \\
  --parsed-dir outputs/parsed_fastcalo \\
  --output-dir outputs/chunks_fastcalo_semantic_fd100_nobm \\
  --exclude-backmatter --cut-policy first_drop \\
  --min-words 100 --plots""", "COMMANDS")
    s.label(500, 300, RIGHT, "EACH CHUNK IN chunks.jsonl")
    s.html((500, 322, RIGHT, 460), '<p class="small"><code>chunk_id</code> (<code>&lt;paper&gt;::semantic::0007</code>), '
           'paper and section, <code>text</code>, <code>words</code>, <code>cut_reason</code> (semantic_drop, '
           'length_limit, section_end; plot), <code>equation_ids</code>, <code>flags</code>. Plot chunks also carry '
           '<code>plot_link</code>: record, series handles, panels of the same figure.</p>')

    s = deck.slide("Semantic chunking: where to cut")
    s.html((LEFT, 112, 390, 400), """<ul>
<li>Each sentence becomes a vector (<code>all-MiniLM-L6-v2</code>).</li>
<li>At every sentence boundary: cosine similarity between the 2 sentences before and the 2 after.</li>
<li><b>Candidate</b> cut: similarity in the lowest 25% of its section.</li>
<li><code>first_drop</code>: cut at the first candidate where the similarity fell by at least 0.10 from the
previous boundary.</li>
<li>The chunk must stay within <code>--min-words</code> and <code>--target-words</code>; otherwise it ends at
the size limit or the section end.</li></ul>""")
    s.image((405, 108, RIGHT, 370), ASSETS / "semantic_chunking.png")
    s.note("With <code>--plots</code>: one figure per section plus an overview comparing word / sentence / "
           "semantic chunk lengths.", 420)

    s = deck.slide("Step 3 · Retrieve")
    s.card((LEFT, 112, 300, 220), "TF-IDF", "Weighted word counts, compared word by word. Title and section heading count twice.")
    s.card((315, 112, 572, 220), "Dense", "Cosine between MiniLM embeddings of the question and of "
           '"title section text". Model: <code>--embedding-model</code>.')
    s.card((LEFT, 235, 572, 340), "Source policy", "<code>explicit_arxiv</code> (default): if the question names an "
           "arXiv id, only that paper is searched and its best chunk is always kept. A paper missing from the "
           "corpus gives no passages, so the model must abstain.")
    s.html((LEFT, 355, 572, 460), '<p class="muted">The query is <b>the question text only</b>. Expected papers and '
           'plot sources in the questions file score retrieval, never rank. Plot chunks share the index with text; '
           'at most one plot chunk per figure enters the passages (<code>--max-plots-per-figure</code>, default 1), '
           'and it links the other panels of its figure.</p>')
    s.code((600, 112, RIGHT, 235), """
python scripts/retrieve.py \\
  --dataset-dir outputs/chunks_... \\
  --questions data/questions/fastcalo_paired.jsonl \\
  --output-dir outputs/retrieval_... \\
  --retriever dense --top-k 5""", "COMMAND", size=9)
    s.label(600, 255, RIGHT, "WHAT TO CHECK")
    s.html((600, 277, RIGHT, 460), '<p class="muted small"><code>contexts.jsonl</code>: labels, ranked passages, '
           'source constraints and provenance.</p><p class="muted small">hit@k / MRR measure paper retrieval; '
           '<code>expected_plot_sources</code> measure separately whether the required plot series was found.</p>')

    s = deck.slide("Step 4 · Generate")
    s.code((LEFT, 112, 460, 225), """
python scripts/generate.py \\
  --retrieval-dir outputs/retrieval_... \\
  --output-dir outputs/qa_... \\
  --model qwen3:4b-instruct \\
  --prompt grounded          # or synthesis""", "COMMAND (on a GPU node, see SLURM)")
    s.html((LEFT, 245, 460, 460), """<ul>
<li>The model sees exactly the saved passages <code>[C1]...[C5]</code>: retrieval is not re-run.</li>
<li><code>grounded</code>: answer only if supported, otherwise abstain. <code>synthesis</code>: may combine
passages and give marked partial answers.</li>
<li><code>--question-ids</code> re-asks only some questions.</li>
<li>Answers are saved one by one: an interrupted job keeps what it finished.</li>
<li><code>--solver</code>: for plot passages the model writes a plan, a program computes the numbers, the
answer cites them. No multimodal model is needed.</li></ul>""")
    s.code((490, 112, RIGHT, 225), """
{"answer_status": "answered",
 "answer": "CaloGAN, a GAN ... [C1]",
 "citations": ["C1", "C3"], "confidence": "high",
 "citation_validation": {"status": "valid"},
 "warnings": [], "generation": {"eval_count": 312}}""", "EACH ANSWER", size=9.2)
    s.html((490, 245, RIGHT, 460), '<p class="small"><b>Citation check:</b> <code>valid</code>, '
           '<code>invented_citation</code> (label not among the passages), <code>answered_without_citation</code>. '
           'It checks labels, not whether the passage supports the claim.</p>'
           '<p class="muted small">Warnings flag <code>json_parse_error</code> and <code>truncated_at_token_limit</code>; '
           'a failed request is recorded as <code>backend_error</code> and the run continues.</p>')

    s = deck.slide("Compute and the solver: numbers from plots")
    flow = [("Retrieve", "plot chunk + plot_link"), ("Plan", "the model writes JSON steps"),
            ("Compute", "kamiknows.compute runs them"), ("Answer", "the model cites the numbers")]
    for i, (title, body) in enumerate(flow):
        x = LEFT + i * 222
        s.card((x, 112, x + 206, 205), title, body)
        if i:
            s.arrow((x - 14, 158), (x - 3, 158))
    s.html((LEFT, 220, 460, 470), """<ul class="small">
<li>Every extracted plot is a <b>Series</b>: a discrete function on a 1D or 2D domain, with units and scales.</li>
<li><b>13 primitive operations</b> in five families (selection, evaluation, algebra, reduction, calculus); any
numeric question on a plot is a composition of them. Units are checked; what a plot cannot give is refused
with a reason.</li>
<li>The model never reads raw values: it plans, the program computes exactly, the result is one more
citable passage. A failed plan goes back with the exact error, up to 3 plans.</li>
<li><code>scripts/compute.py</code> checks hand-written plans with no model: 20 / 20 match.</li></ul>""")
    s.code((490, 220, RIGHT, 400), """
{"steps": [
  {"id": "blue", "op": "reduce",
   "args": {"series": "<asset>/path_2", "how": "max"}},
  {"id": "red", "op": "reduce",
   "args": {"series": "<asset>/path_3", "how": "max"}},
  {"id": "ratio", "op": "combine",
   "args": {"left": "$blue", "right": "$red", "op": "div"}}],
 "result": ["$ratio"]}
-> 1.59851  div(max(blue curve), max(red curve))""", "A PLAN AND ITS RESULT", size=8.8)
    s.note("Full walkthrough, from the question to the answer: <code>docs/compute/KamiKnows_compute.pdf</code>.", 430)

    s = deck.slide("Step 5 · Review")
    s.html((LEFT, 112, 470, 260), """<ul>
<li>One CSV for one or more generation runs: side-by-side comparison of models, prompts, retrievers.</li>
<li>Each row: question, expected behaviour, <b>reference answer</b>, required points and evidence, the model
answer, the citation check, the text of the cited passages.</li>
<li>Reviewers fill three labels and notes, then set <code>review_status</code>.</li></ul>""")
    s.table((LEFT, 280, 470, 470), ["Column", "Allowed values"], [
        ["<code>reviewer_correctness</code>", "correct, partially_correct, incorrect, not_applicable"],
        ["<code>reviewer_citation_support</code>", "supported, partially_supported, unsupported, not_applicable"],
        ["<code>reviewer_abstention</code>", "appropriate, should_have_answered, should_have_abstained, not_applicable"]],
        [0.48, 0.52])
    s.code((500, 112, RIGHT, 245), """
python scripts/review.py prepare \\
  --qa-dirs outputs/qa_A outputs/qa_B \\
  --output-dir outputs/review_A_vs_B
# after filling review.csv
python scripts/review.py summarize \\
  --csv outputs/review_A_vs_B/review.csv""", "COMMANDS")
    s.html((500, 280, RIGHT, 460), '<p><b>Why it matters.</b> <span class="muted">Automatic checks validate '
           'formats and citation labels. Only the review says whether an answer is scientifically correct.</span></p>'
           '<p class="muted small"><code>prepare</code> also writes <code>review.html</code> to read answers next to '
           'the cited passages.</p>')

    s = deck.slide("Running on SLURM, and the chat")
    s.code((LEFT, 112, 560, 210), """
sbatch --job-name=KK-qa-qwen4b scripts/run_ollama.slurm \\
  scripts/generate.py \\
  --retrieval-dir outputs/retrieval_... \\
  --output-dir outputs/qa_... --model qwen3:4b-instruct""", "GENERATION ON A GPU NODE")
    s.html((LEFT, 225, 560, 460), """<ul>
<li>Starts the shared Ollama from <code>/storage-hpc/kamiknows</code> on a port derived from the job id.</li>
<li>Waits until it answers, runs the command and appends <code>--base-url</code>.</li>
<li>Copies <code>slurm.log</code> and <code>ollama.log</code> into the run folder. Live log:
<code>outputs/_slurm_logs/</code>.</li>
<li>1 GPU, 4 CPUs, 16 GB, 1 h by default; override with <code>sbatch --time=02:00:00</code>.</li>
<li>Gemma classification uses the same runner. Parse, chunk, retrieve, calibrate and digitize run on CPU.</li></ul>""")
    s.code((590, 112, RIGHT, 190), """
# 1. login node: wait for the new prompt
srun --account=kamiknows --qos=kamiknows \\
  --partition=bare-metal-GPU --gres=gpu:1 --pty bash""", "INTERACTIVE CHAT", size=8.6)
    s.code((590, 195, RIGHT, 380), """
# 2. inside the GPU session (check hostname)
conda activate kamiknows
export OLLAMA_MODELS=/storage-hpc/kamiknows/models/ollama
export OLLAMA_HOST=127.0.0.1:11434
/storage-hpc/kamiknows/software/ollama/bin/ollama \\
  serve > /tmp/ollama-$USER.log 2>&1 &
until curl -sf $OLLAMA_HOST/api/tags; do sleep 2; done
python scripts/chat.py --dataset-dir outputs/chunks_... \\
  --model qwen3:4b-instruct""", size=8.6)
    s.html((590, 325, RIGHT, 470), '<p class="muted small">Run the two blocks separately; <code>exit</code> frees '
           'the GPU. Each question is independent; turns are saved in <code>outputs/chat/</code>.</p>')

    s = deck.slide("Every run is traceable")
    s.code((LEFT, 112, 450, 265), """
{"step": "retrieve",
 "created_at": "2026-10-21T14:19:12Z",
 "command": ["python", "scripts/retrieve.py", ...],
 "git": {"commit": "46d6999...", "dirty": false},
 "params": {"retriever": "dense", "top_k": 5, ...},
 "inputs": {"chunks": {"path": ..., "sha256": ...},
            "questions": {...}},
 "summary": {"hit_rate": 0.875, "mrr": 0.66, ...}}""", "manifest.json (in every run folder)", size=9.2)
    s.html((LEFT, 285, 450, 460), """<ul>
<li><code>--output-dir</code> must be new: scripts refuse to overwrite; retry as <code>..._retry1</code>.</li>
<li>Each step checks that its input folder comes from the previous step.</li>
<li>After a run: add a note to <code>outputs/index_notes.json</code> and run
<code>python scripts/index_outputs.py</code> (writes <code>outputs/INDEX.md</code>, local only).</li></ul>""")
    s.code((480, 112, RIGHT, 238), """
parsed_<corpus>
chunks_<corpus>_<chunker>[_<variant>]
retrieval_<chunker>_<retriever>_<sources>_<evalset>_top<k>
qa_<chunker>_<retriever>_<sources>_<model>_<evalset>_top<k>
review_<what is reviewed>
figures_<corpus>_inventory_v1 | figures_<corpus>_classified_...
calibrations_<corpus>_... | plots_<corpus>_...
compute_<evalset>_<plans>""", "NAMING PATTERN", size=9)
    s.table((480, 255, RIGHT, 470), ["Tag", "Meaning"], [
        ["<code>fd100</code>", "cut policy first_drop, at least 100 words per chunk"],
        ["<code>nobm</code>", "no back matter (acknowledgments, bibliography)"],
        ["<code>dense</code>", "embedding retriever (MiniLM); <code>tfidf</code> lexical"],
        ["<code>explicit</code>", "source policy explicit_arxiv; <code>allsrc</code> = none"],
        ["<code>paired</code> / <code>plots</code>", "questions file fastcalo_paired.jsonl / fastcalo_plots.jsonl"],
        ["<code>plotvalues</code>", "fastcalo_plots_values.jsonl; <code>plotplans</code>: the same with plans"]],
        [0.28, 0.72])

    s = deck.slide("Questions files")
    s.html((LEFT, 112, 470, 280), """<ul>
<li>One evaluation = <b>one JSONL file</b> in <code>data/questions/</code>: questions and expected answers.</li>
<li><b>Never edit a file that runs have used</b>: copy it to a new descriptive name, edit the copy, use a
new tag in run names.</li>
<li>Expected fields score retrieval and support the review; they never reach the ranking or the prompt.</li></ul>""")
    s.table((LEFT, 290, 470, 470), ["Field", "Used for"], [
        ["<code>question_id</code>, <code>question</code>", "required"],
        ["<code>expected_behavior</code>", "answerable / insufficient_evidence"],
        ["<code>expected_arxiv_ids</code>", "retrieval hit@k and MRR"],
        ["<code>expected_plot_sources</code>", "plot series that must be retrieved"],
        ["<code>reference_answer</code>, <code>required_answer_points</code>", "review sheet"]], [0.55, 0.45])
    s.label(500, 112, RIGHT, "INCLUDED QUESTION SETS")
    s.html((500, 135, RIGHT, 300), '<p><code>fastcalo_paired.jsonl</code>: <b>20 questions</b> on paper text, in '
           '10 pairs. The explicit variant names the arXiv id, the implicit one does not. 16 answerable, 4 to refuse.</p>'
           '<p><code>fastcalo_plots.jsonl</code>: <b>12 questions</b> on plot values, in 6 pairs (figure reference / '
           'natural). <code>fastcalo_plots_values.jsonl</code> adds 4 pairs on values other than peaks: <b>20</b>, '
           '18 answerable. Its <code>_plans</code> copy adds a hand-written plan per question.</p>')
    s.html((500, 315, RIGHT, 360), '<p class="muted">Reference answers are drafts, not yet validated by an expert.</p>')

    s = deck.slide("Question types and examples")
    s.table((LEFT, 112, RIGHT, 420), ["Type (question_type)", "N", "Example (shortened from the included questions)"], [
        ["<code>fact_lookup</code>", "2", "Which generative model family does CaloGAN use?"],
        ["<code>method</code>", "6", "What do Flow I and Flow II generate in CaloFlow?"],
        ["<code>source_lookup</code>", "2", "What speed-up relative to GEANT4 does the WGAN paper report?"],
        ["<code>comparison</code>", "6", "How do CaloGAN and CaloFlow differ in access to a likelihood?"],
        ["<code>no_answer</code>", "4", "What exact ATLAS Run 4 deployment date do the papers prescribe?"],
        ["<code>plot_data</code>", "12", "At what transverse momentum does the unweighted jet distribution peak, and how high?"]],
        [0.22, 0.06, 0.72])
    s.note("The first five types are in fastcalo_paired.jsonl, <code>plot_data</code> in fastcalo_plots.jsonl. "
           "Topic and answerability are separate fields. Valid citations do not establish correctness.", 275)

    s = deck.slide("Quick start")
    s.code((LEFT, 112, RIGHT, 360), """
conda activate kamiknows        # pip install -r requirements.txt
PARSED=outputs/parsed_fastcalo  CHUNKS=outputs/chunks_fastcalo_semantic_fd100_nobm
RETRIEVAL=outputs/retrieval_semantic_fd100_nobm_dense_explicit_paired_top5
ANSWERS=outputs/qa_semantic_fd100_nobm_dense_explicit_qwen4b_paired_top5

python scripts/parse.py    --ids-file data/input/arxiv_ids_fastCalo_5fulltext.txt --output-dir $PARSED
python scripts/chunk.py semantic --parsed-dir $PARSED --output-dir $CHUNKS \\
       --exclude-backmatter --cut-policy first_drop --min-words 100 --plots
python scripts/retrieve.py --dataset-dir $CHUNKS --questions data/questions/fastcalo_paired.jsonl \\
       --output-dir $RETRIEVAL --retriever dense --top-k 5
sbatch scripts/run_ollama.slurm scripts/generate.py --retrieval-dir $RETRIEVAL --output-dir $ANSWERS \\
       --model qwen3:4b-instruct
python scripts/review.py prepare --qa-dirs $ANSWERS --output-dir outputs/review_qwen4b_paired
python scripts/index_outputs.py

pytest -q                       # offline tests of every step""", size=10)
    s.note("Full reference: <code>README.md</code>. Plot extraction: <code>docs/plot_extractor/</code>. Compute: <code>docs/compute/</code>. "
           "Rules for coding agents: <code>AGENTS.md</code>. Every script: <code>--help</code>.", 378)
    return deck


def plot_extraction() -> Deck:
    deck = Deck(f"KamiKnows - plot extraction | {DATE}")
    deck.cover("Kami", "Knows", "Plot extraction: from paper figures<br>to cited numerical evidence.",
               "How figures become chunks the pipeline retrieves, and values the solver computes on.", AUTHORS,
               f"{DATE}, Plot extraction guide", ASSETS / "kami_logo.png")

    s = deck.slide("Why plot extraction")
    s.html((LEFT, 112, 450, 330), """
<p>Many answers in a physics paper are only in a figure: where a distribution peaks, how two curves
compare, which axis is logarithmic. A text-only RAG cannot see them.</p>
<p><b>Idea:</b> read the numbers out of the original figure files once; describe each plot in one text
chunk linked to its values; compute the numbers a question needs with a program.</p>
<ul><li>The generator (Qwen, Mistral) stays text-only.</li>
<li>Every value points back to a figure file, a series and a hash.</li>
<li>What cannot be read is reported, never invented.</li></ul>""")
    s.box((480, 112, RIGHT, 330), "#f7f4ef", "#e6ddd0")
    s.label(495, 122, RIGHT, "EXAMPLE QUESTION")
    s.html((495, 142, RIGHT - 12, 325), '<p>"At approximately what particle-level transverse momentum does the '
           'unweighted hadronic-jet distribution peak, and how high is that peak?"</p>'
           '<p class="muted small">The text says the distribution is "multi-modal" but gives no number. '
           'The values exist only in Figure 1 of arXiv:1805.00850.</p>'
           '<p class="small"><b>Qwen3 4B with the solver:</b> "The unweighted hadronic-jet distribution peaks at '
           'approximately 45 GeV with a height of 0.005347 A.U." [C4, C6]. [C6] holds the two numbers, computed '
           'from the 246 extracted bins.</p>')
    s.note("In an earlier text-only comparison on the same papers, Qwen abstained on every plot question: "
           "the numbers were simply not in the passages. The solver is in <code>docs/compute/KamiKnows_compute.pdf</code>.", 350)

    s = deck.slide("The chain: one CLI, one run directory per step")
    chain = [("inventory", "CPU"), ("classify", "GPU"), ("calibrate", "CPU"), ("digitize", "CPU"),
             ("chunk", "CPU"), ("retrieve", "CPU"), ("generate", "GPU, --solver")]
    width = 108
    for i, (name, where) in enumerate(chain):
        x = LEFT + i * (width + 19)
        fill = "#f8ede0" if i < 4 else "#f7f4ef"
        s.box((x, 115, x + width, 175), fill, None if i < 4 else "#e6ddd0")
        s.html((x + 8, 124, x + width - 4, 172), f'<div class="h" style="font-size:13pt">{name}</div>'
               f'<div class="muted tiny">{where}</div>')
        if i:
            s.arrow((x - 17, 145), (x - 2, 145))
    s.html((LEFT, 182, 540, 200), '<div class="muted tiny">extract_plots.py subcommands</div>')
    s.html((LEFT + 4 * 127, 182, RIGHT, 200), '<div class="muted tiny">existing pipeline commands</div>')
    s.code((LEFT, 215, RIGHT, 400), """
PARSED=outputs/parsed_fastcalo                     # existing parse run: no new download
INVENTORY=outputs/figures_fastcalo_inventory_v1  CLASSIFIED=outputs/figures_fastcalo_classified_vision_v1
CALIBRATED=outputs/calibrations_fastcalo_auto_v1  DIGITIZED=outputs/plots_fastcalo_auto_v1
python scripts/extract_plots.py inventory --parsed-dir $PARSED --output-dir $INVENTORY
sbatch scripts/run_ollama.slurm scripts/extract_plots.py classify --classifier vision --model gemma3:4b \\
  --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CLASSIFIED
python scripts/extract_plots.py calibrate --classified-dir $CLASSIFIED --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CALIBRATED
python scripts/extract_plots.py digitize --classified-dir $CLASSIFIED --inventory-dir $INVENTORY --parsed-dir $PARSED \\
  --calibration-file $CALIBRATED/calibration_proposals.json --output-dir $DIGITIZED""", size=9.2)
    s.note("Each mode reads the previous run, refuses to overwrite and writes <code>manifest.json</code> with "
           "parameters, input hashes and a summary. <code>calibration-review</code> and "
           "<code>digitization-review</code> render HTML reports.", 380)

    s = deck.slide("1 · Inventory: which figures, which files")
    s.html((LEFT, 112, 440, 330), """<ul>
<li>Reads <code>flat.tex</code> and <code>source/</code> of the parse run.</li>
<li>For each <code>figure</code> environment: caption, <code>\\label</code>, every <code>\\includegraphics</code>
file resolved against <code>\\graphicspath</code>, its format and SHA-256.</li>
<li>One figure can have several files (one per panel), and one file can contain several panels.</li>
<li>Unresolved files and skipped macros go to <code>diagnostics.jsonl</code>.</li></ul>""")
    s.table((LEFT, 345, 440, 470), ["fastcalo corpus", ""], [
        ["papers / figures", "5 / 58"], ["figure files referenced", "224 (217 distinct), all resolved"]], [0.55, 0.45])
    s.code((470, 112, RIGHT, 470), """
{"figure_id": "1705_02355v2::figure::0001",
 "caption": "Average Geant4 shower (top), and average
             CaloGAN shower (bottom), with progressive
             calorimeter depth (left to right).",
 "latex_label": "fig:gamma_avg",
 "source_assets": [
   {"asset_id": "1705_02355v2::figure::0001::asset::01",
    "path": "extracted/figs/gamma_0_geant.pdf",
    "format": "pdf",
    "sha256": "bf856ece83ea...",
    "status": "resolved",
    "layout": {"width": "0.15\\\\textwidth", ...}},
   ... 6 files: 3 layers x (Geant4, CaloGAN)]}""", "figures.jsonl (shortened)", size=8.8)

    s = deck.slide("2 · Classify: what is in each image")
    s.html((LEFT, 112, 440, 345), """<ul>
<li>Each figure file is rendered to PNG and sent alone to <b>Gemma 3 4B</b> through Ollama (temperature 0,
JSON schema).</li>
<li>Gemma returns: role (<code>plot</code>, <code>diagram</code>, <code>legend</code>, ...), representation
(<code>histogram_1d</code>, <code>histogram_2d</code>, <code>line_plot</code>, <code>scatter_plot</code>, ...),
panels, and for each axis whether numbers and units are visible.</li>
<li>The prediction only routes calibration. It never supplies numbers and never overrides what the PDF shows.</li>
<li><code>--classifier text</code>: caption keywords only, no GPU.</li></ul>""")
    s.table((LEFT, 355, 440, 470), ["Gemma, 58 figures", ""], [
        ["histogram_2d / histogram_1d", "14 / 14"], ["mixed / scatter / line / unknown", "22 / 4 / 1 / 3"]], [0.62, 0.38])
    s.code((470, 112, RIGHT, 330), """
{"asset_role": "plot",
 "analysis_type": "histogram_2d",
 "composition": "single_panel",
 "axes": {
   "x": {"presence": "present", "numeric_labels": "present",
         "unit_label": "unclear"},
   "y": {"presence": "present", "numeric_labels": "present",
         "unit_label": "unclear"},
   "colorbar": {"presence": "present", "numeric_labels": "present",
                "unit_label": "present"}},
 "reason": "The image shows a heatmap with a colorbar..."}""", "GEMMA ANSWER FOR gamma_0_geant.pdf", size=8.8)
    s.code((470, 335, RIGHT, 470), """
sbatch scripts/run_ollama.slurm scripts/extract_plots.py \\
  classify --classifier vision --model gemma3:4b \\
  --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CLASSIFIED""", "COMMAND (GPU)", size=9)

    s = deck.slide("3 · Calibrate: from page points to physical values")
    for i, (title, body) in enumerate([
        ("Inspect the source", "Vector paths and text, a hybrid (vector axes around an embedded image) or a bitmap."),
        ("Find frames and ticks", "Plot frames from vector rectangles; tick marks along each edge."),
        ("Match labels", "Each tick gets the nearest numeric label, e.g. 0, 500, ..., 2500 or 10<sup>-3</sup> ... 10<sup>-7</sup>."),
        ("Fit the scale", "Linear or log10 fit; at least 3 consistent labels. Units, multipliers and colour bars recorded.")]):
        x = LEFT + i * 222
        s.card((x, 112, x + 206, 238), title, body)
        if i:
            s.arrow((x - 14, 175), (x - 3, 175))
    s.code((LEFT, 255, 520, 470), """
"x": {"scale": "linear", "unit": "GeV", "unit_status": "explicit",
      "anchors": [{"position": 96.44, "value": 0},
                  {"position": 187.75, "value": 500}, ...,
                  {"position": 552.98, "value": 2500}]},
"y": {"scale": "log10", "unit": "A.U.",
      "anchors": [{"position": 59.19, "value": 0.001}, ...,
                  {"position": 257.68, "value": 1e-07}]}""",
           "calibration_proposals.json: Figure 1 of arXiv:1805.00850", size=8.8)
    s.html((545, 255, RIGHT, 470), """<ul class="small">
<li>No usable text in the PDF (bitmaps, outlined labels): <b>RapidOCR</b> reads the labels; the proposal needs the
same consistent anchors and stays marked unverified.</li>
<li>No reliable mapping: the asset gets a status (<code>needs_review_axes</code>, <code>needs_review_ocr</code>,
<code>needs_review_classification</code>) instead of guessed numbers.</li>
<li><code>review.html</code> draws the matched labels and fits on every asset.</li></ul>""")

    s = deck.slide("4 · Digitize: read the values through the axes")
    s.table((LEFT, 112, 520, 300), ["Source", "Backend", "What is read"], [
        ["Vector curves, step histograms", "<code>pymupdf</code>", "PDF path coordinates; steps keep their plateaus as displayed intervals"],
        ["Hybrid 2D maps", "<code>pymupdf</code>", "colour cells of the embedded image through X, Y and the colour bar"],
        ["Bitmaps", "<code>raster</code>", "coloured traces or constant-colour cells from pixels (prototype)"]],
        [0.33, 0.17, 0.5])
    s.html((LEFT, 315, 520, 470), """<ul class="small">
<li><code>--backend auto</code> chooses from the calibration and falls back from vector geometry to pixels when
the vector route is unsupported.</li>
<li>Failed or skipped files stay in the output with a reason.</li>
<li>Structural checks: finite values, consistent lengths, points inside the plot frame.</li></ul>""")
    s.code((545, 112, RIGHT, 470), """
{"figure_id": "1805_00850v2::figure::0001",
 "extraction_status": "success",
 "backend": "pymupdf",
 "source_method": "vector_reconstructed",
 "axes": {"x": {"scale": "linear", "unit": "GeV"},
          "y": {"scale": "log10", "unit": "A.U."}},
 "series": [{
   "series_id": "path_2",
   "geometry": "rendered_step_intervals",
   "x": [45.0, 55.0, 65.0, ...],
   "y": [0.005347, 0.004990, 0.003832, ...],
   "rendered_intervals": [[40, 50], [50, 60], ...],
   "original_bin_edges_verified": false}, ...],
 "review_status": "pending"}""", "digitized_data.jsonl (shortened)", size=8.8)

    s = deck.slide("Check the result: source vs reconstruction")
    s.image((LEFT, 108, RIGHT, 300), ASSETS / "histogram_1d_review.png")
    s.image((LEFT + 60, 305, RIGHT - 60, 462), ASSETS / "histogram_2d_review.png")
    s.note("<code>digitization-review</code>: left the original figure, right a plot redrawn only from the extracted "
           "numbers. Top: Figure 1 of arXiv:1805.00850 (1D step histograms). Bottom: Figure 1 of arXiv:1705.02355 "
           "(first-layer Geant4 shower, 2D map). A visual check, not a validation against original data.", 462)

    s = deck.slide("What has been digitized so far")
    s.table((LEFT, 112, 560, 330), ["Run", "Files", "Result"], [
        ["Calibration, all figure files (automatic)", "224", "186 proposals; the rest flagged for review"],
        ["Digitization, all figure files (automatic, one run)", "224", "119 digitized images, 322 series"],
        ["&nbsp;&nbsp;by geometry", "", "55 2D maps, 93 step histograms, 174 curves"],
        ["<b>In the chunks</b>", "224", "<b>224 plot chunks</b>: 119 images with values, 105 without"]],
        [0.45, 0.12, 0.43])
    s.html((585, 112, RIGHT, 470), """
<div class="h">Current coverage</div>
<ul class="small">
<li>One calibration and one digitization run cover every figure file; chunking and the solver read
that single folder.</li>
<li>The 105 images without values get a chunk saying why (needs calibration, manual review, generic).</li>
<li>The new 1D figures are not yet checked in the visual report; that review is the next step.</li>
</ul>""")
    s.note("Counts from the fastcalo corpus (5 papers). Values are estimates of what is displayed; manual review "
           "pending.", 360)

    s = deck.slide("5 · Into the chunks: one plot, one chunk, one record")
    s.html((LEFT, 108, RIGHT, 150), '<p class="small">A <b>series</b> is one extracted object: a curve or histogram '
           'of one colour, or the cells of one heatmap. Figure 1 of arXiv:1805.00850 has one image with two series '
           '(blue, red); Figure 1 of arXiv:1705.02355 has six images, one heatmap each. Each <b>image</b> gets one chunk.</p>')
    s.box((LEFT, 158, 205, 300), "#f8ede0")
    s.html((LEFT + 10, 166, 200, 296), '<div class="h" style="font-size:12pt">one image</div>'
           '<div class="muted tiny">its digitized record:<br>series, units, scales,<br>246 + 248 values</div>')
    for i, (title, body) in enumerate([
            ("plot chunk (text)", "caption, citing sentences, panel position, axes, per series ranges and maximum"),
            ("plot_link (metadata)", "record, hashes, compute handle of every series, panels of the same figure"),
            ("digitized_data.jsonl", "the values themselves: never in a chunk, read by the solver")]):
        y = 158 + i * 50
        s.box((265, y, 560, y + 44), "#f7f4ef", "#e6ddd0")
        s.html((275, y + 5, 555, y + 42), f'<div class="small"><b>{title}</b>: {body}</div>')
        s.arrow((207, 229), (263, y + 22))
    s.html((585, 158, RIGHT, 340), """<ul class="small">
<li><b>No extracted values in the text</b>: small models could not look them up in long tables
(0 of 8 questions on values other than peaks).</li>
<li><b>Characteristic numbers from compute</b>: ranges, maximum, ties; card and answer cannot disagree.</li>
<li><b>Plots without values</b> get a chunk too: <i>Data status: plot not digitized (needs_calibration)</i>.</li>
<li>Panel positions are shown but kept out of the ranking text.</li></ul>""")
    s.box((LEFT, 352, RIGHT, 452), "#f7f4ef", "#e6ddd0")
    s.html((LEFT + 12, 360, RIGHT - 12, 450), '<p class="small"><b>Multi-panel figures</b>: every panel lists all '
           'the panels of its figure (<code>figure_panels</code>: position and handles). A question about the Geant4 '
           'panel that retrieves a CaloGAN panel still reaches the Geant4 data, and a comparison reaches both.</p>'
           '<p class="small muted">fastcalo: 224 images → 224 plot chunks (median 209 words), next to 110 text chunks. '
           'The earlier layout had 451 plot chunks (summaries, 24-row tables, figure overviews).</p>')
    s.code((LEFT, 462, RIGHT, 505), """
python scripts/chunk.py sentence --parsed-dir $PARSED --exclude-backmatter --digitized-dir $DIGITIZED --output-dir $CHUNKS""",
           size=8.6)

    s = deck.slide("What a plot chunk looks like")
    s.code((LEFT, 112, 500, 470), """
Plot: Figure 1 of arXiv:1705.02355v2, image 4 of 6.
Panel position: bottom-left panel (explicit source layout).
Figure layout (explicit source layout): image 1: top-left
panel; image 2: top-middle panel; ...; image 6: bottom-right.
Caption: Average Geant4 shower (top), and average CaloGAN
shower (bottom), with progressive calorimeter depth (left
to right).
Original caption: Average $\\gamma$ \\textsc{Geant4} shower ...
Paper discussion of this figure: The average calorimeter
deposition per voxel (Fig. [REF]) suggests that ...
Type: heatmap (2D cells, value shown by colour); 1 series.
Axes: X in cell_index (linear), Y in cell_index (linear),
Z (colour) in MeV (log10).
- heatmap cells: 285 cells; X from -0.5 to 95.5 cell_index;
  Y from -0.5 to 2.5 cell_index; maximum 2135 MeV at
  X = 47 cell_index, Y = 0.9955 cell_index (2 tied cells,
  first shown).
Values estimate the displayed figure, not the original
measurements; original bins, counts and uncertainties are
not available.""", "CHUNK TEXT, arXiv:1705.02355 FIG. 1, CALOGAN FIRST LAYER", size=8.2)
    s.code((520, 112, RIGHT, 360), """
"plot_link": {
  "asset_id": "1705_02355v2::figure::0001::asset::04",
  "image": "4 of 6", "position": "bottom-left panel",
  "data_id": "digitized_c0de...",
  "extraction_status": "success",
  "digitized_data_sha256": "8c3f...",
  "series": [{"handle": "<asset 04>/panel_1_cells",
              "label": "heatmap cells", "usable": true}],
  "figure_panels": [
    {"image": "1 of 6", "position": "top-left panel",
     "asset_id": "...::asset::01", "series": [...]},
    ... 6 panels]}""", "PLOT_LINK (METADATA, NOT RANKED)", size=8.2)
    s.html((520, 370, RIGHT, 470), """<ul class="small">
<li>Handles name the data for the solver: <code>&lt;asset_id&gt;/&lt;series_id&gt;</code>.</li>
<li>The chunker refuses runs whose data or source hashes no longer match, and keeps one record per image.</li></ul>""")

    s = deck.slide("6 · Retrieval: plots and text in one index")
    s.html((LEFT, 112, 470, 470), """<ul>
<li>Plot chunks are ranked together with paper text, using the question text only.</li>
<li><code>expected_plot_sources</code> lists the asset/series a question needs. It counts as retrieved when
a retrieved plot chunk has it, directly or among the panels of its figure.</li>
<li><b>One plot chunk per figure</b> (<code>--max-plots-per-figure 1</code>, the default). The panels of a
multi-panel figure share caption and discussion, so they score alike and could fill all 5 slots; the best
one represents the figure, and its <code>figure_panels</code> lead to the others.</li></ul>""")
    s.table((500, 112, RIGHT, 260), ["Dense MiniLM, top-5, 224 plot chunks", "No cap", "One per figure"], [
        ["expected paper retrieved", "18 / 20", "20 / 20"],
        ["required series, figure-reference questions", "9 / 10", "10 / 10"],
        ["required series, natural questions", "6 / 10", "9 / 10"]], [0.6, 0.2, 0.2])
    s.code((500, 275, RIGHT, 460), """
plot_09_referenced ("In Figure 1 of arXiv:1805.00850v2 ...")
no cap:          C1-C5 = Figure 12, five of its six panels
one per figure:  C1 Figure 12, middle-right panel
                 C2 Figure 1, single image      <- needed
                 C3 Figure 11, top-right panel
                 C4 Figure 10, middle-right panel
                 C5 Figure 8, top-left panel""", "RETRIEVED PASSAGES", size=8.4)

    s = deck.slide("The plot questions: fastcalo_plots_values.jsonl")
    s.html((LEFT, 112, 440, 300), """<ul class="small">
<li>10 pairs, 20 questions. In each pair the <b>figure-reference</b> variant cites figure and arXiv id; the
<b>natural</b> variant names neither, and searches the whole corpus.</li>
<li>Pairs 1-6 are the questions of <code>fastcalo_plots.jsonl</code> (peaks, ratio, axes, counts); pairs 7-10 ask
for values other than peaks.</li>
<li>18 are answerable from the extracted values; 2 ask for original event counts and must be refused.</li>
<li><code>fastcalo_plots_values_plans.jsonl</code> adds a hand-written plan and expected results per question.</li></ul>""")
    s.code((470, 112, RIGHT, 470), """
{"question_id": "plot_08_natural",
 "pair_id": "plot_pair_08",
 "question_style": "natural",
 "question_type": "plot_data",
 "question": "Starting from the lowest transverse
   momentum, in which bin does the training-weighted
   hadronic-jet distribution first exceed the
   unweighted one?",
 "expected_behavior": "answerable",
 "expected_arxiv_ids": ["1805.00850v2"],
 "expected_plot_sources": [
   {"asset_id": "1805_00850v2::figure::0001::asset::01",
    "series_id": "path_2"}, {... "path_3"}],
 "reference_answer": "In the bin centred at 195 GeV
   (190-200 GeV): red approximately 0.001657 A.U.
   versus blue approximately 0.001453 A.U. ..."}""", "ONE QUESTION (shortened)", size=8.6)
    s.table((LEFT, 310, 440, 470), ["Field", "Role"], [
        ["<code>question</code>", "the only text used for ranking and in the prompt"],
        ["<code>expected_*</code>, <code>reference_answer</code>", "evaluation and review only"]], [0.5, 0.5])

    s = deck.slide("The ten pairs")
    s.table((LEFT, 112, RIGHT, 480), ["Pair", "Figure", "Asks for", "Reference answer"], [
        ["1", "1705.02355 Fig. 1, Geant4 layer 1", "brightest cell: energy and position", "≈2137 MeV at x=47.54, y=1.00"],
        ["2", "same, Geant4 vs CaloGAN", "are the two peaks within one cell along η?", "47.544 vs 46.995: 0.549 apart, yes"],
        ["3", "1805.00850 Fig. 1, unweighted", "peak position and height", "x=45 GeV, y≈0.005347 A.U."],
        ["4", "same, two histograms", "ratio of the two peak heights", "≈1.599"],
        ["5", "same", "axis scales and units", "X linear in GeV, Y log10 in A.U."],
        ["6", "same", "original events in the top bin", "<b>refuse</b>: not recoverable"],
        ["7", "same, two histograms", "heights in the 500-510 GeV bin", "0.000691 and 0.000761 A.U."],
        ["8", "same, two histograms", "first bin where the weighted one exceeds the other", "190-200 GeV: 0.001657 vs 0.001453"],
        ["9", "same, unweighted", "local maximum between 250 and 400 GeV", "305 GeV, 0.001242 A.U."],
        ["10", "1705.02355 Fig. 1, Geant4 layer 1", "energy in the cell at x≈20, y≈1", "≈0.70 MeV"]],
        [0.06, 0.27, 0.36, 0.31])

    s = deck.slide("How the models answer")
    s.table((LEFT, 112, RIGHT, 300), ["Configuration (20 plot questions)", "Qwen3 4B", "Mistral 7B"],
            RESULTS_ROWS, [0.52, 0.24, 0.24])
    s.html((LEFT, 310, RIGHT, 470), RESULTS_NOTE)

    s = deck.slide("Answers, good and bad")
    s.code((LEFT, 112, 470, 290), """
Q (natural): heights of the unweighted and the
training-weighted distributions in the 500-510 GeV bin?
Qwen + solver: [C6] max(restrict(blue, x, 500, 510))
  = 0.000691114 A.U.; same for red = 0.000761055 A.U.
  "... 0.000691114 A.U., while the training-weighted
   distribution has a height of 0.000761055 A.U."
Reference: 0.000691 and 0.000761 A.U.""", "PLOT PAIR 7: COMPUTED, CORRECT", size=8.6)
    s.code((490, 112, RIGHT, 290), """
Q: ratio between the unweighted and training-weighted
peak heights?
Mistral, plot cards only: "approximately 1.55"
Mistral + solver (natural): [C6] div(max(blue),
  max(red)) = 1.59851 -> "approximately 1.59851"
Reference: 1.599""", "PLOT PAIR 4: THE SOLVER FIXES THE ARITHMETIC", size=8.6)
    s.code((LEFT, 305, 470, 470), """
Q: in which bin does the red histogram first rise
above the blue one?
Planner result: crossing(red, blue, up)[0].x = 195 GeV
  (only the coordinate: no bin, no values)
Qwen: insufficient_evidence
Mistral: "in the 196th bin"
Reference: 190-200 GeV, 0.001657 vs 0.001453 A.U.""", "PLOT PAIR 8: RIGHT NUMBER, BAD PLAN", size=8.6)
    s.code((490, 305, RIGHT, 470), """
Q: are the brightest cells of the Geant4 and CaloGAN
first-layer maps within one cell along eta?
Planner: compares the y coordinates (0.9955 and 0.9955)
Qwen: "both at 0.995509 cell_index ... within one"
Reference: x (eta) 47.544 vs 46.995, 0.549 apart
-> the card says "X in cell_index", not which axis is eta""", "PLOT PAIR 2: WRONG AXIS", size=8.6)

    s = deck.slide("Limits and next steps")
    s.html((LEFT, 112, 450, 470), """
<div class="h">What the extracted values are not</div>
<ul class="small">
<li>Original bin edges, event counts and experimental uncertainties: not recoverable from a drawing.</li>
<li>Legend association: series are named by colour; the caption must say what each colour means.</li>
<li>Broken axes, overlapping traces, 3D plots: extraction stops with a reason.</li>
<li>Source-vs-reconstruction agreement is a visual check, not a scientific validation.</li></ul>""")
    s.html((480, 112, RIGHT, 470), """
<div class="h">Next steps</div>
<ul class="small">
<li>Digitize every calibrated file and check the visual report.</li>
<li>Expert review of the plot answers; a larger question set with hand-written plans.</li>
<li>Better retrieval of the right panel for natural questions.</li>
<li>Compare against original data where it exists (<code>recover</code>: PGFPlots tables, HEPData;
<code>benchmark</code>).</li></ul>""")

    s = deck.slide("All the commands")
    s.code((LEFT, 112, RIGHT, 470), """
PARSED=outputs/parsed_fastcalo  INVENTORY=outputs/figures_fastcalo_inventory_v1  CLASSIFIED=outputs/figures_fastcalo_classified_vision_v1
CALIBRATED=outputs/calibrations_fastcalo_auto_v1  DIGITIZED=outputs/plots_fastcalo_auto_v1  CHUNKS=outputs/chunks_fastcalo_sentence_plots_v1
RETRIEVAL=outputs/retrieval_sentence_plots_dense_explicit_plotvalues_top5  ANSWERS=outputs/qa_sentence_plots_dense_explicit_qwen4b_plotvalues_top5
python scripts/extract_plots.py inventory --parsed-dir $PARSED --output-dir $INVENTORY
sbatch scripts/run_ollama.slurm scripts/extract_plots.py classify --classifier vision --model gemma3:4b \\
       --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CLASSIFIED
python scripts/extract_plots.py calibrate --classified-dir $CLASSIFIED --inventory-dir $INVENTORY --parsed-dir $PARSED --output-dir $CALIBRATED
python scripts/extract_plots.py digitize --classified-dir $CLASSIFIED --inventory-dir $INVENTORY --parsed-dir $PARSED \\
       --calibration-file $CALIBRATED/calibration_proposals.json --output-dir $DIGITIZED
python scripts/extract_plots.py digitization-review --digitized-dir $DIGITIZED --output-dir outputs/plots_fastcalo_visual_v1
python scripts/chunk.py sentence --parsed-dir $PARSED --exclude-backmatter --digitized-dir $DIGITIZED --output-dir $CHUNKS
python scripts/retrieve.py --dataset-dir $CHUNKS --questions data/questions/fastcalo_plots_values.jsonl \\
       --retriever dense --source-policy explicit_arxiv --top-k 5 --output-dir $RETRIEVAL
sbatch scripts/run_ollama.slurm scripts/generate.py --retrieval-dir $RETRIEVAL --output-dir $ANSWERS --model qwen3:4b-instruct --solver
python scripts/review.py prepare --qa-dirs $ANSWERS --output-dir outputs/review_fastcalo_plots""", size=8.6)
    s.note("Guide: <code>docs/plot_extractor/guide.md</code>. Results: <code>docs/plot_extractor/qa_results.md</code>. "
           "Options: <code>python scripts/extract_plots.py &lt;mode&gt; --help</code>.", 400)
    return deck


def compute() -> Deck:
    deck = Deck(f"KamiKnows - compute and solver | {DATE}")
    deck.cover("Kami", "Knows", "Compute and the solver: numbers from plots,<br>computed by a program, not read by the model.",
               "From the question to a cited answer: retrieval, planner, operations, answer.", AUTHORS,
               f"{DATE}, Compute guide", ASSETS / "kami_logo.png")

    s = deck.slide("Why: small models cannot read tables")
    s.html((LEFT, 112, 450, 300), """
<p>A digitized plot holds hundreds of values (246 bins for one histogram, 285 cells for one heatmap).
Putting them in the passages works for retrieval, but a 4B or 7B model cannot look a value up in a long
table: it abstains, takes the wrong row or invents a number.</p>
<p><b>Idea:</b> retrieval finds the plot, the model only <b>plans</b> the computation, a program
<b>computes</b> it exactly on the extracted values, and the model writes the answer from a few numbers.</p>""")
    s.table((480, 112, RIGHT, 300), ["Qwen3 4B, values in the passages", "peak questions", "other values"], [
        ["per-series summaries + 24-row tables", "9 / 10", "0 / 8"],
        ["one chunk per image with every value", "~3 / 10", "1 / 8"]], [0.56, 0.22, 0.22])
    s.html((480, 215, RIGHT, 300), '<p class="muted small">"Other values": a given bin, the first bin where one '
           'histogram overtakes the other, a local maximum in a range, a given heatmap cell. The summaries only '
           'helped because they already held the computed peak.</p>')
    s.box((LEFT, 320, RIGHT, 440), "#f7f4ef", "#e6ddd0")
    s.html((LEFT + 14, 330, RIGHT - 14, 436), """<div class="h" style="font-size:13pt">Three rules</div>
<ul class="small"><li>The model never sees raw values; the program never interprets language.</li>
<li>Every number carries the expression that produced it, and is cited like any passage.</li>
<li>What a plot cannot give (original counts, bin edges, uncertainties) is refused with a reason, never approximated.</li></ul>""")

    s = deck.slide("From the question to the answer")
    flow = [("Question", "text only"), ("Retrieve", "plot card + plot_link"), ("Plan", "LLM writes JSON steps"),
            ("Compute", "operations on the values"), ("Passage", "[C6] computed numbers"), ("Answer", "LLM cites [C6]")]
    width = 133
    for i, (title, body) in enumerate(flow):
        x = LEFT + i * (width + 15)
        s.card((x, 112, x + width, 190), title, body)
        if i:
            s.arrow((x - 13, 151), (x - 2, 151))
    s.table((LEFT, 210, RIGHT, 470), ["Stage", "Who", "Sees", "Produces"], [
        ["retrieve", "MiniLM, top-5, one plot per figure", "the question", "text and plot passages; each plot passage links its record and series handles"],
        ["plan", "the LLM (planner)", "plot passages with handles, axis names and units; the operation catalogue; the question",
         "a JSON plan, or a refusal when the plot cannot give the number"],
        ["compute", "<code>kamiknows.compute</code>", "the plan and only the series of the retrieved plots",
         "results with units and origins, or the exact error"],
        ["correct", "the planner again", "its plan and the error with the operation signature", "a corrected plan (up to 3 plans)"],
        ["answer", "the LLM", "all passages plus the computed passage", "the JSON answer, citing passages by label"]],
        [0.12, 0.17, 0.36, 0.35])

    s = deck.slide("The data: one object for every plot")
    s.table((LEFT, 112, 470, 265), ["Plot", "Domain", "Each element", "Value"], [
        ["curve, scatter", "x", "a point", "y"],
        ["1D histogram", "x", "a bin [lower, upper]", "y"],
        ["heatmap, 2D histogram", "x, y", "a cell", "z (colour)"]], [0.32, 0.16, 0.3, 0.22])
    s.html((LEFT, 270, 470, 470), """<ul class="small">
<li>A <b>Series</b> is a discrete function from a 1D or 2D domain to one value axis. Every operation takes and
returns Series, Points and Quantities, so any composition is again an operation.</li>
<li>Every axis has a <b>unit</b> and the <b>scale</b> it was drawn with (linear, log10). Values are always physical
numbers; the scale matters only to interpolate as drawn.</li>
<li>Arrays are checked finite and read-only; a centre must lie inside its bin.</li>
<li><code>compute/sources/plots.py</code> is the only module that reads <code>digitized_data.jsonl</code>. Tables and
equations will get their own adapter into the same Series.</li></ul>""")
    s.code((500, 112, RIGHT, 300), """
Series(series_id="<asset>/path_2",
       domain=(Axis("x", GeV, "linear"),),
       value_axis=Axis("y", A.U., "log10"),
       centers=[[45], [55], ...],       # n x d
       values=[0.005347, 0.004990, ...],
       lower=[[40], [50], ...],         # bins only
       upper=[[50], [60], ...],
       origin="<asset>/path_2",
       caveats=("values estimate the displayed figure ...",))""", "ONE SERIES (BLUE HISTOGRAM OF 1805.00850 FIG. 1)", size=8.6)
    s.html((500, 315, RIGHT, 470), """<div class="label">CATALOG</div>
<p class="small">Maps a handle <code>&lt;asset_id&gt;/&lt;series_id&gt;</code> to its Series, or to the reason there is
none (<i>plot not digitized (extraction status: needs_calibration)</i>). fastcalo: 322 series, 105 images without
values.</p>""")

    s = deck.slide("Units, unknowns and refusals")
    s.code((LEFT, 112, 450, 300), """
GeV / GeV          -> 1 (dimensionless)
GeV * GeV          -> GeV^2;   (GeV^2)^0.5 -> GeV
GeV/c^2 * c^2      -> GeV
2 GeV - 500 MeV    -> 1.5 GeV      (converted)
1 GeV + 1 A.U.     -> refused: different dimensions
log10(100 MeV)     -> refused: divide by 1 MeV first
unknown * MeV      -> unknown  (never becomes known)""", "UNIT ALGEBRA", size=9.2)
    s.html((LEFT, 315, 450, 470), """<ul class="small">
<li>Symbols outside the conversion table (<code>cell_index</code>, <code>A.U.</code>) are opaque units: they cancel,
but convert only to themselves.</li>
<li>An axis whose unit was never established is <code>unknown</code> (9 heatmaps of 2106.05285).</li></ul>""")
    s.html((480, 112, RIGHT, 300), """<div class="label">NOTCOMPUTABLE: ONE ERROR, WITH THE REASON</div>
<ul class="small"><li>the position is outside the plot or in a gap between bins</li>
<li>the plot was not digitized</li><li>incompatible units, or an operation that needs 1D on a 2D series</li>
<li>division by zero on a single number</li></ul>
<p class="small">The planner, the answer and the reviewer all see the same reason.</p>""")
    s.code((480, 315, RIGHT, 470), """
div(max(<blue>), max(<red>))
argmax(<Geant4 panel>, global)[0].x
crossing(<red>, <blue>, up)[0]
evaluate(<blue>, x=505)""", "ORIGIN: EVERY NUMBER CARRIES ITS EXPRESSION", size=9.2)

    s = deck.slide("The primitives: a basis in five families")
    s.table((LEFT, 112, RIGHT, 420), ["Family", "Operation", "What it does"], [
        ["selection", "<code>restrict</code>", "keep elements whose coordinate or value is in [lower, upper]"],
        ["", "<code>section</code>", "cut a 2D series at a fixed coordinate: a 1D series"],
        ["", "<code>coordinate</code>", "the coordinates (centre, lower or upper bound) as values: widths, moments, fits"],
        ["evaluation", "<code>evaluate</code>", "value at a position: the containing bin or cell, or interpolation as drawn"],
        ["", "<code>solve</code>", "where a 1D series crosses a level (up, down, any)"],
        ["algebra", "<code>combine</code>", "add, sub, mul, div of series, quantities and numbers, with units"],
        ["", "<code>transform</code>", "neg, abs, reciprocal, power, log10, exp10"],
        ["", "<code>component</code>", "one number of a point: a coordinate, a bound, or the value"],
        ["reduction", "<code>reduce</code>", "sum, integral, mean, min, max, count"],
        ["", "<code>project</code>, <code>argextremum</code>", "marginal of a 2D series; where the global or local extremes are"],
        ["calculus", "<code>differentiate</code>, <code>cumulate</code>", "slopes; running integral (exact for histograms)"]],
        [0.14, 0.27, 0.59])
    s.note("<code>describe</code> adds the metadata (units, scales, ranges, caveats). Operations that need 1D refuse a "
           "2D series and suggest <code>section</code> first: compositions cover them.", 430)

    s = deck.slide("Semantics that matter for plots")
    s.html((LEFT, 112, 450, 470), """<ul class="small">
<li><b>Histograms are not interpolated.</b> <code>evaluate</code> returns the bin that contains the position, with its
bounds: <code>x=505</code> gives the 500-510 GeV bin. A shared edge belongs to the upper bin; the last edge is closed.</li>
<li><b>Curves are interpolated as drawn</b>: straight segments in display space, so on a log10 axis the midpoint
of 1 and 100 is 10.</li>
<li><b>A step function crosses between two bins</b>: <code>solve</code> returns the bin it enters.</li>
<li><b><code>combine</code> of two series</b> works on the elements of the first: the second is evaluated at each
centre; elements where it is undefined are dropped and listed in the caveats.</li></ul>""")
    s.html((480, 112, RIGHT, 470), """<ul class="small">
<li><b>Ties are kept</b>: <code>argextremum</code> returns every element at the extreme. The CaloGAN first-layer map
has two cells at 2135 MeV; the passage says so.</li>
<li><b>Local extremes</b> (1D): a run of equal values counts once; the two ends are left to the global extreme.</li>
<li><b><code>crossing(a, b)</code></b> returns points of <code>a</code> itself: for histograms the bin of
<code>a</code> where the new order starts, with its value.</li>
<li><b><code>cumulate</code> of a histogram</b> is exact and piecewise linear, so <code>solve</code> on it gives exact
quantiles.</li></ul>""")

    s = deck.slide("Compositions: why it is a basis")
    s.table((LEFT, 112, RIGHT, 400), ["Quantity", "Written only with primitives (compute/derived.py)"], [
        ["crossing of two series", "<code>solve(combine(a, b, sub), 0, dir)</code>, then <code>evaluate(a, ...)</code>"],
        ["normalized series", "<code>combine(s, reduce(s, integral), div)</code>"],
        ["mean, standard deviation", "<code>coordinate</code>, <code>combine</code>, <code>reduce(integral)</code>, <code>transform(power)</code>"],
        ["quantile, median", "<code>solve(cumulate(s), q * reduce(s, integral), up)</code>"],
        ["full width at half maximum", "<code>argextremum</code>, then <code>solve(s, max / 2)</code> on both sides of the peak"],
        ["least-squares line", "sums of <code>x</code>, <code>y</code>, <code>x²</code>, <code>xy</code> from <code>coordinate</code>, <code>combine</code>, <code>reduce</code>"],
        ["distance between two points", "<code>transform(combine(component(p, x), component(q, x), sub), abs)</code>"]],
        [0.3, 0.7])
    s.note("A test builds a Gaussian histogram and recovers its mean, standard deviation, median and FWHM from these "
           "compositions alone. A question that needs a new function points to a missing primitive, not to a special case.", 415)

    s = deck.slide("Plans: JSON steps with references")
    s.code((LEFT, 112, 470, 330), """
{"steps": [
  {"id": "window", "op": "restrict",
   "args": {"series": "<asset>/path_2", "axis": "x",
            "lower": 250, "upper": 400}},
  {"id": "maxima", "op": "argextremum",
   "args": {"series": "$window", "kind": "max",
            "scope": "local"}}],
 "result": ["$maxima"]}

{"not_computable": "original event counts are not
                    shown by a normalized plot"}""", "A PLAN, AND A REFUSAL", size=8.8)
    s.html((LEFT, 345, 470, 470), """<p class="small">An argument is a series handle, a number in the axis unit,
<code>{"value": 250, "unit": "GeV"}</code>, or a reference: <code>$a</code>, <code>$a[0]</code> (one of several points),
<code>$a.x</code> / <code>$a.value</code>, <code>$a.slope</code> (a record field).</p>""")
    s.table((500, 112, RIGHT, 340), ["Outcome", "Meaning"], [
        ["<code>computed</code>", "results rendered with units, origin and a short text"],
        ["<code>declined</code>", "the plan itself says the plot cannot give the number"],
        ["<code>not_computable</code>", "an operation refused, with the reason and the failed step"],
        ["<code>invalid_plan</code>", "malformed plan: unknown operation, bad argument, bad reference (the planner's fault)"]],
        [0.33, 0.67])
    s.html((500, 350, RIGHT, 470), '<p class="small muted"><code>kamiknows/compute/plan.py</code> validates the plan, '
           'resolves references, checks every argument against <code>registry.py</code> and collects the caveats of '
           'every series it touched. Keeping the four outcomes apart separates planner errors from data limits.</p>')

    s = deck.slide("The catalogue the planner sees")
    s.code((LEFT, 112, RIGHT, 395), """
restrict(series: series, axis: axis, lower?: quantity, upper?: quantity) -> series: Keep the elements whose ...
evaluate(series: series, at: coords) -> point: The series at a position given for every domain axis.
solve(series: series, level: quantity, direction?: up|down|any) -> points: Where a 1D series crosses level ...
combine(left: operand, right: operand, op: add|sub|mul|div) -> series|quantity: Apply add, sub, mul or div ...
component(point: point, axis: axis, which?: center|lower|upper) -> quantity: One number of a point: value, ...
reduce(series: series, how: sum|integral|mean|min|max|count) -> quantity: Reduce a series to one number.
argextremum(series: series, kind?: max|min, scope?: global|local) -> points: Where a series reaches its maximum ...
describe(series: series) -> record: Axes (unit, scale, range), number of elements, whether they are bins, ...
crossing(first: series, second: series, direction?: up|down|any) -> points: Points of first where it crosses ...
quantile(series: series, q: number) -> quantity: Coordinate below which a fraction q of the integral lies ...
fwhm(series: series) -> quantity: Full width at half maximum of the global peak of a 1D series.
... 22 operations in all""", "registry.catalogue() (shortened)", size=8.4)
    s.note("Each line is generated: the signature from <code>registry.py</code>, the description from the first line of "
           "the function's docstring, the choices from the constants the function checks. Documentation and catalogue "
           "cannot drift apart; a test checks that every signature matches its function.", 405)

    s = deck.slide("Checking the math without a model")
    s.code((LEFT, 112, 500, 215), """
python scripts/compute.py \\
  --questions data/questions/fastcalo_plots_values_plans.jsonl \\
  --digitized-dir $DIGITIZED --output-dir outputs/compute_...""", "COMMAND (CPU, NO MODEL)", size=9)
    s.html((LEFT, 230, 500, 470), """<ul class="small">
<li>Each of the 20 plot questions has a hand-written <code>expected_plan</code> and the
<code>expected_results</code> taken from its reference answer (not from running the code).</li>
<li>The step runs every plan and compares the results within 1% (only the keys given are checked).</li>
<li>Result: <b>18 computed + 2 declined, 20 / 20 match</b>. A later wrong answer is then the planner's
fault, not the operations' or the data's.</li>
<li>It also found a tie the reference answer ignores: two CaloGAN cells at 2135 MeV.</li></ul>""")
    s.code((530, 112, RIGHT, 470), """
{"question_id": "plot_08_referenced",
 "question": "... in which bin does the red
   training-weighted histogram first rise
   above the blue unweighted one?",
 "expected_plan": {"steps": [
   {"id": "first", "op": "crossing",
    "args": {"first": "<red>", "second": "<blue>",
             "direction": "up"}},
   {"id": "red", "op": "evaluate", "args":
    {"series": "<red>", "at": {"x": "$first[0].x"}}},
   {"id": "blue", "op": "evaluate", "args":
    {"series": "<blue>", "at": {"x": "$first[0].x"}}}],
  "result": ["$red", "$blue"]},
 "expected_results": {"status": "computed",
   "results": [{"coords": {"x": {"lower": 190,
     "upper": 200}}, "value": {"value": 0.001657}},
               ...]}}""", "ONE ANNOTATED QUESTION (shortened)", size=8.2)

    s = deck.slide("What retrieval hands over: the plot card")
    s.code((LEFT, 112, 500, 440), """
Plot: Figure 1 of arXiv:1805.00850v2, single image.
Caption: Distribution of the jet transverse momentum at
particle level for the hadronic jets in the input
dataset. The blue histogram shows the distribution as
coming from the dataset, while the red one shows the
distribution obtained after applying the weights ...
Paper discussion of this figure: The distribution of
p^gen_T for the selected hadronic jets is shown by the
blue histogram in figure [REF]. ...
Type: 1D histogram (drawn steps); 2 series.
Axes: X in GeV (linear), Y in A.U. (log10).
- blue curve: 246 bins; X from 40 to 2540 GeV; maximum
  0.005347 A.U. at X = 45 GeV.
- red curve: 248 bins; X from 40 to 2540 GeV; maximum
  0.003345 A.U. at X = 45 GeV.
Values estimate the displayed figure, not the original
measurements; original bins, counts and uncertainties
are not available.""", "THE TEXT (RANKED AND SHOWN)", size=8.4)
    s.code((520, 112, RIGHT, 330), """
"plot_link": {
  "asset_id": "1805_00850v2::figure::0001::asset::01",
  "data_id": "digitized_ce3f...",
  "extraction_status": "success",
  "digitized_data_sha256": "df5e...",
  "series": [
    {"handle": "<asset>/path_2", "label": "blue curve",
     "usable": true},
    {"handle": "<asset>/path_3", "label": "red curve",
     "usable": true}],
  "figure_panels": [...]}       # multi-panel figures""", "THE LINK (METADATA)", size=8.4)
    s.html((520, 340, RIGHT, 470), """<ul class="small">
<li>One plotted image = one chunk = one digitized record; plots without values get a chunk saying why.</li>
<li>Ranges and maxima come from <code>compute</code>, so card and answer agree.</li>
<li>Handles stay out of the text: they are ids, noise for the embedding.</li></ul>""")

    s = deck.slide("The planner prompt")
    s.html((LEFT, 112, 430, 470), """<ol class="small" style="margin-left:14pt">
<li><b>Task</b>: plan, do not calculate. The program holds every value of every listed series although the
passages show only a summary: never decline because a value is not printed.</li>
<li><b>Three replies</b>: a plan; <code>not_computable</code> (what a plot cannot show); <code>no_computation</code>
(e.g. axis units already in the card).</li>
<li><b>Arguments and references</b>, then the generated <b>catalogue</b> and two generic examples.</li>
<li><b>The plot passages</b>, each with its handles, axis names and units, and the other panels of its figure.</li>
<li><b>The question</b>, restated at the end.</li></ol>
<p class="small muted">Only the handles of retrieved plot chunks can be used: the plan runs on a catalog restricted
to them.</p>""")
    s.code((450, 112, RIGHT, 470), """
[C4] Figure 1, bottom-left panel (arXiv:1705.02355v2)
Plot: Figure 1 of arXiv:1705.02355v2, image 4 of 6.
Panel position: bottom-left panel ...
Caption: Average Geant4 shower (top), and average
CaloGAN shower (bottom), with progressive
calorimeter depth (left to right). ...
Series handles:
- heatmap cells: <asset 04>/panel_1_cells (domain
  axes: x in cell_index, y in cell_index; value
  axis: z in MeV)
Other panels of this figure:
- top-left panel, heatmap cells: <asset 01>/... (...)
- top-middle panel, heatmap cells: <asset 02>/...
- ...
Write the JSON object for this question:
In Figure 1 of arXiv:1705.02355v2, compare ...""", "PLOT PASSAGES AS THE PLANNER SEES THEM (shortened)", size=8.2)

    s = deck.slide("When a plan fails: correction")
    s.code((LEFT, 112, RIGHT, 300), """
Q: At approximately what particle-level transverse momentum does the unweighted hadronic-jet
   distribution peak, and how high is that peak?
plan 1: argextremum(<blue>) -> p;  component(p[0], x);  component(p[0], "y")
        -> not_computable: argmax(<blue>, global)[0] has no coordinate 'y': its coordinates are ['x'],
           and its value is axis 'value'
plan 2: argextremum(<blue>) -> p;  component(p[0], x);  component(p[0], "value")
        -> computed: 45 GeV; 0.00534672 A.U.""", "QWEN3 4B, A NATURAL QUESTION", size=8.8)
    s.html((LEFT, 315, 460, 470), """<ul class="small">
<li>A malformed plan or a refused operation goes back to the planner with its own reply and the exact error;
argument errors include the operation signature.</li>
<li>Up to 3 plans. A correction must be a plan: a refusal after an error is sent back too.</li>
<li>A refusal at the first reply is accepted (the two event-count questions).</li></ul>""")
    s.code((490, 315, RIGHT, 470), """
argextremum: kind must be one of the listed
words (max, min), got 'localmax'; signature:
argextremum(series: series, kind?: max|min,
  scope?: global|local) -> points""", "AN ARGUMENT ERROR AS RETURNED", size=8.6)

    s = deck.slide("The computed passage and the answer")
    s.code((LEFT, 112, RIGHT, 285), """
[C6] Computed from the extracted plot data
Numbers computed exactly by a program from the extracted plot data (deterministic operations, not
estimated by a language model):
1. 47.5439 cell_index (computed as argmax(arXiv:1705.02355v2 Figure 1, top-left panel, heatmap cells, global)[0].x)
2. 46.9951 cell_index (computed as argmax(arXiv:1705.02355v2 Figure 1, bottom-left panel, heatmap cells, ...)[0].x)
3. -0.548788 cell_index (computed as sub(...))
Note: step c_peak_x found 2 points (ties or several crossings).
Caveats: values estimate the displayed figure, not the original measurements; ... read from the colour scale.""",
           "PLOT PAIR 2, FIGURE REFERENCE", size=8.4)
    s.code((LEFT, 300, 520, 470), """
"The brightest-cell eta position in the first-layer
Geant4 photon-shower map (top-left panel) is at 47.54
cell_index, and in the first-layer CaloGAN map
(bottom-left panel) it is at 46.995 cell_index. The
difference ... is within one cell-index unit." [C6]""", "QWEN3 4B ANSWER", size=8.6)
    s.html((545, 300, RIGHT, 470), """<ul class="small">
<li>Handles become readable names (figure, panel, series label) in the passage.</li>
<li>The answer prompt adds: use the computed numbers for numerical claims, cite them with the plot passage,
and say so when the number could not be computed.</li>
<li>The citation check treats <code>[C6]</code> like any retrieved passage.</li></ul>""")

    s = deck.slide("Results on the 20 plot questions")
    s.table((LEFT, 112, RIGHT, 330), ["Configuration (20 plot questions)", "Qwen3 4B", "Mistral 7B"], RESULTS_ROWS,
            [0.52, 0.24, 0.24])
    s.html((LEFT, 340, RIGHT, 470), RESULTS_NOTE)

    s = deck.slide("What still fails")
    s.html((LEFT, 112, 450, 470), """
<div class="h">Planner</div>
<ul class="small">
<li>A valid plan for the wrong quantity (the global maxima instead of the value at 505 GeV), or the right
number without what the answer needs (a crossing coordinate without its bin): the answer model abstains.</li>
<li>The wrong axis: cards give units but not axis titles (η, φ), so "along η" can become <code>y</code>.</li>
<li>Many similar plots in the context: declines although the right plot is there, the wrong panel of the
right figure, or a correct computed number that the answer model does not trust.</li>
<li>Mistral 7B rarely writes a valid plan within 3 attempts, and still invents values when nothing is computed.</li></ul>""")
    s.html((480, 112, RIGHT, 470), """
<div class="h">Elsewhere</div>
<ul class="small">
<li><b>Retrieval</b>: with one plot chunk per figure the required series is reached for 19 / 20 questions;
one natural question still gets another figure of the right paper.</li>
<li><b>Coverage</b>: 119 of the 224 plotted images have values; the others can only be refused.</li>
<li><b>Evaluation</b>: 20 questions, inspected by hand; one or two questions are within noise. Expert review and a
larger question set with plans are next.</li></ul>""")

    s = deck.slide("Commands and files")
    s.code((LEFT, 112, RIGHT, 300), """
DIGITIZED=outputs/plots_fastcalo_...
python scripts/chunk.py sentence --parsed-dir $PARSED --exclude-backmatter --digitized-dir $DIGITIZED --output-dir $CHUNKS
python scripts/retrieve.py --dataset-dir $CHUNKS --questions data/questions/fastcalo_plots_values.jsonl \\
       --retriever dense --source-policy explicit_arxiv --top-k 5 --output-dir $RETRIEVAL
sbatch scripts/run_ollama.slurm scripts/generate.py --retrieval-dir $RETRIEVAL --output-dir $ANSWERS \\
       --model qwen3:4b-instruct --solver
python scripts/compute.py --questions data/questions/fastcalo_plots_values_plans.jsonl \\
       --digitized-dir $DIGITIZED --output-dir outputs/compute_plotplans_expected""", size=8.8)
    s.table((LEFT, 315, RIGHT, 470), ["File", "Role"], [
        ["<code>kamiknows/compute/</code> types, units, select, evaluate, algebra, reduce, calculus", "Series and the 13 primitives"],
        ["<code>derived.py</code>, <code>metadata.py</code>, <code>registry.py</code>, <code>plan.py</code>", "compositions, describe, catalogue, plan executor"],
        ["<code>sources/plots.py</code>, <code>catalog.py</code>, <code>pipeline.py</code>", "digitized data in, handles, the compute step"],
        ["<code>kamiknows/generation.py</code> (<code>--solver</code>)", "planner prompt, correction, computed passage"]],
        [0.6, 0.4])
    return deck


def main() -> int:
    pages = overview().save(ROOT / "KamiKnows_pipeline.pdf")
    print(f"KamiKnows_pipeline.pdf: {pages} pages")
    pages = plot_extraction().save(ROOT / "docs/plot_extractor/KamiKnows_plot_extraction.pdf")
    print(f"docs/plot_extractor/KamiKnows_plot_extraction.pdf: {pages} pages")
    pages = compute().save(ROOT / "docs/compute/KamiKnows_compute.pdf")
    print(f"docs/compute/KamiKnows_compute.pdf: {pages} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
