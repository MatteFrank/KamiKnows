"""Build the two KamiKnows slide decks from this file.

    python docs/slides/build_slides.py

Writes KamiKnows_pipeline.pdf (repository guide, at the repository root) and
docs/plot_extractor/KamiKnows_plot_extraction.pdf (plot extraction walkthrough).
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
DATE = "17/11/2026"
AUTHORS = "E. Apicella, F. Corchia, M. Franchini"


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
scripts/          one command per step
  parse.py  chunk.py  retrieve.py
  generate.py  review.py  chat.py
  extract_plots.py  (plot extraction modes)
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
extraction. The chat reuses the same functions as the benchmark.</p><br>
<div class="h">outputs/ = what each run produced</div>
<p class="muted">Never versioned, never overwritten. <code>outputs/INDEX.md</code> lists the local
runs; versioned files never refer to them.</p>""")

    s = deck.slide("The pipeline: five steps, five commands")
    steps = [("1. Parse", "parse.py", "Download LaTeX; extract sections, equations and figure files.", "sections.json"),
             ("2. Chunk", "chunk.py", "Split text into passages; optionally add plot values.", "chunks.jsonl"),
             ("3. Retrieve", "retrieve.py", "Top-k passages per question: TF-IDF or dense.", "contexts.jsonl"),
             ("4. Generate", "generate.py", "The model answers from the saved passages and cites them.", "answers.jsonl"),
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
           '<div class="pre">figure files -> inventory -> classify -> calibrate -> digitize -> chunk.py --digitized-dirs</div>')
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
python scripts/extract_plots.py inventory --parsed-dir $P --output-dir $I
sbatch scripts/run_ollama.slurm scripts/extract_plots.py classify --classifier vision \\
  --model gemma3:4b --inventory-dir $I --parsed-dir $P --output-dir $V
python scripts/extract_plots.py calibrate --classified-dir $V --inventory-dir $I --parsed-dir $P --output-dir $A
python scripts/extract_plots.py digitize --classified-dir $V --inventory-dir $I --parsed-dir $P \\
  --calibration-file $A/calibration_proposals.json --output-dir $D
python scripts/chunk.py sentence --parsed-dir $P --exclude-backmatter --digitized-dirs $D --output-dir $C""",
           size=9.4)
    s.note("The values join the chunks as text with units, scales and provenance: the generator stays "
           "text-only. Full walkthrough: <code>docs/plot_extractor/KamiKnows_plot_extraction.pdf</code>.", 400)

    s = deck.slide("Step 2 · Chunk")
    s.table((LEFT, 112, 470, 300), ["Chunker", "How it cuts"], [
        ["<code>word</code>", "Fixed windows of <code>--target-words</code> (400), overlap <code>--overlap-words</code> (60). "
                              "A final tail under 80 words joins the previous window."],
        ["<code>sentence</code>", "Packs whole sentences up to the target; the overlap is made of whole sentences."],
        ["<code>semantic</code>", "Like sentence, but cuts where the embedding similarity between neighbouring "
                                  "sentences drops (a topic change). See next slide."]], [0.22, 0.78])
    s.html((LEFT, 315, 470, 460), '<p class="muted">Chunks never cross sections. <code>--exclude-backmatter</code> '
           'drops acknowledgments and bibliography (logged in <code>section_exclusions.jsonl</code>).</p>'
           '<p class="muted"><code>--digitized-dirs</code> adds plot chunks (series summaries, coordinate tables, '
           'figure overviews) to any chunker.</p>')
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
           'length_limit, section_end, plot_summary, ...), <code>equation_ids</code>, <code>flags</code>. '
           'Plot chunks also carry <code>plot_provenance</code> and <code>axis_metadata</code>.</p>')

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
           'a retrieved panel brings its figure overview along.</p>')
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
<li>Plot values arrive as text passages; no multimodal model is needed.</li></ul>""")
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
    s.code((590, 112, RIGHT, 245), """
srun --account=kamiknows \\
  --qos=kamiknows --gres=gpu:1 \\
  --partition=bare-metal-GPU --pty bash
# start the shared Ollama server, then
python scripts/chat.py \\
  --dataset-dir outputs/chunks_... \\
  --model qwen3:4b-instruct""", "INTERACTIVE CHAT", size=9.2)
    s.html((590, 260, RIGHT, 460), '<p class="muted small">Same retrieval and generation as the benchmark. Each '
           'question is independent; every turn is saved in <code>outputs/chat/</code>.</p>')

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
calibrations_<corpus>_... | plots_<corpus>_...""", "NAMING PATTERN", size=9)
    s.table((480, 255, RIGHT, 470), ["Tag", "Meaning"], [
        ["<code>fd100</code>", "cut policy first_drop, at least 100 words per chunk"],
        ["<code>nobm</code>", "no back matter (acknowledgments, bibliography)"],
        ["<code>dense</code>", "embedding retriever (MiniLM); <code>tfidf</code> lexical"],
        ["<code>explicit</code>", "source policy explicit_arxiv; <code>allsrc</code> = none"],
        ["<code>paired</code> / <code>plots</code>", "questions file fastcalo_paired.jsonl / fastcalo_plots.jsonl"]],
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
           '<p><code>fastcalo_plots.jsonl</code>: <b>12 questions</b> on plot values, in 6 pairs. One variant cites '
           'the figure, the other is a natural question with no figure or arXiv id. 10 answerable, 2 to refuse.</p>')
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
P=outputs/parsed_fastcalo  C=outputs/chunks_fastcalo_semantic_fd100_nobm
R=outputs/retrieval_semantic_fd100_nobm_dense_explicit_paired_top5
Q=outputs/qa_semantic_fd100_nobm_dense_explicit_qwen4b_paired_top5

python scripts/parse.py    --ids-file data/input/arxiv_ids_fastCalo_5fulltext.txt --output-dir $P
python scripts/chunk.py semantic --parsed-dir $P --output-dir $C \\
       --exclude-backmatter --cut-policy first_drop --min-words 100 --plots
python scripts/retrieve.py --dataset-dir $C --questions data/questions/fastcalo_paired.jsonl \\
       --output-dir $R --retriever dense --top-k 5
sbatch scripts/run_ollama.slurm scripts/generate.py --retrieval-dir $R --output-dir $Q \\
       --model qwen3:4b-instruct
python scripts/review.py prepare --qa-dirs $Q --output-dir outputs/review_qwen4b_paired
python scripts/index_outputs.py

pytest -q                       # offline tests of every step""", size=10)
    s.note("Full reference: <code>README.md</code>. Plot extraction: <code>docs/plot_extractor/</code>. "
           "Rules for coding agents: <code>AGENTS.md</code>. Every script: <code>--help</code>.", 378)
    return deck


def plot_extraction() -> Deck:
    deck = Deck(f"KamiKnows - plot extraction | {DATE}")
    deck.cover("Kami", "Knows", "Plot extraction: from paper figures<br>to cited numerical evidence.",
               "How figures become passages that the text pipeline can retrieve and cite.", AUTHORS,
               f"{DATE}, Plot extraction guide", ASSETS / "kami_logo.png")

    s = deck.slide("Why plot extraction")
    s.html((LEFT, 112, 450, 330), """
<p>Many answers in a physics paper are only in a figure: where a distribution peaks, how two curves
compare, which axis is logarithmic. A text-only RAG cannot see them.</p>
<p><b>Idea:</b> read the numbers out of the original figure files once, turn them into text passages
with units and provenance, and let the normal pipeline retrieve and cite them.</p>
<ul><li>The generator (Qwen, Mistral) stays text-only.</li>
<li>Every value points back to a figure file, a series and a hash.</li>
<li>What cannot be read is reported, never invented.</li></ul>""")
    s.box((480, 112, RIGHT, 330), "#f7f4ef", "#e6ddd0")
    s.label(495, 122, RIGHT, "EXAMPLE QUESTION")
    s.html((495, 142, RIGHT - 12, 325), '<p>"At approximately what particle-level transverse momentum does the '
           'unweighted hadronic-jet distribution peak, and how high is that peak?"</p>'
           '<p class="muted small">The text says the distribution is "multi-modal" but gives no number. '
           'The values exist only in Figure 1 of arXiv:1805.00850.</p>'
           '<p class="small"><b>Qwen3 4B with plot chunks:</b> "The unweighted hadronic-jet distribution peaks at a '
           'transverse momentum of 45 GeV. The peak height is 0.005346717 (in arbitrary units)." [C3, C5]</p>')
    s.note("In an earlier text-only comparison on the same papers, Qwen abstained on every plot question: "
           "the numbers were simply not in the passages.", 350)

    s = deck.slide("The chain: one CLI, one run directory per step")
    chain = [("inventory", "CPU"), ("classify", "GPU"), ("calibrate", "CPU"), ("digitize", "CPU"),
             ("chunk", "CPU"), ("retrieve", "CPU"), ("generate", "GPU")]
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
P=outputs/parsed_fastcalo                     # existing parse run: no new download
I=outputs/figures_fastcalo_inventory_v1  V=outputs/figures_fastcalo_classified_vision_v1
A=outputs/calibrations_fastcalo_auto_v1  D=outputs/plots_fastcalo_auto_v1
python scripts/extract_plots.py inventory --parsed-dir $P --output-dir $I
sbatch scripts/run_ollama.slurm scripts/extract_plots.py classify --classifier vision --model gemma3:4b \\
  --inventory-dir $I --parsed-dir $P --output-dir $V
python scripts/extract_plots.py calibrate --classified-dir $V --inventory-dir $I --parsed-dir $P --output-dir $A
python scripts/extract_plots.py digitize --classified-dir $V --inventory-dir $I --parsed-dir $P \\
  --calibration-file $A/calibration_proposals.json --output-dir $D""", size=9.2)
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
  --inventory-dir $I --parsed-dir $P --output-dir $V""", "COMMAND (GPU)", size=9)

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
        ["Calibration, all figure files (last full automatic run)", "224", "182 proposals; the rest flagged for review"],
        ["Digitization, hybrid 2D maps (automatic)", "76", "54 successes: 2D maps from 1705.02355, 1712.10321, 2106.05285"],
        ["Digitization, Figure 1 of 1805.00850", "1", "2 step-histogram series (blue, red)"],
        ["<b>In the chunks used for the questions</b>", "55", "<b>56 series</b>, 451 plot chunks"]],
        [0.45, 0.12, 0.43])
    s.html((585, 112, RIGHT, 470), """
<div class="h">Current coverage</div>
<ul class="small">
<li>The automatic digitization was run on the hybrid 2D maps and on one 1D figure, not yet on all 182
calibrated files.</li>
<li>The questions use only these 56 series.</li>
<li>Extending the run to every proposal, and checking the result in the visual report, is the next step.</li>
</ul>""")
    s.note("Counts from the fastcalo corpus (5 papers). Values are estimates of what is displayed; manual review "
           "pending.", 360)

    s = deck.slide("5 · Into the chunks")
    s.code((LEFT, 112, 450, 205), """
python scripts/chunk.py sentence \\
  --parsed-dir $P --exclude-backmatter \\
  --digitized-dirs $D [$D2 ...] \\
  --output-dir outputs/chunks_fastcalo_sentence_plots_v1""", "COMMAND", size=9.2)
    s.html((LEFT, 215, 450, 470), """<ul class="small">
<li><b>Summary</b> per series: figure, image, colour, caption, the paper sentences that cite the figure,
scales, units, peak and range.</li>
<li><b>Coordinate tables</b> with the values, split to fit the word budget.</li>
<li><b>Figure overview</b>: the peak of every panel, so that sibling panels can be compared.</li>
<li>Each chunk carries <code>plot_provenance</code> (asset, series, hashes) and <code>axis_metadata</code>
(unit, unit status, scale, multiplier).</li>
<li>Only successful series are added; hash mismatches stop the chunker.</li></ul>""")
    s.code((470, 112, RIGHT, 470), """
Digitized plot: Figure 1, source image 1 of 1; blue curve.
Source file: extracted/gen_pt.pdf. Caption: Distribution of
the jet transverse momentum at particle level for the hadronic
jets in the input dataset. The blue histogram shows the
distribution as coming from the dataset, while the red one
shows the distribution obtained after applying the weights
used for training.
Geometry: rendered_step_intervals. Series appearance: blue
curve. Dataset association requires caption or legend evidence.
Values estimate displayed geometry, not original measurements.
Paper discussion of this figure: The distribution of p^gen_T
for the selected hadronic jets is shown by the blue histogram
in figure [REF]. ...
X scale: linear; unit: GeV; unit status: explicit.
Y scale: log10; unit: A.U.; unit status: explicit.
Peak: x=45; y=0.005346717. ...""", "SUMMARY CHUNK (shortened)", size=8.4)

    s = deck.slide("6 · Retrieval: plots and text in one index")
    s.html((LEFT, 112, 470, 470), """<ul>
<li>Plot chunks are ranked together with paper text, using the question text only.</li>
<li>When a panel is retrieved, its <b>figure overview</b> is attached once, so a question about two panels
sees both.</li>
<li><code>expected_plot_sources</code> in the questions file lists the asset/series that must be retrieved.
The summary reports it separately from paper hit@k: finding the right paper is not enough.</li></ul>""")
    s.table((500, 112, RIGHT, 260), ["Dense MiniLM, top-5, fastcalo_plots.jsonl", "Result"], [
        ["questions with the expected paper retrieved", "12 / 12"],
        ["required plot series retrieved, figure-reference questions", "6 / 6"],
        ["required plot series retrieved, natural questions", "6 / 6"]], [0.75, 0.25])
    s.code((500, 275, RIGHT, 440), """
plot_03_natural
C1 1805_00850v2  text  Inputs and problem formulation
C2 1805_00850v2  text  Inputs and problem formulation
C3 1805_00850v2  plot  Figure 1, red curve: summary
C4 1805_00850v2  text  Results
C5 1805_00850v2  plot  Figure 1, blue curve: summary""", "RETRIEVED PASSAGES", size=8.8)

    s = deck.slide("The plot questions: fastcalo_plots.jsonl")
    s.html((LEFT, 112, 440, 300), """<ul class="small">
<li>6 pairs, 12 questions. In each pair the <b>figure-reference</b> variant cites figure and arXiv id; the
<b>natural</b> variant names neither, and searches the whole corpus.</li>
<li>10 are answerable from the extracted values; 2 ask for original event counts, which a normalized figure
cannot give, and must be refused.</li>
<li>Reference answers check faithfulness to the extracted values, not scientific accuracy.</li></ul>""")
    s.code((470, 112, RIGHT, 470), """
{"question_id": "plot_03_natural",
 "pair_id": "plot_pair_03",
 "question_style": "natural",
 "question_type": "plot_data",
 "question": "At approximately what particle-level
   transverse momentum does the unweighted hadronic-jet
   distribution peak, and how high is that peak?",
 "expected_behavior": "answerable",
 "expected_arxiv_ids": ["1805.00850v2"],
 "expected_plot_sources": [
   {"asset_id": "1805_00850v2::figure::0001::asset::01",
    "series_id": "path_2"}],
 "reference_answer": "The maximum displayed Y is
   approximately 0.005347 A.U. at x=45 GeV; original
   measurement bins are not verified."}""", "ONE QUESTION (shortened)", size=8.8)
    s.table((LEFT, 310, 440, 470), ["Field", "Role"], [
        ["<code>question</code>", "the only text used for ranking and in the prompt"],
        ["<code>expected_*</code>, <code>reference_answer</code>", "evaluation and review only"]], [0.5, 0.5])

    s = deck.slide("The six pairs")
    s.table((LEFT, 112, RIGHT, 470), ["Pair", "Figure", "Asks for", "Reference answer"], [
        ["1", "1705.02355 Fig. 1, Geant4 layer 1", "brightest cell: energy and position", "≈2137 MeV at x=47.54, y=1.00 (cell index)"],
        ["2", "same, Geant4 vs CaloGAN", "are the two peaks within one cell along η?", "47.544 vs 46.995: 0.549 apart, yes"],
        ["3", "1805.00850 Fig. 1, unweighted", "peak position and height", "x=45 GeV, y≈0.005347 A.U."],
        ["4", "same, two histograms", "ratio of the two peak heights", "0.005347 / 0.003345 ≈ 1.60"],
        ["5", "same", "axis scales and units", "X linear in GeV, Y log10 in A.U."],
        ["6", "same", "exact number of original events in the top bin", "<b>refuse</b>: not recoverable from a normalized plot"]],
        [0.06, 0.27, 0.32, 0.35])

    s = deck.slide("How the models answer")
    s.table((LEFT, 112, RIGHT, 285), ["Grounded prompt, same retrieved passages", "Qwen3 4B", "Mistral 7B"], [
        ["answered / abstained", "7 / 5", "12 / 0"],
        ["answerable questions answered correctly (inspection)", "6 of 10", "4 of 10 (ratios ≈1.57-1.58)"],
        ["count questions correctly refused", "2 of 2", "0 of 2"],
        ["citation labels valid", "12 of 12", "12 of 12"]], [0.6, 0.2, 0.2])
    s.html((LEFT, 300, RIGHT, 410), """<ul class="small">
<li><b>Qwen</b> refuses when unsure: correct on both count controls, but also refuses 3 answerable questions
(two of them natural queries). One answer uses the CaloGAN panel for a Geant4 question.</li>
<li><b>Mistral</b> always answers: it says the two peaks are <i>not</i> within one cell, calls the log axis linear,
and invents an event count ("1805", the arXiv number).</li>
<li>Correct numbers in the context are necessary, not sufficient: the model must still pick the right panel,
do the arithmetic and know when to refuse.</li></ul>""")
    s.note("Correctness counts come from implementation inspection, not from the expert review "
           "(<code>review.csv</code> still pending). Valid citation labels do not prove support.", 420)

    s = deck.slide("Answers, good and bad")
    s.code((LEFT, 112, 470, 290), """
Q (natural): ratio between the unweighted and
training-weighted peak heights?
Qwen:    insufficient_evidence
Mistral: "approximately 1.57"  [C1, C4, C5]
Reference: 0.005347 / 0.003345 = 1.599""", "PLOT PAIR 4", size=9)
    s.code((490, 112, RIGHT, 290), """
Q (figure reference): ratio of the blue peak
to the red peak?
Qwen: "0.005346717 / 0.003344805 = 1.601" [C1, C2]
Mistral: "approximately 1.58" [C1, C4]""", "PLOT PAIR 4, OTHER VARIANT", size=9)
    s.code((LEFT, 250, 470, 470), """
Q: exactly how many original unweighted events
contributed to the highest bin?
Qwen:    insufficient_evidence          (correct)
Mistral: "... is 1805."  [C3]           (invented)""", "PLOT PAIR 6 (MUST REFUSE)", size=9)
    s.code((490, 250, RIGHT, 470), """
Q (natural): energy and position of the brightest
cell, Geant4 shower, first layer?
Qwen: "2134.985 MeV at x=46.99513, y=0.9955"  [C5]
Reference: 2137 MeV at x=47.54 (Geant4 panel)
-> the cited passage is the CaloGAN panel""", "PLOT PAIR 1 (WRONG PANEL)", size=9)

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
<li>Expert review of the 24 answers in <code>review.csv</code>.</li>
<li>More plot questions, including other papers and 1D curves.</li>
<li>Compare against original data where it exists (<code>recover</code>: PGFPlots tables, HEPData;
<code>benchmark</code>).</li></ul>""")

    s = deck.slide("All the commands")
    s.code((LEFT, 112, RIGHT, 470), """
P=outputs/parsed_fastcalo  I=outputs/figures_fastcalo_inventory_v1  V=outputs/figures_fastcalo_classified_vision_v1
A=outputs/calibrations_fastcalo_auto_v1  D=outputs/plots_fastcalo_auto_v1  C=outputs/chunks_fastcalo_sentence_plots_v1
R=outputs/retrieval_sentence_plots_dense_explicit_plots_top5  Q=outputs/qa_sentence_plots_dense_explicit_qwen4b_plots_top5
python scripts/extract_plots.py inventory --parsed-dir $P --output-dir $I
sbatch scripts/run_ollama.slurm scripts/extract_plots.py classify --classifier vision --model gemma3:4b \\
       --inventory-dir $I --parsed-dir $P --output-dir $V
python scripts/extract_plots.py calibrate --classified-dir $V --inventory-dir $I --parsed-dir $P --output-dir $A
python scripts/extract_plots.py digitize --classified-dir $V --inventory-dir $I --parsed-dir $P \\
       --calibration-file $A/calibration_proposals.json --output-dir $D
python scripts/extract_plots.py digitization-review --digitized-dir $D --output-dir outputs/plots_fastcalo_visual_v1
python scripts/chunk.py sentence --parsed-dir $P --exclude-backmatter --digitized-dirs $D --output-dir $C
python scripts/retrieve.py --dataset-dir $C --questions data/questions/fastcalo_plots.jsonl \\
       --retriever dense --source-policy explicit_arxiv --top-k 5 --output-dir $R
sbatch scripts/run_ollama.slurm scripts/generate.py --retrieval-dir $R --output-dir $Q --model qwen3:4b-instruct
python scripts/review.py prepare --qa-dirs $Q --output-dir outputs/review_fastcalo_plots""", size=8.6)
    s.note("Guide: <code>docs/plot_extractor/guide.md</code>. Results: <code>docs/plot_extractor/qa_results.md</code>. "
           "Options: <code>python scripts/extract_plots.py &lt;mode&gt; --help</code>.", 400)
    return deck


def main() -> int:
    pages = overview().save(ROOT / "KamiKnows_pipeline.pdf")
    print(f"KamiKnows_pipeline.pdf: {pages} pages")
    pages = plot_extraction().save(ROOT / "docs/plot_extractor/KamiKnows_plot_extraction.pdf")
    print(f"docs/plot_extractor/KamiKnows_plot_extraction.pdf: {pages} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
