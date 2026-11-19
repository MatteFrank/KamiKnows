# Plot QA results

What the plot questions currently show, on the fastcalo corpus. These are
observations from inspecting the answers; the expert review is still pending,
and a small, hand-written question set is not a benchmark: a difference of one
or two questions is within noise.

## Current configuration: every plot, one plot chunk per figure, the solver

| | |
|---|---|
| Questions | `data/questions/fastcalo_plots_values.jsonl`: 10 pairs, 20 questions (18 answerable, 2 to refuse). Pairs 1-6 are the questions of `fastcalo_plots.jsonl`; pairs 7-10 ask for values other than peaks (a given bin, the first bin where one histogram overtakes the other, a local maximum in a range, a given heatmap cell) |
| Plot data | one automatic calibration and one digitization of all 224 images of the corpus: 141 digitized (488 series: 55 2D maps, 259 step histograms, 174 curves), 83 with the reason they were not |
| Chunks | sentence chunker, back matter excluded, plus one plot chunk per image: 224 plot chunks, series named by colour and drawing ([guide, step 5](guide.md#5-ingest-into-the-chunks)) |
| Retrieval | dense MiniLM, `explicit_arxiv`, top-5, at most one plot chunk per figure (`--max-plots-per-figure 1`) |
| Generation | `grounded` prompt, question restated, context 32768 tokens, temperature 0, `--solver` (planner with up to 3 plans) |

This is the fair setting: every plot of the corpus is in the index, not only
the figures the questions are about, so the right plot competes with many
similar ones (for 1805.00850, Figures 6, 8, 10, 11 and 12 are also jet
transverse-momentum plots with several panels).

**Retrieval.** The paper is retrieved for 20 of 20 questions, and the required
series is reached for 10/10 figure-reference and 9/10 natural questions
(directly or through the panels of a retrieved figure). Without the
one-per-figure cap the panels of one figure filled the five slots: 18/20
papers and 15/20 required series (Figure 12 of 1805.00850 retrieved for a
question that names Figure 1). The remaining miss, the natural question of
pair 3, retrieves Figure 6 of the right paper.

**The operations alone.** `scripts/compute.py` runs a hand-written plan for
each question on the same data: 18 computed and 2 declined, 20/20 matching the
reference answers within 1%.

**Answers** (correct of 18 answerable; "new" = pairs 7-10, of 8):

| Configuration | Qwen3 4B | Mistral 7B |
|---|---|---|
| **current** | **7 (+1 partial); new 2** | **2 (+1 partial); new 1** |
| 119 digitized images (before histograms in pieces or filled were read) | 7 (+1 partial); new 2 | 2; new 2 |
| same, without the one-per-figure cap | 5; new 1 | 2; new 1 |
| earlier: 77 plot chunks (below) | 9; new 2 | 5 (+1 partial); new 1 |

- Qwen refuses both count questions in every run. Mistral refused both until
  the current run, where it answers one of them with "246", the number of bins.
- With the required series retrieved for 19 of 20 questions, the misses are
  the planner's and the answer model's: Qwen declines to plan although the
  right plot is in the context (pair 7 natural, pair 10 figure-reference),
  plans on the wrong panel of the right figure (pair 1 natural), reads the x
  axis twice for a question on both axes (pair 5), and once refuses a
  correctly computed 0.70 MeV as coming from "a different figure" (pair 10
  natural). Its pair 2 answer computes the right separation, 0.549, and then
  calls it "greater than one cell" (partial).
- Qwen never invents a value: without the number it abstains. Mistral 7B
  computes a result for 7 of 20 questions, often does not use it (a computed
  195 GeV answered as "the 101st bin"), and invents values when its plans fail.
- Fewer correct answers than with 77 plot chunks is the cost of the harder
  setting, not of worse data: the plots of the questions and their values are
  identical in both.
- Near-identical runs differ by one or two questions; read the trend.

## Reading histograms drawn in pieces or filled

A digitization of every image first gave 119 digitized images. Of the 105
others, 28 had gone to the pixel route and 27 of those failed. Only 3 of the 28
were bitmaps: the other 25 were vector PDFs with calibrated axes whose curves
had not been selected as series. Reading the PDF paths of those files showed
three causes:

| Files | What the PDF contains | Why it was not selected |
|---|---|---|
| 16 (`sum_pred` figures of 1805.00850) | stroked step histograms broken into pieces where bins are empty | a series had to be one connected polyline |
| 4 (`lateral_width` of the two CaloGAN papers) | filled histograms closed along a baseline far below the page | filled paths were never candidates |
| 2 (`depthweightedenergy` of the two CaloGAN papers) | tick labels drawn as outlines, so the axes came from OCR | the OCR route did not look for vector curves at all |

The selection now reads histograms in pieces and filled histograms (with the
rules in the [guide](guide.md#which-vector-paths-are-data)), and keeps vector
curves when the axes come from OCR. Result on the same corpus: 141 digitized
images (+22, all 22 files above), 488 series; no image that was digitized
before lost a series or changed a value, 34 of them gained a series (a fill or
an outline that had been dropped). The hand-written plans still match 20/20 and
retrieval is identical (paper 20/20, required series 19/20).

The plot answers do not change beyond noise: the 20 questions are about two
figures that were already digitized. Measuring the effect on answers needs
questions on the recovered figures. In the same work a planner step that
emptied a series (a `restrict` on a range without bins, followed by
`evaluate`) crashed a Mistral run; such a selection is now refused where it
happens, with its expression.

## Earlier configuration: 77 plot chunks

The same questions and generation, with plot chunks only for the 77 images of
two partial digitization runs (the hybrid 2D maps and Figure 1 of
1805.00850): 55 with values, 22 not digitized. Retrieval without the
one-per-figure cap: paper retrieved for 18 of 20 questions, required series
reached for 10/10 figure-reference and 8/10 natural questions; the two misses
were the natural questions on the Geant4 first layer, which retrieved
1712.10321 Fig. 3.

| Configuration | Qwen3 4B | Mistral 7B |
|---|---|---|
| plot cards only | 5 (+1 partial); new 0 | 3 (+1 partial); new 0 |
| `--solver`, one plan | ~8; new 1 | ~5; new 1 |
| `--solver` with plan correction, three close variants | 9-11; new 2-3 | 5-7; new 1-3 |
| final solver configuration | 9; new 2 | 5 (+1 partial); new 1 |

The errors were plans: the wrong quantity (global maxima instead of the value
at 505 GeV), the right number without what the answer needs (a crossing
coordinate without its bin), the wrong axis (the cards give units, not axis
titles such as η), and corrections that do not converge in 3 plans.

The sections below describe the earlier layout, with the extracted values in
the chunks, on the 12 questions of `fastcalo_plots.jsonl`; they are kept as the
record of why the layout changed.

## Earlier layout: values in the chunks

### Setup

| | |
|---|---|
| Questions | `data/questions/fastcalo_plots.jsonl`: 6 pairs, 12 questions (10 answerable, 2 to refuse) |
| Plot evidence | 56 series from 55 figure files: 54 2D maps (arXiv 1705.02355, 1712.10321, 2106.05285) and the two step histograms of Figure 1 of 1805.00850 |
| Chunks | sentence chunker, back matter excluded, plus 451 plot chunks: 56 series summaries, 383 coordinate blocks, 12 figure overviews ([guide, step 5](guide.md#5-ingest-into-the-chunks)) |
| Retrieval | dense MiniLM, `explicit_arxiv` source policy, top-5, figure overviews attached to retrieved panels |
| Generation | `grounded` prompt with the question restated after the passages, context 32768 tokens, temperature 0; Qwen3 4B (`qwen3:4b-instruct`) and Mistral 7B (`mistral:7b`) on the same passages |

The plot evidence comes from two digitization runs: an automatic run over the
76 hybrid figure files (vector axes around an embedded image), with 54
successes, and a run restricted to Figure 1 of 1805.00850. A digitization of
every calibrated file has not been run yet, so 1D curves are represented by
that single figure.

Each pair has a **figure-reference** question, which cites the figure and the
arXiv id, and a **natural** question that names neither and searches the whole
corpus.

| Pair | Figure | Asks for | Reference answer |
|---|---|---|---|
| 1 | 1705.02355 Fig. 1, Geant4, first layer | brightest cell: energy and position | ≈2137 MeV at x=47.54, y=1.00 (cell index) |
| 2 | same, Geant4 vs CaloGAN | are the two maxima within one cell along η? | 47.544 vs 46.995: 0.549 apart, yes |
| 3 | 1805.00850 Fig. 1, unweighted | peak position and height | x=45 GeV, y≈0.005347 A.U. |
| 4 | same, both histograms | ratio of the two peak heights | 0.005347 / 0.003345 ≈ 1.60 |
| 5 | same | axis scales and units | X linear in GeV, Y log10 in A.U. |
| 6 | same | exact number of original events in the top bin | refuse: a normalized plot cannot give it |

### Retrieval

The required plot series were in the retrieved passages for all 12 questions
(6/6 figure-reference, 6/6 natural). Paper hit@5 is 12/12. This measures
whether the numbers were available to the model, not whether they were used
correctly.

### Answers

Correct: right value and units. Partial: right value with part of the answer
missing, or a ratio more than 1% off.

| | Qwen3 4B | Mistral 7B |
|---|---:|---:|
| answerable questions correct | 9 of 10 | 2 of 10 |
| partial | 0 | 4 |
| count questions refused | 2 of 2 | 1 of 2 |
| citation labels valid | 12 of 12 | 12 of 12 |

**Qwen** answers every answerable question but one: both variants of pairs 1,
3, 4 and 5, and the figure-reference variant of pair 2 (47.544 vs 46.995, 0.549
cell apart). It computes the ratio from the two extracted peaks (1.601 and
1.603 for 1.599). It refuses both count questions, saying that the values are
normalized displayed heights. Its one miss is the natural question of pair 2,
where it answers `insufficient_evidence`.

**Mistral** gets the figure-reference question of pair 1 and the peak of pair 3
right. It gives the peak position of pair 3 without its height in the natural
variant, ratios of 1.57 and 1.54, and the axis scales of pair 5 without the GeV
unit. Its natural answer to pair 1 reports 2134.985 MeV, the peak of the
CaloGAN panel given as the Geant4 answer. It says the two maxima of pair 2 are
*not* within one cell, and answers the figure-reference count question with
"1805", the arXiv number.

### What changed the results

Same questions, same chunks and retrieval, Qwen3 4B:

| Configuration | correct | refused |
|---|---:|---:|
| question only before the passages, server default context | 6 of 10 | 2 of 2 |
| one chunk per image (compact tables), question restated | 5 of 10 | 2 of 2 |
| per-series chunks, question restated (adopted) | 9 of 10 | 2 of 2 |

- **Restating the question after the passages** is the change that matters:
  with the question only before several coordinate blocks, Qwen answered
  `insufficient_evidence` on questions whose values were in its context.
- **One chunk per image** retrieved the expected series for 10 of 12 questions
  instead of 12, and the answers followed. Its contexts were also longer: about
  16,600 characters on average instead of 10,300.
- With one chunk per image holding every value as `x, y, z` triples, prompts
  reached 32,000 tokens; Ollama's default context silently cut them and the
  models returned data dumps instead of answers. This is why `--num-ctx` is
  fixed (default 32768).

### Effect on the text questions

The restated question applies to every prompt, so it was checked on
`data/questions/fastcalo_paired.jsonl` (16 answerable, 4 to refuse) with the
semantic chunker (`fd100`, back matter excluded), dense retrieval, top-5, the
same retrieved passages before and after the change:

| | Qwen3 4B before | after | Mistral 7B before | after |
|---|---:|---:|---:|---:|
| answered / abstained | 7 / 13 | 14 / 6 | 18 / 2 | 16 / 4 |
| answerable questions correct | 6 (+1 partial) | 10 (+1 partial) | — | — |
| answerable questions wrong | 0 | 3 | — | — |
| questions to refuse, refused | 4 of 4 | 4 of 4 | 2 of 4 | 4 of 4 |

Qwen answers more and mostly correctly, but where it used to abstain it now
also gives three wrong answers: physical cell sizes instead of the image
dimensions, "GAN" instead of the U-net generator, and exact energy
conservation attributed to the CaloGAN companion paper. Mistral stops inventing
a deployment procedure for the two questions it must refuse, and loses a
malformed answer that listed 125 citation labels. Mistral's correctness on the
answerable text questions was not scored.

### What this shows

- Plot extraction makes the numbers available: without plot chunks the
  generator has nothing to answer from (in an earlier text-only comparison on
  the same papers, Qwen abstained on every plot question).
- Having the numbers is necessary, not sufficient. Choosing the right panel,
  doing the arithmetic and refusing unanswerable questions remain model
  behaviours: Mistral 7B picks the wrong panel and invents a count where Qwen3
  4B does not.
- Valid citation labels say nothing about whether the cited passage supports
  the answer; in every wrong answer above the citations are valid.

### Reproduce

```bash
python scripts/retrieve.py --dataset-dir <chunks with plots> \
  --questions data/questions/fastcalo_plots.jsonl \
  --retriever dense --source-policy explicit_arxiv --top-k 5 --output-dir <retrieval dir>
sbatch scripts/run_ollama.slurm scripts/generate.py --retrieval-dir <retrieval dir> \
  --output-dir <qa dir> --model qwen3:4b-instruct
python scripts/review.py prepare --qa-dirs <qa dirs> --output-dir <review dir>
```

The chunk directory is built as in the [guide](guide.md) (step 5), from the two
digitization runs described above.
