# Plot QA results

What the plot questions currently show, on the fastcalo corpus. These are
observations from inspecting the answers; the expert review (`review.csv`,
reviewer columns) is still pending, and a small, hand-written question set is
not a benchmark.

## Setup

| | |
|---|---|
| Questions | `data/questions/fastcalo_plots.jsonl`: 6 pairs, 12 questions (10 answerable, 2 to refuse) |
| Plot evidence | 56 series from 55 figure files: 54 2D maps (arXiv 1705.02355, 1712.10321, 2106.05285) and the two step histograms of Figure 1 of 1805.00850 |
| Chunks | sentence chunker, back matter excluded, plus 451 plot chunks (summaries, coordinate tables, figure overviews) |
| Retrieval | dense MiniLM, `explicit_arxiv` source policy, top-5, figure overviews attached to retrieved panels |
| Generation | `grounded` prompt, temperature 0; Qwen3 4B (`qwen3:4b-instruct`) and Mistral 7B (`mistral:7b`) on the same passages |

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

## Retrieval

The required plot series were in the retrieved passages for all 12 questions
(6/6 figure-reference, 6/6 natural). Paper hit@5 is 12/12. This measures
whether the numbers were available to the model, not whether they were used
correctly.

## Answers

| | Qwen3 4B | Mistral 7B |
|---|---:|---:|
| answered / abstained | 7 / 5 | 12 / 0 |
| answerable questions answered correctly | 6 of 10 | 4 of 10 |
| count questions refused | 2 of 2 | 0 of 2 |
| citation labels valid | 12 of 12 | 12 of 12 |

Mistral's four correct answers include both ratios, given only approximately
(1.57 and 1.58 for 1.60).

**Qwen** answers pairs 3 and 5 in both variants, and the figure-reference
variants of pairs 2 and 4 (ratio 0.005346717 / 0.003344805 = 1.601). It refuses
both count questions, as it should, but also the figure-reference question of
pair 1 and the natural questions of pairs 2 and 4. Its natural answer to pair 1
reports 2134.985 MeV at x=46.995: the coordinates of the CaloGAN panel, which
was the panel retrieved, given as the Geant4 answer.

**Mistral** answers everything. It reads the wrong layer for pair 1 (9561.661 MeV
at x=5.0), uses the CaloGAN panel in the natural variant like Qwen, says the two
maxima of pair 2 are *not* within one cell, describes the log Y axis as linear
in both variants of pair 5, and answers the count questions: once with "1805"
(the arXiv number), once with a proposal to count displayed points.

## What this shows

- Plot extraction makes the numbers available: without plot chunks the
  generator has nothing to answer from (in an earlier text-only comparison on
  the same papers, Qwen abstained on every plot question).
- Having the numbers is necessary, not sufficient. Choosing the right panel,
  doing the arithmetic and refusing unanswerable questions remain model
  behaviours, and the two models fail in opposite directions.
- Valid citation labels say nothing about whether the cited passage supports
  the answer; in every wrong answer above the citations are valid.

## Reproduce

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
