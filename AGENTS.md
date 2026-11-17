# KamiKnows - notes for coding agents

Read README.md first: it describes the pipeline, the commands and the run
conventions. This file lists the rules to follow when working in the repo.

## Scope

- KamiKnows is a retrieval + grounded generation pipeline over a fixed,
  small corpus. No training or fine-tuning, and no growth of the corpus
  beyond the id lists in `data/input/`, unless the task explicitly asks.
- Never invent citations, arXiv ids, chunk ids or sources. Citations are the
  passage labels (C1, C2, ...) produced by retrieval; the model may only
  choose among them.
- Keep separate in code and in reports: retrieval quality vs answer quality,
  citation-label validity vs factual support, backend errors vs model
  quality. Automatic checks never establish scientific correctness; say
  "manual review pending" when no human has reviewed the answers.

## Language

- All authored repository content must be in English: code, comments, docstrings,
  prompts, documentation, CLI messages, diagnostics, reports and generated output.
- Do not translate user-facing report text into the conversation language.
- Keep HTML language attributes and test expectations consistent with English.

## Code

- `kamiknows/` holds the logic, one module per step (`parsing`, `chunking`,
  `retrieval`, `generation`, `review`) plus `common` and `embeddings`; the
  optional plot extraction route lives in the `plot_extractor/` package.
  `scripts/` only parses arguments and calls one `run_*` function. Do not put
  logic in scripts, and do not import scripts from other code.
- One command per step; variants are options (sub-commands for the chunker),
  never new scripts. A new chunker, retriever or prompt is a new option value
  in the existing module.
- Every step writes through `common.create_output_dir` (refuses existing
  directories) and ends with `common.write_manifest`, recording all
  parameters and input files. Each step reads its input with
  `common.read_manifest(dir, "<previous step>")`.
- The ranking query is the question text only. Eval annotations
  (`expected_*`) are for evaluation, never for retrieval or prompting.
- Embedding models load from the local Hugging Face cache only; fail with a
  clear error rather than falling back to anything else.
- Keep `tests/` offline (no network, Ollama or real models: use the fixtures
  in `tests/conftest.py`, fake encoders and a monkeypatched
  `generation.ollama_generate`). Add or update a test for every behaviour
  change and run `pytest -q` before finishing.
- Match the existing style: type hints, short docstrings that explain why,
  no dead code, no compatibility shims for removed features.
- Name things after what they are, not after why they were made: no
  `demo`, `talk` or `presentation` in identifiers, run names or docs. Method
  and prompt identifiers carry no iteration counters (`axes_auto`, not
  `axes_auto_v14`); the manifest already records the git commit.

## Runs

- Never modify or delete an existing run directory in `outputs/`, and never
  touch `outputs_backup/` (runs of the pre-refactor code, reference only).
  A failed run is retried in a new directory with `_retry<N>`.
- Name run directories following the pattern and vocabulary in README.md
  ("Run directories and naming"); extend the vocabulary there when a new
  value appears.
- GPU work (generation) goes through `scripts/run_ollama.slurm`; do not
  start Ollama on the login node. The SLURM log is written to
  `outputs/_slurm_logs/` and copied into the run directory at the end.
- After every run: add a one-line note in `outputs/index_notes.json`
  (what changed compared with which baseline) and run
  `python scripts/index_outputs.py`, which writes `outputs/INDEX.md`.
- Versioned files (code, docs, questions, slides) never name directories in
  `outputs/`: they are not in git. Describe results by their configuration
  (chunker, retriever, model, questions file, top-k), not by run name.
- Change one thing at a time when comparing configurations, and compare runs
  on the same questions file and top-k.

## Questions

- An evaluation is one JSONL file in `data/questions/` holding questions and
  expected answers (format in README.md, "Questions files").
- Never edit a questions file that runs have used: copy it to a new name
  with a descriptive suffix (e.g. `fastcalo_paired_hard.jsonl`), edit the copy,
  and use a new evalset tag in run names.
