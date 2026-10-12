# KamiKnows

KamiKnows is a modular scientific-LLM research prototype for turning scientific literature into structured, traceable, reviewable knowledge and for testing grounded scientific RAG workflows.

The current repository is no longer a Phase-0-only prototype. It contains a controlled FastCaloSimulation / calorimetry workflow from arXiv full-text source parsing to RAG-ready datasets, retrieval evaluation, grounded QA, manual-review artifacts, reranking, and generator/configuration benchmarking.

> Current development stage: **Phase 2 - robust scientific RAG, in progress**.

This repository does **not** yet demonstrate scientific reliability, scientific discovery, production readiness, a final embedding/retrieval solution, or a final backbone LLM.

---

## 1. Current status

Current canonical FastCalo corpus:

```text
papers: 5
chunks_v1: 108
equations: 28
hardened evaluation questions: 13
chunks flagged very_long_text: 31
```

Implemented workflow:

```text
arXiv IDs
-> arXiv LaTeX/source parsing
-> per-paper structured artifacts
-> RAG-ready dataset v0
-> normalized chunk metadata v1
-> TF-IDF retrieval baseline
-> embedding-model evaluation pipeline
-> grounded QA with deterministic citation labels
-> manual grounding-review artifacts
-> lightweight deterministic reranking
-> RAG generator/configuration benchmark
```

Current Phase-2 task status:

| Task | Status | Main point |
| --- | --- | --- |
| P2T1 retrieval baseline | completed | deterministic TF-IDF baseline and diagnostics |
| P2T2 embedding selection | pipeline completed, scientific comparison pending | previous run used hash fallback because `sentence-transformers` was unavailable |
| P2T3 chunk metadata v1 | completed | 108 traceable chunks; 31 flagged `very_long_text` |
| P2T4 grounded QA v1 | completed | citation-constrained QA over 13 questions |
| P2T5 manual grounding review | artifacts completed, human labels pending | 39 answer records prepared for review |
| P2T6 light reranking | completed | hit@5 unchanged; MRR slightly improved |
| P2T7 generator benchmark | executed but invalid | 156/156 generation records were backend errors |

Important current limitation:

```text
P2T7 must be rerun after validating the Ollama runtime and installed model tags.
The existing P2T7 result is infrastructure/error evidence, not a model-quality comparison.
```

---

## 2. Current model configuration

The canonical grounded-QA and P2T7 configurations currently use:

```text
qwen3:0.6b
qwen3:4b-instruct
mistral:7b
```

`qwen3:0.6b` remains the small smoke-test/baseline model. It is **not** the only supported model and should not be interpreted as the selected final KamiKnows model.

The current Ollama-based code accepts other installed model tags through CLI overrides. When testing additional models, write results to a new output directory instead of overwriting the canonical baseline.

Current embedding registry:

```text
sentence-transformers/all-MiniLM-L6-v2
intfloat/e5-small-v2
BAAI/bge-m3
```

The previous P2T2 run used deterministic hash fallback for these entries. That run validates pipeline wiring only.

---

## 3. Repository layout

Main areas:

```text
configs/
  phase2_p2t4_qa_grounded_v1.json
  phase2_p2t5_manual_review_v1.json
  phase2_p2t6_light_reranking_v1.json
  phase2_p2t7_rag_generator_benchmark_v1.json

data/
  input/
  eval/

kamiknows/
  extraction/
  fulltext/
  ingestion/
  models/
  quality/
  rag/
  rag_ready/
  storage/

scripts/
  run_fulltext_parsing_pilot.py
  build_rag_ready_dataset.py
  build_rag_ready_chunk_metadata_v1.py
  run_retriever_baseline_v1.py
  run_embedding_model_selection_v1.py
  run_qa_grounded_v1.py
  run_manual_grounding_review_v1.py
  run_light_reranking_v1.py
  run_rag_generator_benchmark_v1.py
  ...

outputs/
  fulltext_fastcalo_5_v0/
  rag_ready_fastcalo_v0/
  rag_ready_fastcalo_v1/
  rag_v1_fastcalosim_retrieval/
  rag_v1_fastcalosim_embedding_selection/
  rag_v1_fastcalosim_qa_grounded_config_v1/
  rag_v1_fastcalosim_manual_review/
  rag_v1_fastcalosim_reranking_light_v1/
  rag_v1_fastcalosim_generator_benchmark_v1/

docs/
  PHASE2_STATUS.md
  COMMANDS.md
  chunk_schema_v1.md
  rag_citation_contract_v1.md
  task_reports/
```

---

## 4. Installation

Run commands from the repository root.

### 4.1 Clone

```bash
git clone https://github.com/MatteFrank/KamiKnows.git
cd KamiKnows
```

### 4.2 Python environment

Use an isolated Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Current core `requirements.txt` contains:

```text
requests>=2.32.0
pytest>=8.0.0
jsonschema>=4.0.0
```

These dependencies are sufficient for the core repository/tests and the deterministic retrieval pipeline, but **not for real transformer embeddings**.

### 4.3 Real semantic embeddings

For P2T2 with real embeddings, install `sentence-transformers`.

On GPU/HPC systems, install a PyTorch build appropriate for the local CUDA/driver environment first, then install Sentence Transformers.

```bash
python -m pip install sentence-transformers
```

Verify:

```bash
python -c "import sentence_transformers; print(sentence_transformers.__version__)"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

For scientific embedding comparisons, use:

```text
--encoder-backend sentence-transformers
```

Do **not** use the hash fallback as scientific evidence.

### 4.4 Ollama

Generation currently uses the Ollama HTTP API. Ollama is an external runtime and is not installed by `requirements.txt`.

The default endpoint used by KamiKnows is:

```text
http://localhost:11434
```

Pull the current canonical benchmark models:

```bash
ollama pull qwen3:0.6b
ollama pull qwen3:4b-instruct
ollama pull mistral:7b
```

Start the server when required:

```bash
ollama serve
```

In another shell, verify availability:

```bash
ollama list
curl http://localhost:11434/api/tags
```

Quick generation test:

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "qwen3:0.6b",
    "prompt": "Reply with OK only.",
    "stream": false
  }'
```

Repeat with every model that will be used in a benchmark.

On HPC systems, run the model service where GPU resources are actually allocated and set `--base-url` if the Ollama server is not reachable at the default localhost endpoint.

---

## 5. Test the repository before a run

Run the offline/default test suite:

```bash
pytest -q
```

Ollama-dependent tests are opt-in:

```bash
KAMIKNOWS_RUN_OLLAMA_TEST=1 pytest -s tests/test_qwen_ollama_basic.py
KAMIKNOWS_RUN_OLLAMA_TEST=1 pytest -s tests/test_qwen_abstract_to_json_ollama.py
```

Do not start a multi-model benchmark until:

```text
pytest passes
+ Ollama responds
+ every requested model appears in `ollama list`
+ a direct generation test succeeds for every requested model
```

---

# 6. Canonical current workflow

The commands below describe the current FastCalo workflow from source papers to the Phase-2 benchmark.

Existing baseline outputs may already be present in the repository. Rebuild them only when you intentionally want to reproduce the pipeline from the beginning.

---

## Step 1 - Define the controlled paper set

Canonical current input:

```text
data/input/arxiv_ids_fastCalo_5fulltext.txt
```

Current IDs:

```text
1705.02355v2
1712.10321v1
1805.00850v2
1807.01954v2
2106.05285v3
```

Use one arXiv ID per line. Comments beginning with `#` are allowed.

---

## Step 2 - Full-text LaTeX/source parsing

```bash
python3 scripts/run_fulltext_parsing_pilot.py \
  --ids-file data/input/arxiv_ids_fastCalo_5fulltext.txt \
  --output-dir outputs/fulltext_fastcalo_5_v0
```

Optional network timeout:

```bash
python3 scripts/run_fulltext_parsing_pilot.py \
  --ids-file data/input/arxiv_ids_fastCalo_5fulltext.txt \
  --output-dir outputs/fulltext_fastcalo_5_v0 \
  --timeout-seconds 120
```

This stage:

```text
downloads arXiv metadata/source
-> identifies and flattens LaTeX
-> extracts plain text
-> sections
-> equations
-> initial chunks
-> quality report
```

Main outputs:

```text
outputs/fulltext_fastcalo_5_v0/
  fulltext_manifest.json
  parsing_quality_report.json
  parsing_quality_report.md
  processed/papers/<paper_id>/
    metadata.json
    source_download.json
    flat.tex
    plain_text.txt
    paper.json
    sections.json
    equations.json
    chunks.jsonl
```

Current implementation is primarily arXiv-LaTeX/source based. It is not a general production PDF parser.

Inspect before continuing:

```bash
cat outputs/fulltext_fastcalo_5_v0/parsing_quality_report.md
```

---

## Step 3 - Build RAG-ready dataset v0

```bash
python3 scripts/build_rag_ready_dataset.py \
  --fulltext-dir outputs/fulltext_fastcalo_5_v0 \
  --output-dir outputs/rag_ready_fastcalo_v0 \
  --domain "HEP / FastCaloSimulation / calorimetry"
```

Main outputs:

```text
outputs/rag_ready_fastcalo_v0/
  chunks.jsonl
  papers.jsonl
  equations.jsonl
  eval_questions_v0.jsonl
  rag_manifest_v0.json
  chunk_quality_report.json
  chunk_quality_report.md
```

Inspect:

```bash
cat outputs/rag_ready_fastcalo_v0/chunk_quality_report.md
cat outputs/rag_ready_fastcalo_v0/rag_manifest_v0.json
```

---

## Step 4 - Build chunk metadata v1

```bash
python3 scripts/build_rag_ready_chunk_metadata_v1.py \
  --input-dir outputs/rag_ready_fastcalo_v0 \
  --output-dir outputs/rag_ready_fastcalo_v1
```

Main outputs:

```text
outputs/rag_ready_fastcalo_v1/
  chunks_v1.jsonl
  chunk_audit_v0_summary.json
  chunk_audit_v0_details.jsonl
  chunk_id_map_v0_to_v1.json
  eval_questions_v0.jsonl
  rag_manifest_v1.json
  retrieval_smoke_summary_v1.json
```

Current known issue:

```text
31 / 108 chunks are flagged `very_long_text`
```

The current v1 step adds metadata/traceability. It does not yet implement the planned chunk-splitting optimization.

---

## Step 5 - Run deterministic TF-IDF retrieval baseline

Canonical run:

```bash
python3 scripts/run_retriever_baseline_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_retrieval \
  --top-k-values 3 5 8
```

You can change retrieval depths:

```bash
python3 scripts/run_retriever_baseline_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_retrieval_k_test \
  --top-k-values 1 3 5 10
```

Main outputs:

```text
retriever_baseline_v1.jsonl
retriever_eval_summary_v1.json
retriever_error_analysis_v1.md
retrieval_debug_report.md
retrieval_run_config.json
retrieval_manifest.json
```

The current retriever is deterministic `tfidf_local_v1`.

BM25 and hybrid lexical+dense retrieval are **not implemented yet** in the current repository.

---

## Step 6 - Run real embedding-model comparison

For a scientifically meaningful run, force the real Sentence Transformers backend:

```bash
python3 scripts/run_embedding_model_selection_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_embedding_selection_real_v1 \
  --top-k 5 \
  --encoder-backend sentence-transformers
```

By default this evaluates all registry models:

```text
all-MiniLM-L6-v2
e5-small-v2
bge-m3
```

Run only selected models:

```bash
python3 scripts/run_embedding_model_selection_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_embedding_selection_minilm_e5_v1 \
  --top-k 5 \
  --encoder-backend sentence-transformers \
  --models all-MiniLM-L6-v2 e5-small-v2
```

A hash-only smoke test is possible:

```bash
python3 scripts/run_embedding_model_selection_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_embedding_selection_hash_smoke \
  --top-k 5 \
  --encoder-backend hash
```

Do not compare hash-fallback scores as if they measured semantic embedding quality.

Current repository limitation:

```text
embedding selection is an evaluation pipeline;
the main QA/reranking path still uses TF-IDF retrieval.
```

A future dense/hybrid retriever must explicitly integrate the selected embedding model into QA retrieval.

---

## Step 7 - Grounded QA v1

### Canonical config-based run

```bash
python3 scripts/run_qa_grounded_v1.py \
  --config configs/phase2_p2t4_qa_grounded_v1.json
```

Current config:

```text
RAG-ready dataset:
  outputs/rag_ready_fastcalo_v1

eval set:
  data/eval/rag_eval_set_v1_hardened.jsonl

retriever:
  tfidf_local_v1

top_k:
  5

models:
  qwen3:0.6b
  qwen3:4b-instruct
  mistral:7b

generation:
  temperature = 0.0
  top_p = 0.9
  num_predict = 512
  timeout = 120 s
```

Default config output:

```text
outputs/rag_v1_fastcalosim_qa_grounded_config_v1
```

### One-model smoke run

Use a new directory:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --config configs/phase2_p2t4_qa_grounded_v1.json \
  --models qwen3:0.6b \
  --output-dir outputs/rag_v1_fastcalosim_qa_smoke_qwen06
```

### Alternative model set

Example:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --config configs/phase2_p2t4_qa_grounded_v1.json \
  --models qwen3:4b-instruct qwen3:8b mistral:7b \
  --output-dir outputs/rag_v1_fastcalosim_qa_model_compare_v2 \
  --report-path docs/task_reports/phase2_task04_qa_model_compare_v2.md
```

The extra model must already exist in the local Ollama runtime.

The QA runner also supports overrides for:

```text
--rag-ready-dir
--output-dir
--report-path
--top-k
--base-url
--models
--temperature
--top-p
--num-predict
--timeout-seconds
--eval-questions-file
```

For controlled scientific comparisons, change one dimension at a time and write each run to a distinct output directory.

---

## Step 8 - Create manual grounding-review artifacts

```bash
python3 scripts/run_manual_grounding_review_v1.py \
  --config configs/phase2_p2t5_manual_review_v1.json
```

Main outputs:

```text
outputs/rag_v1_fastcalosim_manual_review/
  manual_review_queue_v1.jsonl
  manual_review_sheet_v1.csv
  manual_review_guidelines_v1.md
  eval_question_audit_v1.jsonl
  eval_question_audit_summary_v1.json
  rag_eval_set_v1_hardened_draft.jsonl
  rag_eval_set_hardening_notes_v1.md
  manual_review_summary_v1.json
```

Canonical hardened eval set:

```text
data/eval/rag_eval_set_v1_hardened.jsonl
```

Human action required:

```text
open and complete:
outputs/rag_v1_fastcalosim_manual_review/manual_review_sheet_v1.csv
```

Review dimensions include:

```text
manual_grounding
manual_answer_correctness
manual_citation_support
manual_abstention_behavior
severity
reviewer notes
```

Current repository status: the review queue was generated with 39 pending records. Human scientific judgment remains required.

There is currently no dedicated Phase-2 script that imports a completed `manual_review_sheet_v1.csv` and produces a final human-label benchmark summary. Do not reuse the old Phase-0 manual-quality summarizer as if it were the same workflow.

---

## Step 9 - Light reranking v1

Canonical run:

```bash
python3 scripts/run_light_reranking_v1.py \
  --config configs/phase2_p2t6_light_reranking_v1.json
```

Current settings:

```text
baseline retriever: tfidf_local_v1
candidate_top_n: 20
final_top_k: 5

score weights:
  base_retrieval_score: 0.55
  query_text_overlap: 0.25
  title_section_overlap: 0.10
  question_type_section_prior: 0.10
  quality_penalty: 0.05
```

Override candidate/final depths:

```bash
python3 scripts/run_light_reranking_v1.py \
  --config configs/phase2_p2t6_light_reranking_v1.json \
  --candidate-top-n 30 \
  --final-top-k 8 \
  --output-dir outputs/rag_v1_fastcalosim_reranking_light_test \
  --report-path docs/task_reports/phase2_task06_light_reranking_test.md
```

Current result:

```text
baseline hit@5: 0.727
reranked hit@5: 0.727
baseline MRR: 0.400
reranked MRR: 0.424
```

Interpretation: the light lexical reranker changes ordering but does not solve the main retrieval misses.

---

## Step 10 - RAG generator/configuration benchmark v1

Before running P2T7, verify the Ollama service and every model.

Recommended preflight:

```bash
ollama list
curl http://localhost:11434/api/tags

ollama run qwen3:0.6b "Reply only: OK"
ollama run qwen3:4b-instruct "Reply only: OK"
ollama run mistral:7b "Reply only: OK"
```

Because the existing committed P2T7 run contains 156/156 backend errors, preserve it and write the corrected run to a new directory:

```bash
python3 scripts/run_rag_generator_benchmark_v1.py \
  --config configs/phase2_p2t7_rag_generator_benchmark_v1.json \
  --output-dir outputs/rag_v1_fastcalosim_generator_benchmark_v1_rerun \
  --report-path docs/task_reports/phase2_task07_rag_generator_benchmark_v1_rerun.md
```

Canonical benchmark dimensions:

```text
context sources:
  tfidf_baseline_top5
  light_reranked_top5

prompt profiles:
  grounded_json_v1
  strict_abstention_v2

models:
  qwen3:0.6b
  qwen3:4b-instruct
  mistral:7b

parameter set:
  deterministic_512
```

Override models:

```bash
python3 scripts/run_rag_generator_benchmark_v1.py \
  --config configs/phase2_p2t7_rag_generator_benchmark_v1.json \
  --models qwen3:4b-instruct qwen3:8b mistral:7b \
  --output-dir outputs/rag_v1_fastcalosim_generator_benchmark_qwen8b \
  --report-path docs/task_reports/phase2_task07_generator_benchmark_qwen8b.md
```

Override prompt profiles:

```bash
python3 scripts/run_rag_generator_benchmark_v1.py \
  --config configs/phase2_p2t7_rag_generator_benchmark_v1.json \
  --prompt-profiles grounded_json_v1 \
  --output-dir outputs/rag_v1_fastcalosim_generator_benchmark_grounded_only \
  --report-path docs/task_reports/phase2_task07_generator_benchmark_grounded_only.md
```

Override context sources:

```bash
python3 scripts/run_rag_generator_benchmark_v1.py \
  --config configs/phase2_p2t7_rag_generator_benchmark_v1.json \
  --context-sources tfidf_baseline_top5 \
  --output-dir outputs/rag_v1_fastcalosim_generator_benchmark_tfidf_only \
  --report-path docs/task_reports/phase2_task07_generator_benchmark_tfidf_only.md
```

Main P2T7 outputs:

```text
benchmark_matrix_v1.json
benchmark_answers_v1.jsonl
benchmark_summary_v1.json
benchmark_config_comparison_v1.md
citation_validation_summary_v1.json
abstention_analysis_v1.json
benchmark_error_analysis_v1.md
recommended_rag_config_v1.json
benchmark_manifest_v1.json
```

A valid model comparison requires:

```text
backend errors understood and near zero
+ valid structured outputs
+ citation validation inspected
+ abstention behavior inspected
+ human grounding review
```

Automatic structural metrics alone are not a scientific-correctness claim.

---

# 7. Configuration files and override policy

Canonical configs:

```text
configs/phase2_p2t4_qa_grounded_v1.json
configs/phase2_p2t5_manual_review_v1.json
configs/phase2_p2t6_light_reranking_v1.json
configs/phase2_p2t7_rag_generator_benchmark_v1.json
```

General policy:

```text
config JSON = reproducible baseline
CLI option = explicit experimental override
new experiment = new output directory
```

Do not silently change a canonical config and overwrite previous outputs when performing model/retrieval/prompt comparisons.

---

# 8. Current evaluation set

Canonical Phase-2 eval data:

```text
data/eval/rag_eval_set_v1_hardened.jsonl
data/eval/rag_eval_set_v1_hardened.md
```

Current size:

```text
13 questions
```

It includes answerable, insufficient-evidence, multi-source and source-constrained questions.

Expected source IDs are used for post-hoc evaluation. They are not passed to the model as hidden ground truth for generation.

---

# 9. Citation policy

Grounded QA uses deterministic retrieved-context labels:

```text
[C1], [C2], [C3], ...
```

Automatic validation checks that generated citation labels correspond to retrieved chunks.

This proves only structural citation validity, not factual entailment or scientific correctness.

---

# 10. Reproducibility and run discipline

Before changing a model, retriever, prompt, chunking strategy, or evaluation set:

```text
1. freeze the input dataset
2. record config and model tag
3. use a new output directory
4. run tests
5. verify external services/models
6. run the experiment
7. inspect structural metrics
8. inspect error analysis
9. perform human scientific review where required
10. only then compare against the previous baseline
```

Never interpret backend errors, hash fallback embeddings, formal JSON validity, or citation-label validity as evidence of scientific answer quality.

---

# 11. What is currently implemented vs not implemented

Implemented:

```text
arXiv metadata ingestion
abstract extraction workflow
arXiv LaTeX/source full-text pilot
section/equation extraction
RAG-ready dataset construction
chunk metadata/audit v1
TF-IDF retrieval
embedding benchmark infrastructure
Sentence-Transformers integration path
grounded QA
citation-label validation
hardened evaluation set
manual-review artifact generation
light deterministic reranking
generator/configuration benchmark infrastructure
Ollama model plugin
```

Not yet implemented or not yet scientifically validated:

```text
production PDF parsing
BM25 retrieval
hybrid lexical+dense retrieval
dense embeddings integrated into the main QA retriever
vector database production path
semantic reranker / cross-encoder reranker
completed human grounding labels
automatic scientific entailment validation
scientific graph/state representation
solver-based quantitative validation
simulation/data validation layer
fine-tuning / LoRA / QLoRA in the current Phase-2 workflow
discovery generation
large-scale corpus processing
production scientific reliability
```

---

# 12. Current next steps

1. **Repair and rerun P2T7**
   - verify Ollama on the execution host;
   - verify all requested model tags;
   - run one-model smoke generation first;
   - rerun the full matrix in a new output directory.

2. **Rerun P2T2 with real embeddings**
   - install `sentence-transformers`;
   - force `--encoder-backend sentence-transformers`;
   - compare MiniLM, E5 and BGE on the same corpus/eval set;
   - do not use the previous hash-fallback ranking as a semantic result.

3. **Improve retrieval**
   - add BM25;
   - integrate dense retrieval into the main QA path;
   - evaluate lexical vs dense vs hybrid retrieval under the same eval set.

4. **Chunking v2**
   - address the 31 `very_long_text` chunks;
   - preserve section/equation traceability;
   - rerun retrieval after chunk changes.

5. **Complete human grounding review**
   - fill `manual_review_sheet_v1.csv`;
   - add a reproducible summarizer for completed human labels;
   - use human labels before any scientific reliability claim.

---

# 13. Legacy scripts

The repository still contains Phase-0/Phase-1 utilities for regression, tutorials, abstract-level extraction, and historical pilots, including:

```text
download_arxiv_metadata_batch.py
run_batch_arxiv_extraction.py
run_hep_pilot.py
run_model_mini_benchmark.py
run_mini_rag.py
create_manual_quality_checklist.py
summarize_manual_quality_review.py
run_quality_gate.py
summarize_hep_pilot_run.py
```

These are not the canonical entry point for the current Phase-2 FastCalo workflow.

Do not use old Phase-0 commands or old manual-quality scripts to replace the current P2T4-P2T7 workflow.

---

# 14. Main current documentation

Read in this order:

```text
README.md
docs/PHASE2_STATUS.md
docs/COMMANDS.md
docs/chunk_schema_v1.md
docs/rag_citation_contract_v1.md

docs/task_reports/phase2_task02_embedding_model_selection_v1.md
docs/task_reports/phase2_task03_chunking_metadata_v1.md
docs/task_reports/phase2_task04_qa_grounded_v1.md
docs/task_reports/phase2_task05_manual_grounding_review_v1.md
docs/task_reports/phase2_task06_light_reranking_v1.md
docs/task_reports/phase2_task07_rag_generator_benchmark_v1.md
```

Historical Phase-0/Phase-1 documents remain useful for provenance but should not be treated as the current operational status.

---

# 15. Scientific interpretation rule

KamiKnows separates:

```text
pipeline correctness
retrieval quality
citation validity
answer groundedness
scientific correctness
```

Passing one layer does not imply passing the next.

The current repository is a controlled scientific-RAG research scaffold. It is intended to make experiments reproducible and reviewable before scaling to broader corpora, stronger models, semantic/hybrid retrieval, scientific-state representations, solver/data validation, and later training/discovery work.
