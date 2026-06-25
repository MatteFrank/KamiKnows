# KamiKnows Fase 1 Architecture Review

Date: 2026-06-23

## Current Architecture Summary

KamiKnows is ready as a small, controlled, abstract-level Fase 1 pilot system. The repository is not laid out under `src/kamiknows/`; the current package is top-level `kamiknows/`, with direct script execution supported by inserting the repository root into `sys.path`.

Current workflow:

```text
arXiv metadata
-> title + abstract
-> ModelPlugin backend
-> abstract-to-JSON extraction
-> schema validation
-> traceable JSONL
-> formal summary
-> manual quality checklist
-> manual review summary
-> dataset manifest
-> quality gate
-> post-pilot analysis
```

Main package areas:

- `kamiknows/ingestion/`: arXiv metadata validation, ID-list loading, arXiv Atom API fetching and parsing.
- `kamiknows/models/`: `ModelPlugin`, `FakeExtractionModel`, and `OllamaPlugin`.
- `kamiknows/extraction/`: prompt registry, extraction schema, model-output JSON parsing, and schema validation.
- `kamiknows/storage/`: JSONL read/write helpers.
- `kamiknows/quality/`: manual checklist generation, manual review parsing, and quality gate evaluation.
- `kamiknows/pilot/`: post-pilot analysis.
- `kamiknows/dataset_manifest.py`: file inventory, hashes, sizes, roles, record counts, and run context.
- `kamiknows/run_metadata.py`: run IDs, timestamps, backend/model, prompt hash, and schema version.

Main workflow scripts:

- `scripts/download_arxiv_metadata_batch.py`
- `scripts/run_batch_arxiv_extraction.py`
- `scripts/run_hep_pilot.py`
- `scripts/run_model_mini_benchmark.py`
- `scripts/create_manual_quality_checklist.py`
- `scripts/summarize_manual_quality_review.py`
- `scripts/create_dataset_manifest.py`
- `scripts/run_quality_gate.py`
- `scripts/summarize_hep_pilot_run.py`

Prompt/schema locations:

- Prompt registry: `kamiknows/extraction/prompt_registry.py`
- Extraction schema: `kamiknows/extraction/abstract_to_json.py`
- Run-record documentation: `docs/run_record_schema.md`
- Prompt versioning documentation: `docs/prompt_versioning.md`

Test coverage is broad for Fase 0/Fase 1 entry: 27 test files cover metadata validation, arXiv downloader parsing/mocking, extraction parsing and schema validation, JSONL storage, scripts, prompt registry, manifests, manual review, quality gate, HEP pilot flow, and post-pilot analysis. Qwen/Ollama tests are opt-in through `KAMIKNOWS_RUN_OLLAMA_TEST=1`, with `KNOWKAMI_RUN_OLLAMA_TEST=1` still accepted as a deprecated alias.

## Strengths

- Metadata ingestion is conceptually and mostly technically separate from model interpretation. Metadata can be downloaded and frozen before extraction, and `run_hep_pilot.py` writes `pilot_metadata.json` before model execution.
- Model backends are replaceable through the small `ModelPlugin.generate(prompt, temperature)` interface.
- Qwen/Ollama is correctly treated as the first practical baseline backend through `OllamaPlugin`, while `FakeExtractionModel` remains a deterministic software-test backend.
- The extraction layer is model-agnostic: `abstract_to_json()` receives a `ModelPlugin` and validates the parsed JSON against `EXTRACTION_SCHEMA`.
- JSONL outputs are traceable: batch records include `source`, `extraction`, and `run` blocks.
- Run metadata records backend, model, prompt version, prompt template hash, and extraction schema version.
- Manual review, dataset manifest, quality gate, and post-pilot analysis are first-class workflow steps rather than afterthoughts.
- The documentation repeatedly enforces the current boundary: no RAG, PDF parsing, LaTeX parsing, embeddings, vector DB, fine-tuning, large-scale scraping, or automatic scientific truth judging.
- HEP pilot size is guarded by `validate_pilot_sample_size()`, normally requiring 10-20 papers.
- Output artifacts are auditable: manifests include hashes, sizes, file roles, missing paths, and JSONL record counts.

## Risks

- Some reusable workflow logic still lives in `scripts/` and is imported by other scripts, for example `run_hep_pilot.py` imports `run_batch()` from `scripts/run_batch_arxiv_extraction.py`, and benchmark scripts import helpers from other scripts. This is acceptable for Fase 0 but makes repeated Fase 1 comparisons more fragile than package-level orchestration modules would be.
- Backend selection is hardcoded in argparse choices as `["fake", "ollama"]`. Mistral and DeepSeek can be run through Ollama by changing `--model`, but adding distinct backend families later will require touching script argument choices and backend factory logic.
- `build_model_for_record()` constructs a new backend object for each metadata record. This is fine for a 10-20 paper pilot, but it is not ideal for later backends with expensive initialization.
- The current top-level package layout works locally but is less clean than a `src/` layout or editable install. The user request mentions `src/kamiknows/`, but that path does not currently exist.
- Default model configuration is repeated in scripts via `DEFAULT_OLLAMA_MODEL = "qwen3:0.6b"` and related defaults. This is okay for the first baseline, but repeated hardcoding will become noisy when comparing Qwen, Mistral, and DeepSeek.
- Prompt/schema version labels can be overridden from CLI, but the prompt hash comes from the current template. A future prompt change must update versioning discipline carefully.
- The prompt template still says `You are KnowKami...` inside `kamiknows/extraction/prompt_registry.py`. This preserves prompt behavior after the rename, but it should be explicitly recorded as intentional prompt freeze before model comparisons.
- `append_jsonl_record()` writes any serializable dictionary. The extraction object is validated before record construction in the normal batch path, but whole-record validation remains informal.
- `run_hep_pilot.py` creates a manifest before manual review summary and quality gate outputs exist. The documented workflow later regenerates the manifest, but users must remember that first manifest is not the final reviewed manifest.
- Generated cache artifacts such as `__pycache__/` and `.pytest_cache/` are present locally and should not be considered project artifacts.

## Recommended Minimal Refactors

These are not required before the first pilot if time is short, but they are the smallest useful cleanups before repeated model comparisons.

1. Move reusable script logic into package modules:
   - `scripts/run_batch_arxiv_extraction.py` orchestration helpers could move to `kamiknows/workflows/batch_extraction.py`.
   - `scripts/summarize_jsonl.py` summary helpers could move to `kamiknows/quality/formal_summary.py`.
   - `scripts/run_hep_pilot.py` should then call package functions rather than importing other scripts.

2. Add a small backend factory module:
   - Example target: `kamiknows/models/factory.py`.
   - Keep only `fake` and `ollama` for now.
   - Make future Mistral/DeepSeek additions a model-name/config choice when they run through Ollama, not benchmark-logic changes.

3. Centralize pilot/model defaults:
   - Keep `qwen3:0.6b` as the documented default.
   - Put baseline defaults such as Ollama base URL, timeout, temperature, and Qwen model name in one package-level config module or documented constants module.

4. Add a whole-record JSONL validator:
   - Validate `source`, `extraction`, and `run` blocks together before writing pilot JSONL.
   - Keep the schema small and do not change extraction fields yet.

5. Document the prompt-name freeze:
   - Either explicitly note that the prompt still contains `KnowKami` to avoid behavior drift, or schedule a separate prompt-versioned change from `KnowKami` to `KamiKnows`.
   - Do not silently change the prompt before the first baseline pilot.

6. Add a final-pilot manifest step to the docs:
   - Make it explicit that after manual review summary and quality gate, the manifest should include `manual_review_summary`, `quality_gate_report`, and `post_pilot_analysis`.

## Things To Avoid

- Do not add RAG, PDF parsing, LaTeX parsing, embeddings, vector DB, LoRA, fine-tuning, or discovery generation yet.
- Do not expand the extraction schema before a real pilot error pattern justifies it.
- Do not change the model prompt during the architecture-prep step.
- Do not compare models on different metadata snapshots.
- Do not treat fake backend output as scientific quality evidence.
- Do not process 100+ papers before the 10-20 paper quality gate has been completed and understood.
- Do not use automatic formal validity as a proxy for scientific correctness.
- Do not add Mistral/DeepSeek-specific benchmark branches before the backend factory/config boundary is clean.
- Do not let CLI convenience hide the phase separation between metadata ingestion, model interpretation, formal checks, and manual scientific review.

## Proposed Next 5 Codex Tasks

1. Run a dry-run HEP pilot with the fake backend and `--allow-small-sample` only to verify artifact paths after the rename.
2. Freeze a real 10-20 paper HEP metadata file from a controlled query or curated ID list, then do not alter that metadata during the first comparison.
3. Run the first Qwen/Ollama pilot over the frozen metadata with `qwen3:0.6b`, producing JSONL, formal summary, checklist, manifest, and pilot report.
4. Complete manual review for the checklist, summarize it, regenerate the manifest to include review/gate artifacts, run the quality gate, and generate post-pilot analysis.
5. Only after reading the post-pilot analysis, choose one next change: prompt revision, model comparison on the same frozen metadata, query/metadata adjustment, or schema revision if repeated errors justify it.

## Acceptance Criteria For Starting The First 10-20 Paper Pilot

- The repository test suite passes in the project environment.
- The working package imports use `kamiknows`.
- The first pilot uses a frozen metadata JSON list or a fixed ID-list source.
- The sample size is 10-20 papers unless running an explicit dry-run/test.
- The backend is `ollama` and the baseline model is Qwen, currently `qwen3:0.6b`.
- Ollama is running locally and the selected Qwen model has been pulled.
- The prompt and extraction schema are unchanged from the audited Fase 0 baseline.
- Every generated extraction is written as traceable JSONL with `source`, `extraction`, and `run`.
- Formal summary is generated and inspected.
- Manual quality checklist is generated and assigned to human review.
- Dataset manifest is generated, and after manual review it is regenerated or supplemented to include review and quality artifacts.
- Quality gate is run after manual review summary exists.
- Post-pilot analysis is generated and read before scaling or changing model/prompt/schema.
- No RAG/PDF/LaTeX/embedding/vector/fine-tuning components are introduced.

## Overall Assessment

KamiKnows is architecturally ready to start the first controlled 10-20 paper HEP pilot, provided the team treats the first run as a baseline audit rather than a model-selection conclusion.

The main recommendation is to run the first Qwen/Ollama pilot now, then make only minimal package-organization refactors before repeated model comparisons. The current architecture already protects the most important Fase 1 boundaries: frozen metadata, replaceable model backend, schema-validated extraction, traceable JSONL, human review, manifest, quality gate, and post-pilot analysis.
