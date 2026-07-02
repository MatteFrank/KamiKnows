# Phase 2 Task 04b - QA Grounded Config Refactor

Date: 2026-07-02

## Objective

Centralize QA grounded v1 runtime defaults in a JSON config and update the
default model set without changing the scientific logic, corpus, parser,
retriever, or existing P2T4 outputs.

## Context

P2T4 QA grounded v1 is complete. The existing output directory is:

```text
outputs/rag_v1_fastcalosim_qa_grounded
```

Known P2T4 metrics:

- questions: 13
- retrieval expected-source hit rate: 0.833
- retrieval misses: `eval_v0_q002`, `eval_v0_q005`
- qwen3:0.6b: 11 answered, 2 insufficient, 0 backend errors
- qwen3:4b: 13 backend-error records
- mistral:7b: 12 answered, 1 insufficient, 0 backend errors
- invalid citation records: 0
- invented citation labels: 0
- manual grounding review: pending

## Changes Made

Added canonical config:

```text
configs/phase2_p2t4_qa_grounded_v1.json
```

The config now centralizes:

- input RAG-ready directory;
- output directory and report path;
- retriever top-k;
- Ollama base URL;
- answer-generation model list and sampling options;
- eval-set mode;
- static v1 question toggle;
- explicit disabled LLM question-generation toggle;
- citation policy;
- no-training, no-discovery, no-scaling, and no-parser-change scope.

The config default output directory is intentionally new:

```text
outputs/rag_v1_fastcalosim_qa_grounded_config_v1
```

This avoids overwriting the completed P2T4 output directory.

Updated default generator models:

```text
qwen3:0.6b
qwen3:4b-instruct
mistral:7b
```

`qwen3:4b` is no longer a default because the local P2T4 run found it
available as an Ollama tag but it returned backend-error records for all
generation calls.

## Eval Question Source

The QA grounded v1 eval set is still deterministic and static:

```text
data/eval/rag_eval_set_v1_hardened.jsonl
```

No LLM generates questions today. The config field:

```json
"llm_question_generation_enabled": false
```

is an explicit guardrail, not an implemented generation pathway.

## CLI Behavior

The existing command without `--config` remains supported:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_v1 \
  --output-dir outputs/rag_v1_fastcalosim_qa_grounded \
  --report-path docs/task_reports/phase2_task04_qa_grounded_v1.md \
  --top-k 5 \
  --models qwen3:0.6b qwen3:4b-instruct mistral:7b
```

The new config-based command is:

```bash
python3 scripts/run_qa_grounded_v1.py \
  --config configs/phase2_p2t4_qa_grounded_v1.json
```

Explicit CLI arguments override config values.

## Outputs

No RAG-ready dataset was regenerated.

No existing P2T4 output directory was rerun or overwritten for this refactor.

## Tests

Focused tests were added for:

- JSON config loading;
- CLI override precedence over config values;
- legacy CLI resolution without config;
- updated default models;
- deterministic eval-set construction from data files;
- disabled LLM question generation guardrail.

Validation command:

```bash
pytest -q tests/test_qa_grounded_v1.py
```

Result:

```text
11 passed
```

## Limits

- Superseded by P2T5b: operational RAG eval questions now live in
  `data/eval/rag_eval_set_v1_hardened.jsonl`.
- The config does not introduce LLM-generated questions.
- The config does not change retrieval, citation validation, or answer
  generation logic.
- Manual grounding review remains pending for generated P2T4 answers.

## Next Step

Run manual grounding review on the existing P2T4 answers, then decide whether
to rerun QA grounded v1 in a new output directory using the config default
`qwen3:4b-instruct`.
