# Model licenses - KamiKnows Fase 0

This file records the models considered in the first tutorial phase.
It is a working note, not a legal opinion.

## Qwen3-0.6B

- Model name: Qwen3-0.6B
- Local Ollama tag: `qwen3:0.6b`
- Provider/source: Qwen / Alibaba Cloud
- License: Apache 2.0
- Intended role in KamiKnows: first small local tutorial model; smoke tests; simple extraction-to-JSON experiments.
- Main limitation: useful for pipeline testing, not a final scientific-quality model choice.
- Status: candidate tutorial backend.

## Mistral-7B-Instruct-v0.3

- Model name: Mistral-7B-Instruct-v0.3
- Possible Ollama tag: `mistral`
- Provider/source: Mistral AI
- License: Apache 2.0 for the model weights.
- Intended role in KamiKnows: stronger 7B baseline for JSON stability and abstract understanding comparisons.
- Main limitation: larger than Qwen3-0.6B; requires more memory; not selected as final backbone without benchmark.
- Status: candidate comparison backend.

## DeepSeek-R1-Distill-Qwen-1.5B

- Model name: DeepSeek-R1-Distill-Qwen-1.5B
- Local Ollama tag: `deepseek-r1:1.5b`
- Provider/source: DeepSeek AI, distilled from Qwen-family model.
- License: MIT for DeepSeek-R1 materials; note Qwen-family derivation and keep provenance tracked.
- Intended role in KamiKnows: possible critic/judge/reasoning backend in later tests.
- Main limitation: may produce long reasoning text; should be constrained carefully for structured extraction.
- Status: candidate critic backend.

## Project rule

Open weights do not automatically mean fully open source.
For every model used in experiments, record:

- exact model name;
- exact version or tag;
- source URL;
- license;
- date checked;
- local runtime used;
- intended role;
- benchmark results before any architectural decision.
