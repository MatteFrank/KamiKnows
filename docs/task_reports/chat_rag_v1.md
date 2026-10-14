# Interactive RAG CLI

## Objective, inputs and outputs

Implement user-written terminal questions over existing chunks, with TF-IDF or
real cached MiniLM and native/shared Ollama. Add `scripts/chat_rag.py`, usage
instructions in `docs/chat_rag.md` and offline tests in `tests/test_chat_rag.py`.
Existing datasets, benchmarks and run outputs remain unchanged.

The index loads once. Explicit-source filtering/balancing and full-text prompt
hydration reuse existing code. Each turn writes context before generation and
answer immediately after completion, including backend failures and raw output.
Session manifests include dataset hash, generator parameters and query policy.
EOF/Ctrl+C and /exit close the loop without deleting completed records.

## Commands and validation

Full pytest suite: 258 passed, 2 skipped. CLI --help and git diff --check passed.
Offline tests cover dense candidate filtering and complete evidence,
interactive inputs, valid citations, backend failures and persistence. No real
LLM inference or new GPU job was started for this implementation.

## Limits and next step

Questions are independent, not conversational history. Query text contains no
benchmark keyword expansion; interactive metrics are not benchmark comparisons.
No token streaming; complete answers are displayed after each request. The
existing QA parser and its warnings remain visible and unchanged. Citation
validation does not prove factual support; manual review is still required.
Missing requested sources yield explicit empty context, but an LLM may ignore
abstention instructions. MiniLM requires a cached model; Ollama must be started
separately on the assigned GPU node. Follow docs/chat_rag.md for the live trial.
