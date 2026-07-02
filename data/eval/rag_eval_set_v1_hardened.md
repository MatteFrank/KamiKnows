# RAG Eval Set v1 Hardened

Canonical machine-readable file:

```text
data/eval/rag_eval_set_v1_hardened.jsonl
```

This file externalizes the complete Phase 2 RAG eval set. It replaces runtime
question text embedded in Python code. The set contains 13 deterministic
questions: 8 hardened v0 questions and 5 v1 extension questions.

Manual review remains required. The hardened eval set improves test clarity and
reproducibility, but it does not validate generated answer quality.

## Summary

- Total questions: 13
- Answerable questions: 11
- Insufficient-evidence questions: 2 (`eval_v0_q008`, `eval_v1_q013`)
- Multi-source questions requiring all expected sources: `eval_v0_q005`, `eval_v0_q006`, `eval_v1_q012`
- Questions still flagged for manual question review: `eval_v0_q002`, `eval_v0_q005`, `eval_v0_q006`, `eval_v0_q007`, `eval_v0_q008`, `eval_v1_q012`
- Questions still flagged for expected-source review: `eval_v0_q002`, `eval_v0_q005`, `eval_v0_q006`, `eval_v0_q007`, `eval_v0_q008`, `eval_v1_q012`

## Questions

| ID | Type | Expected behavior | Expected sources | Requires all sources | Review flags |
| --- | --- | --- | --- | --- | --- |
| eval_v0_q001 | method | answerable | 1705.02355v2 | false | none |
| eval_v0_q002 | method | answerable | 1712.10321v1 | false | manual question, expected source |
| eval_v0_q003 | limitation | answerable | 1705.02355v2 | false | none |
| eval_v0_q004 | limitation | answerable | 1805.00850v2 | false | none |
| eval_v0_q005 | comparison | answerable | 1705.02355v2; 1712.10321v1 | true | manual question, expected source |
| eval_v0_q006 | comparison | answerable | 1805.00850v2; 1807.01954v2 | true | manual question, expected source |
| eval_v0_q007 | source_lookup | answerable | 1705.02355v2; 1712.10321v1; 1805.00850v2; 1807.01954v2; 2106.05285v3 | false | manual question, expected source |
| eval_v0_q008 | unanswerable | insufficient_evidence | candidate context only | false | manual question, expected source |
| eval_v1_q009 | fact_lookup | answerable | 1705.02355v2 | false | none |
| eval_v1_q010 | method | answerable | 1705.02355v2 | false | none |
| eval_v1_q011 | limitation | answerable | 1705.02355v2 | false | none |
| eval_v1_q012 | comparison | answerable | 1705.02355v2; 2106.05285v3 | true | manual question, expected source |
| eval_v1_q013 | no_answer | insufficient_evidence | none | false | none |

## Maintenance Rule

When editing eval questions, update the JSONL first and keep this Markdown
summary aligned. Do not add operational eval question text back into Python.
