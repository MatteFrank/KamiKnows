# KamiKnows RAG Citation Contract v1

## Purpose

The v1 citation contract defines the structured answer shape future RAG generation must produce before an answer can be accepted for automated downstream processing.

It is a validation contract, not a scientific correctness claim. Manual review remains required.

## Why v0 Citations Were Empty

Mini-RAG v0 prompted the model to cite `chunk_id` and `arxiv_id`, but the answer text was free-form. The model could mention an arXiv ID in prose without emitting a machine-readable citation object. As a result, `rag_answers.jsonl` preserved `retrieved_chunk_ids`, but generated `citations` could remain empty or unreliable.

That made citation correctness hard to check automatically.

## Required JSON Answer Format

Future QA generation should return a JSON object with this shape:

```json
{
  "answer": "...",
  "citations": [
    {
      "chunk_id": "...",
      "arxiv_id": "...",
      "section_heading": "...",
      "supports": "short explanation"
    }
  ],
  "insufficient_evidence": false,
  "limitations": "...",
  "used_chunk_ids": []
}
```

## Citation Rules

- `citations` must be a subset of the retrieved chunk IDs for that question.
- `used_chunk_ids` must also be a subset of the retrieved chunk IDs.
- No citation may reference an unseen chunk.
- No citation may invent a `chunk_id`.
- Each citation should explain what it supports in the `supports` field.
- If the retrieved context is insufficient, `insufficient_evidence` should be `true` and the answer should abstain or narrow its claim.

## Automatic Acceptance Gate

An answer is not accepted automatically when:

- `citations` is empty;
- a citation omits `chunk_id`;
- a citation references an unknown chunk;
- a citation references a known chunk that was not retrieved for the question;
- the answer claims evidence while `insufficient_evidence` is true.

The helper in `kamiknows/rag/citation_validation.py` currently validates the citation subset rule and returns one of:

- `valid`
- `missing_citations`
- `unknown_citation`
- `cites_unretrieved_chunk`

## Manual Review

Passing citation validation only means the cited chunk IDs are structurally allowed. A reviewer still needs to check:

- whether the cited text actually supports the answer;
- whether the answer is complete;
- whether caveats and limitations are preserved;
- whether the answer avoids unsupported scientific claims;
- whether the expected-source labels were appropriate.

This contract hardens traceability. It does not replace scientific review.
