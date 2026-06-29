"""Grounded QA prompt construction for mini-RAG v0."""

from __future__ import annotations

from typing import Any


def build_grounded_qa_prompt(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Build a concise grounded QA prompt with citation instructions."""
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            "\n".join(
                [
                    f"[CONTEXT {chunk.get('rank')}]",
                    f"chunk_id: {chunk.get('chunk_id')}",
                    f"arxiv_id: {chunk.get('arxiv_id')}",
                    f"title: {chunk.get('title')}",
                    f"section_heading: {chunk.get('section_heading')}",
                    "text:",
                    str(chunk.get("text") or ""),
                ]
            )
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "<NO_CONTEXT>"
    return f"""You are KamiKnows, a scientific question-answering assistant.

Answer the question using ONLY the provided context.
Cite chunk_id and arxiv_id for each factual statement.
Use citation style: (chunk_id: <chunk_id>, arxiv_id: <arxiv_id>).
If the context is insufficient, say so explicitly.
Do not invent sources.
Do not use outside knowledge.
Separate supported answer from uncertainty/limitations.
Keep the answer concise and technical.

Question:
{question}

Context:
{context_text}

Answer:
"""
