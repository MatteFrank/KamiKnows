"""Generation wrapper for mini-RAG v0."""

from __future__ import annotations

import re
from typing import Any

from kamiknows.models.ollama_plugin import OllamaPlugin
from kamiknows.rag.prompting import build_grounded_qa_prompt


def extract_citations_from_answer(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Extract explicit retrieved chunk_id mentions from a raw answer."""
    citations = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        if re.search(rf"\b{re.escape(chunk_id)}\b", answer):
            citations.append(
                {
                    "chunk_id": chunk_id,
                    "arxiv_id": str(chunk.get("arxiv_id") or ""),
                    "section_heading": str(chunk.get("section_heading") or ""),
                }
            )
            seen.add(chunk_id)
    return citations


def build_answer_record(
    *,
    context_record: dict[str, Any],
    answer: str,
    citations: list[dict[str, str]],
    backend: str,
    model: str,
    top_k: int,
    generation_status: str,
    error_categories: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build one rag_answers.jsonl record."""
    return {
        "question_id": context_record.get("question_id", ""),
        "question": context_record.get("question", ""),
        "question_type": context_record.get("question_type", ""),
        "answer": answer,
        "citations": citations,
        "retrieved_chunk_ids": [
            chunk.get("chunk_id", "")
            for chunk in context_record.get("retrieved_chunks", [])
        ],
        "backend": backend,
        "model": model,
        "top_k": top_k,
        "grounding_status": "not_reviewed",
        "generation_status": generation_status,
        "error_categories": error_categories or [],
        "notes": notes,
    }


def generate_answer_records(
    *,
    retrieved_contexts: list[dict[str, Any]],
    backend: str,
    model: str,
    top_k: int,
    retrieval_only: bool,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate or skip answer records for each retrieved context."""
    if retrieval_only:
        return [
            build_answer_record(
                context_record=context,
                answer="",
                citations=[],
                backend=backend,
                model=model,
                top_k=top_k,
                generation_status="skipped_retrieval_only",
                notes="Generation skipped because --retrieval-only was used.",
            )
            for context in retrieved_contexts
        ]

    if backend != "ollama":
        return [
            build_answer_record(
                context_record=context,
                answer="",
                citations=[],
                backend=backend,
                model=model,
                top_k=top_k,
                generation_status="backend_error",
                error_categories=["backend_error"],
                notes=f"Unsupported backend: {backend}",
            )
            for context in retrieved_contexts
        ]

    plugin = OllamaPlugin(model=model, base_url=base_url)
    records = []
    for context in retrieved_contexts:
        prompt = build_grounded_qa_prompt(
            question=str(context.get("question") or ""),
            retrieved_chunks=list(context.get("retrieved_chunks", []) or []),
        )
        try:
            answer = plugin.generate(prompt, temperature=temperature)
        except Exception as exc:
            records.append(
                build_answer_record(
                    context_record=context,
                    answer="",
                    citations=[],
                    backend=backend,
                    model=model,
                    top_k=top_k,
                    generation_status="backend_error",
                    error_categories=["backend_error"],
                    notes=str(exc),
                )
            )
            continue

        records.append(
            build_answer_record(
                context_record=context,
                answer=answer,
                citations=extract_citations_from_answer(
                    answer,
                    list(context.get("retrieved_chunks", []) or []),
                ),
                backend=backend,
                model=model,
                top_k=top_k,
                generation_status="success",
                notes="Citations are extracted heuristically from explicit chunk_id mentions; manual review required.",
            )
        )
    return records
