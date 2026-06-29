"""Mini-RAG retrieval/generation evaluation summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any

ERROR_TAXONOMY = (
    "retrieval_miss",
    "answer_not_grounded",
    "wrong_citation",
    "hallucination",
    "incomplete_answer",
    "failed_abstention",
    "json_or_format_error",
    "backend_error",
    "empty_context",
    "unknown_error",
)


def evaluate_expected_source_hit(
    retrieved_chunks: list[dict[str, Any]],
    expected_source_arxiv_ids: list[str],
) -> dict[str, Any]:
    """Evaluate whether retrieval hit any expected source arXiv ID."""
    expected = {str(arxiv_id) for arxiv_id in expected_source_arxiv_ids if arxiv_id}
    if not expected:
        return {
            "expected_source_hit": False,
            "expected_source_hit_rank": None,
            "notes": "No expected source arXiv IDs were specified.",
        }

    for chunk in retrieved_chunks:
        if str(chunk.get("arxiv_id")) in expected:
            return {
                "expected_source_hit": True,
                "expected_source_hit_rank": chunk.get("rank"),
                "notes": "At least one expected source was retrieved.",
            }
    return {
        "expected_source_hit": False,
        "expected_source_hit_rank": None,
        "notes": "No retrieved chunk matched the expected source arXiv IDs.",
    }


def build_retrieved_context_record(
    *,
    question: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """Build one retrieved_contexts.jsonl record."""
    expected = list(question.get("expected_source_arxiv_ids", []) or [])
    return {
        "question_id": question.get("question_id", ""),
        "question": question.get("question", ""),
        "question_type": question.get("question_type", ""),
        "top_k": top_k,
        "retriever": {
            "name": "tfidf_local_v0",
            "version": "retriever_v0",
        },
        "retrieved_chunks": retrieved_chunks,
        "expected_source_arxiv_ids": expected,
        "retrieval_eval": evaluate_expected_source_hit(retrieved_chunks, expected),
    }


def build_eval_summary(
    *,
    retrieved_contexts: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    top_k: int,
    retrieval_only: bool,
) -> dict[str, Any]:
    """Build rag_eval_summary.json."""
    expected_source_questions = sum(
        1 for record in retrieved_contexts if record.get("expected_source_arxiv_ids")
    )
    expected_source_hits = sum(
        1
        for record in retrieved_contexts
        if record.get("retrieval_eval", {}).get("expected_source_hit")
    )
    attempted = sum(
        1 for answer in answers if answer.get("generation_status") != "skipped_retrieval_only"
    )
    succeeded = sum(1 for answer in answers if answer.get("generation_status") == "success")
    backend_errors = sum(1 for answer in answers if answer.get("generation_status") == "backend_error")

    hit_rate = (
        expected_source_hits / expected_source_questions
        if expected_source_questions
        else 0.0
    )
    return {
        "rag_eval_summary_version": "rag_eval_summary_v0",
        "questions_total": len(retrieved_contexts),
        "retrieval": {
            "top_k": top_k,
            "expected_source_questions": expected_source_questions,
            "expected_source_hits": expected_source_hits,
            "expected_source_hit_rate": hit_rate,
        },
        "generation": {
            "answers_attempted": attempted,
            "answers_succeeded": succeeded,
            "backend_errors": backend_errors,
            "retrieval_only": retrieval_only,
        },
        "manual_review": {
            "required": True,
            "status": "pending",
        },
        "scope": "mini-RAG v0; retrieval plus grounded generation; no automatic scientific correctness judgment",
    }


def build_error_analysis_markdown(
    *,
    answers: list[dict[str, Any]],
    retrieved_contexts: list[dict[str, Any]],
) -> str:
    """Build the Markdown error-analysis scaffold."""
    counts = Counter()
    retrieval_misses = 0
    empty_contexts = 0
    for record in retrieved_contexts:
        if not record.get("retrieved_chunks"):
            empty_contexts += 1
        if record.get("expected_source_arxiv_ids") and not record.get("retrieval_eval", {}).get("expected_source_hit"):
            retrieval_misses += 1
    for answer in answers:
        for category in answer.get("error_categories", []) or []:
            counts[category] += 1
    counts["retrieval_miss"] += retrieval_misses
    counts["empty_context"] += empty_contexts

    lines = [
        "# KamiKnows Mini-RAG v0 Error Analysis",
        "",
        "This file is an error taxonomy scaffold. Grounding, citation correctness, hallucination, and answer completeness remain not reviewed until manual inspection.",
        "",
        "## Error Taxonomy",
        "",
    ]
    for category in ERROR_TAXONOMY:
        lines.append(f"- `{category}`: {counts.get(category, 0)}")

    lines.extend(
        [
            "",
            "## Review Status",
            "",
            "- Retrieval misses are computed automatically against expected source arXiv IDs.",
            "- Generation grounding and citation correctness are not marked as PASS automatically.",
            "- Manual review is required before using results as scientific QA.",
            "",
        ]
    )
    return "\n".join(lines)
