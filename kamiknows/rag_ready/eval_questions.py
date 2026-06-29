"""Static evaluation-question scaffold for RAG-ready dataset v0."""

from __future__ import annotations

from typing import Any

QUESTION_TYPES = {
    "method",
    "limitation",
    "comparison",
    "definition",
    "source_lookup",
    "unanswerable",
}


def _paper_refs(papers: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return papers[: min(count, len(papers))]


def _ids(papers: list[dict[str, Any]]) -> list[str]:
    return [str(paper.get("arxiv_id")) for paper in papers if paper.get("arxiv_id")]


def _question(
    *,
    question_id: str,
    question: str,
    question_type: str,
    expected_source_arxiv_ids: list[str],
    expected_section_keywords: list[str],
    evaluation_criteria: str,
    notes: str,
) -> dict[str, Any]:
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unknown question_type: {question_type}")
    return {
        "question_id": question_id,
        "question": question,
        "question_type": question_type,
        "expected_source_arxiv_ids": expected_source_arxiv_ids,
        "expected_section_keywords": expected_section_keywords,
        "evaluation_criteria": evaluation_criteria,
        "notes": notes,
    }


def generate_eval_questions(
    *,
    papers: list[dict[str, Any]],
    domain: str,
) -> list[dict[str, Any]]:
    """Generate static source-constrained evaluation questions.

    These records define retrieval/evaluation constraints. They do not contain
    generated answers and do not call any model.
    """
    selected = _paper_refs(papers, 5)
    first = selected[0:1]
    second = selected[1:2] or first
    third = selected[2:3] or first
    comparison_pair_a = _paper_refs(selected, 2)
    comparison_pair_b = selected[2:4] if len(selected) >= 4 else comparison_pair_a
    all_ids = _ids(selected)

    questions = [
        _question(
            question_id="eval_v0_q001",
            question="What simulation or modeling method is proposed in the first selected FastCalo paper?",
            question_type="method",
            expected_source_arxiv_ids=_ids(first),
            expected_section_keywords=["method", "model", "simulation", "approach"],
            evaluation_criteria="A valid answer must cite only the expected source paper and describe the method using retrieved evidence, not prior knowledge.",
            notes=f"Domain: {domain}. Do not require an exact generated answer in this scaffold.",
        ),
        _question(
            question_id="eval_v0_q002",
            question="Which neural or statistical modeling method is used for calorimeter shower generation in another selected paper?",
            question_type="method",
            expected_source_arxiv_ids=_ids(second),
            expected_section_keywords=["GAN", "generative", "model", "architecture", "training"],
            evaluation_criteria="A valid answer must identify the method only if supported by chunks from the constrained source paper.",
            notes="Use this to test method retrieval precision.",
        ),
        _question(
            question_id="eval_v0_q003",
            question="What limitations or validation caveats are stated for the first selected paper's approach?",
            question_type="limitation",
            expected_source_arxiv_ids=_ids(first),
            expected_section_keywords=["limitation", "validation", "future", "uncertainty", "conclusion"],
            evaluation_criteria="A valid answer must quote or paraphrase limitations from retrieved text and avoid inventing caveats.",
            notes="Tests whether retrieval finds cautionary language.",
        ),
        _question(
            question_id="eval_v0_q004",
            question="What limitations, open issues, or future-work constraints are stated in a later FastCalo paper?",
            question_type="limitation",
            expected_source_arxiv_ids=_ids(third),
            expected_section_keywords=["limitation", "future", "outlook", "discussion", "conclusion"],
            evaluation_criteria="A valid answer must be grounded in the expected paper and should not generalize beyond the retrieved source.",
            notes="Pairs with q003 for limitation coverage.",
        ),
        _question(
            question_id="eval_v0_q005",
            question="Compare the modeling goals of two selected FastCalo papers using only their retrieved chunks.",
            question_type="comparison",
            expected_source_arxiv_ids=_ids(comparison_pair_a),
            expected_section_keywords=["introduction", "method", "simulation", "calorimeter"],
            evaluation_criteria="A valid comparison must mention evidence from both expected papers and avoid claims unsupported by retrieved chunks.",
            notes="Cross-paper comparison scaffold.",
        ),
        _question(
            question_id="eval_v0_q006",
            question="Compare how two later papers discuss validation or performance of fast calorimeter simulation.",
            question_type="comparison",
            expected_source_arxiv_ids=_ids(comparison_pair_b),
            expected_section_keywords=["validation", "performance", "results", "comparison"],
            evaluation_criteria="A valid answer must compare retrieved evidence from both constrained papers, not provide a generic survey.",
            notes="Designed to test multi-source retrieval behavior later.",
        ),
        _question(
            question_id="eval_v0_q007",
            question="Which selected source discusses calorimeter simulation as a computational bottleneck, and where should the evidence be found?",
            question_type="source_lookup",
            expected_source_arxiv_ids=all_ids,
            expected_section_keywords=["introduction", "simulation", "computational", "CPU", "bottleneck"],
            evaluation_criteria="A valid answer should identify the source paper(s) and cite the chunk/section evidence. Multiple expected IDs are allowed.",
            notes="Source lookup question, not an answer-generation target.",
        ),
        _question(
            question_id="eval_v0_q008",
            question="What exact production-ready detector simulation replacement should be deployed based on these five papers?",
            question_type="unanswerable",
            expected_source_arxiv_ids=all_ids,
            expected_section_keywords=["validation", "limitations", "future", "production"],
            evaluation_criteria="A valid answer should state that the dataset is insufficient for an exact deployment recommendation unless retrieved chunks explicitly support it.",
            notes="Intentionally insufficient-evidence question to test refusal/uncertainty behavior later.",
        ),
    ]

    return questions
