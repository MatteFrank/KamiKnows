"""Lightweight lexical reranking for Phase 2 Task 06.

The reranker is intentionally deterministic and local. It uses question text,
question type, candidate chunk text/metadata, quality flags, and the baseline
TF-IDF score. It does not use expected source IDs, notes, or evaluation
criteria as ranking signals.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from kamiknows.rag.load_dataset import read_json, read_jsonl
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1, tokenize_v1
from kamiknows.run_metadata import utc_now_iso


DEFAULT_SCORE_WEIGHTS = {
    "base_retrieval_score": 0.55,
    "query_text_overlap": 0.25,
    "title_section_overlap": 0.10,
    "question_type_section_prior": 0.10,
    "quality_penalty": 0.05,
}

RERANKING_OUTPUT_FILES = {
    "baseline_retrieval_results": "baseline_retrieval_results_v1.jsonl",
    "reranked_retrieval_results": "reranked_retrieval_results_v1.jsonl",
    "reranking_eval_summary": "reranking_eval_summary_v1.json",
    "reranking_question_deltas": "reranking_question_deltas_v1.jsonl",
    "reranking_error_analysis": "reranking_error_analysis_v1.md",
    "reranking_manifest": "reranking_manifest_v1.json",
}

SECTION_PRIORS = {
    "method": ("method", "model", "simulation", "approach", "architecture", "training"),
    "limitation": ("limitation", "discussion", "conclusion", "future", "validation", "outlook"),
    "no_answer": ("limitation", "discussion", "conclusion", "future", "validation"),
    "unanswerable": ("limitation", "discussion", "conclusion", "future", "validation"),
    "comparison": ("introduction", "method", "results", "validation", "comparison"),
    "source_lookup": ("introduction", "method", "results", "validation", "performance"),
}


class LightRerankerV1:
    """Deterministic lexical reranker over TF-IDF candidates."""

    name = "lexical_light_reranker_v1"
    version = "reranker_v1"

    def __init__(self, score_weights: dict[str, float] | None = None) -> None:
        weights = dict(DEFAULT_SCORE_WEIGHTS)
        if score_weights:
            weights.update({key: float(value) for key, value in score_weights.items()})
        self.score_weights = weights

    def rerank(
        self,
        question_record: dict[str, Any],
        candidate_chunks: list[dict[str, Any]],
        final_top_k: int,
    ) -> list[dict[str, Any]]:
        """Return reranked chunks with final citation labels."""
        if final_top_k < 1:
            raise ValueError("final_top_k must be >= 1")
        if not candidate_chunks:
            return []

        max_base_score = max(float(chunk.get("score") or chunk.get("original_score") or 0.0) for chunk in candidate_chunks)
        scored = []
        for chunk in candidate_chunks:
            scored.append(
                self._score_candidate(
                    question_record=_ranking_question_view(question_record),
                    candidate=chunk,
                    max_base_score=max_base_score,
                )
            )

        scored.sort(
            key=lambda record: (
                -float(record["final_score"]),
                int(record.get("original_rank") or 0),
                str(record.get("chunk_id") or ""),
            )
        )
        top = scored[:final_top_k]
        for rank, record in enumerate(top, start=1):
            record["reranked_rank"] = rank
            record["citation_id"] = f"C{rank}"
            record["citation_label"] = f"[C{rank}]"
        return top

    def _score_candidate(
        self,
        *,
        question_record: dict[str, Any],
        candidate: dict[str, Any],
        max_base_score: float,
    ) -> dict[str, Any]:
        question_text = str(question_record.get("question") or "")
        question_type = str(question_record.get("question_type") or "")
        query_tokens = _content_tokens(question_text)
        query_bigrams = _ngrams(query_tokens, 2)
        query_trigrams = _ngrams(query_tokens, 3)

        text = str(candidate.get("text") or "")
        title = str(candidate.get("title") or "")
        section = str(candidate.get("section") or candidate.get("section_heading") or "")
        title_section = f"{title} {section}"
        text_tokens = _content_tokens(text)
        title_section_tokens = _content_tokens(title_section)

        base_score = float(candidate.get("score") or candidate.get("original_score") or 0.0)
        normalized_base = base_score / max_base_score if max_base_score > 0 else 0.0
        text_overlap = _overlap_ratio(query_tokens, text_tokens)
        title_section_overlap = _overlap_ratio(query_tokens, title_section_tokens)
        phrase_overlap = max(
            _overlap_ratio(query_bigrams, _ngrams(text_tokens, 2)),
            _overlap_ratio(query_trigrams, _ngrams(text_tokens, 3)),
        )
        section_prior = _section_prior_score(question_type=question_type, section=section)
        quality_penalty = _quality_penalty(candidate)
        empty_text_penalty = 0.40 if not text.strip() else 0.0

        query_text_component = min(1.0, (0.75 * text_overlap) + (0.25 * phrase_overlap))
        rerank_score = (
            self.score_weights["query_text_overlap"] * query_text_component
            + self.score_weights["title_section_overlap"] * title_section_overlap
            + self.score_weights["question_type_section_prior"] * section_prior
            - self.score_weights["quality_penalty"] * quality_penalty
            - empty_text_penalty
        )
        final_score = (
            self.score_weights["base_retrieval_score"] * normalized_base
            + rerank_score
        )

        return {
            "original_rank": int(candidate.get("rank") or candidate.get("original_rank") or 0),
            "reranked_rank": None,
            "original_score": round(base_score, 8),
            "rerank_score": round(rerank_score, 8),
            "final_score": round(final_score, 8),
            "score_breakdown": {
                "normalized_base_retrieval_score": round(normalized_base, 8),
                "query_text_overlap": round(text_overlap, 8),
                "query_phrase_overlap": round(phrase_overlap, 8),
                "query_text_component": round(query_text_component, 8),
                "title_section_overlap": round(title_section_overlap, 8),
                "question_type_section_prior": round(section_prior, 8),
                "quality_penalty": round(quality_penalty, 8),
                "empty_text_penalty": round(empty_text_penalty, 8),
                "score_weights": dict(self.score_weights),
            },
            **_candidate_metadata(candidate),
        }


def run_light_reranking_v1(
    *,
    rag_ready_dir: Path,
    eval_questions_file: Path,
    output_dir: Path,
    report_path: Path,
    config_path: Path | None = None,
    candidate_top_n: int = 20,
    final_top_k: int = 5,
    score_weights: dict[str, float] | None = None,
    pytest_status: str | None = None,
) -> dict[str, Any]:
    """Run baseline retrieval, light reranking, evaluation, and reporting."""
    if candidate_top_n < 1:
        raise ValueError("candidate_top_n must be >= 1")
    if final_top_k < 1:
        raise ValueError("final_top_k must be >= 1")
    if candidate_top_n < final_top_k:
        raise ValueError("candidate_top_n must be >= final_top_k")

    rag_ready_dir = Path(rag_ready_dir)
    eval_questions_file = Path(eval_questions_file)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = read_jsonl(rag_ready_dir / "chunks_v1.jsonl")
    manifest = read_json(rag_ready_dir / "rag_manifest_v1.json")
    eval_questions = read_jsonl(eval_questions_file)

    baseline_records = build_baseline_retrieval_records(
        chunks=chunks,
        eval_questions=eval_questions,
        candidate_top_n=candidate_top_n,
        final_top_k=final_top_k,
    )
    reranked_records = build_reranked_retrieval_records(
        baseline_records=baseline_records,
        final_top_k=final_top_k,
        score_weights=score_weights,
    )
    question_deltas = build_question_deltas(
        baseline_records=baseline_records,
        reranked_records=reranked_records,
        final_top_k=final_top_k,
    )
    eval_summary = build_reranking_eval_summary(
        question_deltas=question_deltas,
        candidate_top_n=candidate_top_n,
        final_top_k=final_top_k,
    )

    output_paths = {
        key: output_dir / filename
        for key, filename in RERANKING_OUTPUT_FILES.items()
    }
    write_jsonl(output_paths["baseline_retrieval_results"], baseline_records)
    write_jsonl(output_paths["reranked_retrieval_results"], reranked_records)
    write_json(output_paths["reranking_eval_summary"], eval_summary)
    write_jsonl(output_paths["reranking_question_deltas"], question_deltas)
    output_paths["reranking_error_analysis"].write_text(
        build_error_analysis_markdown(question_deltas=question_deltas, eval_summary=eval_summary),
        encoding="utf-8",
    )
    manifest_record = build_reranking_manifest(
        rag_ready_dir=rag_ready_dir,
        eval_questions_file=eval_questions_file,
        output_dir=output_dir,
        report_path=report_path,
        config_path=config_path,
        output_paths=output_paths,
        candidate_top_n=candidate_top_n,
        final_top_k=final_top_k,
        pytest_status=pytest_status,
    )
    write_json(output_paths["reranking_manifest"], manifest_record)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_task_report_markdown(
            rag_ready_dir=rag_ready_dir,
            eval_questions_file=eval_questions_file,
            output_dir=output_dir,
            baseline_records=baseline_records,
            question_deltas=question_deltas,
            eval_summary=eval_summary,
            manifest=manifest,
        ),
        encoding="utf-8",
    )
    Path("data/eval/rag_eval_set_v1_hardened.md").write_text(
        build_eval_set_hardened_markdown(eval_questions),
        encoding="utf-8",
    )

    return {
        "chunks": chunks,
        "rag_manifest": manifest,
        "eval_questions": eval_questions,
        "baseline_records": baseline_records,
        "reranked_records": reranked_records,
        "question_deltas": question_deltas,
        "eval_summary": eval_summary,
        "manifest": manifest_record,
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "report_path": str(report_path),
    }


def build_baseline_retrieval_records(
    *,
    chunks: list[dict[str, Any]],
    eval_questions: list[dict[str, Any]],
    candidate_top_n: int,
    final_top_k: int,
) -> list[dict[str, Any]]:
    """Run wide TF-IDF retrieval and save baseline top-k plus candidates."""
    retriever = TfidfRetrieverV1(chunks)
    records = []
    for question in eval_questions:
        question_text = str(question.get("question") or "")
        question_type = str(question.get("question_type") or "")
        expected_keywords = [str(value) for value in question.get("expected_section_keywords", []) or []]
        retrieved = [
            _retrieved_candidate_dict(result.rank, result.score, result.chunk)
            for result in retriever.retrieve(
                question_text,
                top_k=candidate_top_n,
                question_type=question_type,
                expected_section_keywords=expected_keywords,
            )
        ]
        baseline_top_k = [_assign_citation(chunk, rank) for rank, chunk in enumerate(retrieved[:final_top_k], start=1)]
        records.append(
            {
                "question_id": question.get("question_id", ""),
                "question": question_text,
                "question_type": question_type,
                "expected_behavior": question.get("expected_behavior", ""),
                "expected_arxiv_ids": list(question.get("expected_arxiv_ids", []) or []),
                "expected_paper_ids": list(question.get("expected_paper_ids", []) or []),
                "required_all_expected_sources": bool(question.get("required_all_expected_sources")),
                "candidate_top_n": candidate_top_n,
                "final_top_k": final_top_k,
                "baseline_top_k_results": baseline_top_k,
                "candidate_top_n_results": retrieved,
                "baseline_retrieval_eval": evaluate_retrieval(
                    question=question,
                    retrieved_chunks=baseline_top_k,
                    top_k=final_top_k,
                ),
                "candidate_pool_eval": evaluate_retrieval(
                    question=question,
                    retrieved_chunks=retrieved,
                    top_k=candidate_top_n,
                ),
            }
        )
    return records


def build_reranked_retrieval_records(
    *,
    baseline_records: list[dict[str, Any]],
    final_top_k: int,
    score_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply the light reranker to each question candidate pool."""
    reranker = LightRerankerV1(score_weights=score_weights)
    records = []
    for record in baseline_records:
        question_view = {
            "question_id": record.get("question_id", ""),
            "question": record.get("question", ""),
            "question_type": record.get("question_type", ""),
        }
        reranked = reranker.rerank(
            question_record=question_view,
            candidate_chunks=list(record.get("candidate_top_n_results", []) or []),
            final_top_k=final_top_k,
        )
        records.append(
            {
                "question_id": record.get("question_id", ""),
                "question": record.get("question", ""),
                "question_type": record.get("question_type", ""),
                "expected_behavior": record.get("expected_behavior", ""),
                "expected_arxiv_ids": list(record.get("expected_arxiv_ids", []) or []),
                "expected_paper_ids": list(record.get("expected_paper_ids", []) or []),
                "required_all_expected_sources": bool(record.get("required_all_expected_sources")),
                "candidate_top_n": record.get("candidate_top_n"),
                "final_top_k": final_top_k,
                "reranked_top_k_results": reranked,
                "reranked_retrieval_eval": evaluate_retrieval(
                    question=record,
                    retrieved_chunks=reranked,
                    top_k=final_top_k,
                ),
                "reranker": {
                    "name": LightRerankerV1.name,
                    "version": LightRerankerV1.version,
                    "score_weights": reranker.score_weights,
                    "no_ground_truth_used_for_reranking": True,
                },
            }
        )
    return records


def build_question_deltas(
    *,
    baseline_records: list[dict[str, Any]],
    reranked_records: list[dict[str, Any]],
    final_top_k: int,
) -> list[dict[str, Any]]:
    """Compare baseline top-k and reranked top-k per question."""
    reranked_by_id = {str(record.get("question_id") or ""): record for record in reranked_records}
    deltas = []
    for baseline in baseline_records:
        question_id = str(baseline.get("question_id") or "")
        reranked = reranked_by_id[question_id]
        baseline_eval = baseline["baseline_retrieval_eval"]
        reranked_eval = reranked["reranked_retrieval_eval"]
        candidate_eval = baseline["candidate_pool_eval"]
        baseline_rank = baseline_eval.get("expected_source_hit_rank")
        reranked_rank = reranked_eval.get("expected_source_hit_rank")
        delta_rank = _rank_delta(baseline_rank, reranked_rank, final_top_k)
        category = classify_delta(
            baseline_eval=baseline_eval,
            reranked_eval=reranked_eval,
            candidate_eval=candidate_eval,
            question=baseline,
            delta_rank=delta_rank,
        )
        deltas.append(
            {
                "question_id": question_id,
                "question": baseline.get("question", ""),
                "question_type": baseline.get("question_type", ""),
                "expected_behavior": baseline.get("expected_behavior", ""),
                "expected_arxiv_ids": list(baseline.get("expected_arxiv_ids", []) or []),
                "required_all_expected_sources": bool(baseline.get("required_all_expected_sources")),
                "baseline_hit_at_5": bool(baseline_eval.get("expected_source_hit")),
                "reranked_hit_at_5": bool(reranked_eval.get("expected_source_hit")),
                "baseline_hit_at_1": baseline_rank == 1,
                "reranked_hit_at_1": reranked_rank == 1,
                "baseline_rank": baseline_rank,
                "reranked_rank": reranked_rank,
                "delta_rank": delta_rank,
                "candidate_pool_contains_expected_source": bool(candidate_eval.get("expected_source_hit")),
                "candidate_pool_rank": candidate_eval.get("expected_source_hit_rank"),
                "baseline_missing_expected_source_ids": baseline_eval.get("missing_expected_source_ids", []),
                "reranked_missing_expected_source_ids": reranked_eval.get("missing_expected_source_ids", []),
                "candidate_pool_missing_expected_source_ids": candidate_eval.get("missing_expected_source_ids", []),
                "delta_category": category,
                "error_analysis_tags": error_tags_for_delta(
                    question=baseline,
                    category=category,
                    baseline_results=list(baseline.get("baseline_top_k_results", []) or []),
                    reranked_results=list(reranked.get("reranked_top_k_results", []) or []),
                    candidate_eval=candidate_eval,
                    reranked_eval=reranked_eval,
                ),
            }
        )
    return deltas


def build_reranking_eval_summary(
    *,
    question_deltas: list[dict[str, Any]],
    candidate_top_n: int,
    final_top_k: int,
) -> dict[str, Any]:
    """Build aggregate baseline-vs-reranked metrics."""
    expected = [delta for delta in question_deltas if delta.get("expected_arxiv_ids")]
    total_expected = len(expected)
    baseline_hits = [delta for delta in expected if delta.get("baseline_hit_at_5")]
    reranked_hits = [delta for delta in expected if delta.get("reranked_hit_at_5")]
    baseline_hit_at_1 = [delta for delta in expected if delta.get("baseline_hit_at_1")]
    reranked_hit_at_1 = [delta for delta in expected if delta.get("reranked_hit_at_1")]
    improved = [delta["question_id"] for delta in expected if delta["delta_category"] == "improved"]
    worsened = [delta["question_id"] for delta in expected if delta["delta_category"] == "worsened"]
    unchanged = [
        delta["question_id"]
        for delta in expected
        if delta["delta_category"] in {"unchanged_hit", "unchanged_miss"}
    ]
    required_all = [delta for delta in expected if delta.get("required_all_expected_sources")]
    candidate_contains = [delta for delta in expected if delta.get("candidate_pool_contains_expected_source")]
    not_fixable = [
        delta["question_id"]
        for delta in expected
        if not delta.get("candidate_pool_contains_expected_source")
    ]
    return {
        "reranking_eval_summary_version": "reranking_eval_summary_v1",
        "created_at": utc_now_iso(),
        "questions_total": len(question_deltas),
        "expected_source_questions": total_expected,
        "candidate_top_n": candidate_top_n,
        "final_top_k": final_top_k,
        "baseline_hit_at_5": _rate(len(baseline_hits), total_expected),
        "reranked_hit_at_5": _rate(len(reranked_hits), total_expected),
        "baseline_hit_at_5_count": len(baseline_hits),
        "reranked_hit_at_5_count": len(reranked_hits),
        "baseline_hit_at_1": _rate(len(baseline_hit_at_1), total_expected),
        "reranked_hit_at_1": _rate(len(reranked_hit_at_1), total_expected),
        "baseline_hit_at_1_count": len(baseline_hit_at_1),
        "reranked_hit_at_1_count": len(reranked_hit_at_1),
        "baseline_mrr": _mrr(expected, "baseline_rank"),
        "reranked_mrr": _mrr(expected, "reranked_rank"),
        "baseline_retrieval_misses": [
            delta["question_id"] for delta in expected if not delta.get("baseline_hit_at_5")
        ],
        "reranked_retrieval_misses": [
            delta["question_id"] for delta in expected if not delta.get("reranked_hit_at_5")
        ],
        "improved_questions": improved,
        "worsened_questions": worsened,
        "unchanged_questions": unchanged,
        "all_expected_sources_hit_at_5": {
            "baseline": sum(1 for delta in required_all if delta.get("baseline_hit_at_5")),
            "reranked": sum(1 for delta in required_all if delta.get("reranked_hit_at_5")),
            "total": len(required_all),
        },
        "candidate_pool_contains_expected_source_count": len(candidate_contains),
        "candidate_pool_contains_expected_source": _rate(len(candidate_contains), total_expected),
        "cases_not_fixable_by_reranking": not_fixable,
        "delta_category_counts": dict(Counter(delta["delta_category"] for delta in question_deltas)),
        "no_ground_truth_used_for_reranking": True,
        "notes": (
            "Expected source identifiers are used only for post-hoc evaluation, "
            "not for reranking scores."
        ),
    }


def evaluate_retrieval(
    *,
    question: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """Evaluate retrieved chunks against expected IDs after ranking."""
    expected_arxiv_ids = [str(value) for value in question.get("expected_arxiv_ids", []) or []]
    expected_paper_ids = [str(value) for value in question.get("expected_paper_ids", []) or []]
    if not expected_arxiv_ids and not expected_paper_ids:
        return {
            "top_k": top_k,
            "has_expected_source": False,
            "expected_source_hit": False,
            "expected_source_hit_rank": None,
            "required_all_expected_sources": bool(question.get("required_all_expected_sources")),
            "matched_expected_source_ids": [],
            "missing_expected_source_ids": [],
            "notes": "No expected source IDs specified.",
        }

    required_all = bool(question.get("required_all_expected_sources"))
    expected_groups = [[arxiv_id, arxiv_id.replace(".", "_")] for arxiv_id in expected_arxiv_ids]
    if not expected_groups:
        expected_groups = [[paper_id] for paper_id in expected_paper_ids]
    matched: list[str] = []
    ranks: list[int] = []
    for chunk in retrieved_chunks:
        rank = int(chunk.get("reranked_rank") or chunk.get("rank") or chunk.get("original_rank") or 0)
        ids = {str(chunk.get("arxiv_id") or ""), str(chunk.get("paper_id") or "")}
        for group in expected_groups:
            if any(identifier in ids for identifier in group):
                matched.append(group[0])
                ranks.append(rank)

    matched_unique = _unique(matched)
    missing = [group[0] for group in expected_groups if group[0] not in set(matched_unique)]
    if required_all:
        hit = not missing
        hit_rank = max(ranks) if hit and ranks else None
    else:
        hit = bool(matched_unique)
        hit_rank = min(ranks) if ranks else None

    return {
        "top_k": top_k,
        "has_expected_source": True,
        "expected_source_hit": hit,
        "expected_source_hit_rank": hit_rank,
        "required_all_expected_sources": required_all,
        "matched_expected_source_ids": matched_unique,
        "missing_expected_source_ids": missing if required_all or not hit else [],
        "notes": (
            "Expected source hit."
            if hit
            else "Expected source absent or incomplete in retrieved set."
        ),
    }


def classify_delta(
    *,
    baseline_eval: dict[str, Any],
    reranked_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
    question: dict[str, Any],
    delta_rank: int | None,
) -> str:
    if question.get("expected_arxiv_ids") and not candidate_eval.get("expected_source_hit"):
        return "not_fixable_by_reranking"
    baseline_hit = bool(baseline_eval.get("expected_source_hit"))
    reranked_hit = bool(reranked_eval.get("expected_source_hit"))
    if not baseline_hit and reranked_hit:
        return "improved"
    if baseline_hit and not reranked_hit:
        return "worsened"
    if baseline_hit and reranked_hit:
        if delta_rank is not None and delta_rank < 0:
            return "improved"
        if delta_rank is not None and delta_rank > 0:
            return "worsened"
        return "unchanged_hit"
    return "unchanged_miss"


def error_tags_for_delta(
    *,
    question: dict[str, Any],
    category: str,
    baseline_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    candidate_eval: dict[str, Any],
    reranked_eval: dict[str, Any],
) -> list[str]:
    tags = []
    if category == "improved":
        tags.append("fixed_by_reranking")
    if category == "worsened":
        tags.append("worsened_by_reranking")
    if category == "unchanged_hit":
        tags.append("unchanged_hit")
    if category in {"unchanged_miss", "not_fixable_by_reranking"}:
        tags.append("unchanged_miss")
    if question.get("expected_arxiv_ids") and not candidate_eval.get("expected_source_hit"):
        tags.append("expected_source_absent_from_candidate_pool")
    if question.get("required_all_expected_sources") and reranked_eval.get("missing_expected_source_ids"):
        tags.append("multi_source_incomplete")
    if any(chunk.get("quality_flags") for chunk in baseline_results + reranked_results):
        tags.append("very_long_text_context_risk")
    if question.get("needs_manual_question_review") or question.get("needs_expected_source_review"):
        tags.append("likely_query_or_eval_issue")
    return _unique(tags)


def build_error_analysis_markdown(
    *,
    question_deltas: list[dict[str, Any]],
    eval_summary: dict[str, Any],
) -> str:
    lines = [
        "# Light Reranking v1 Error Analysis",
        "",
        "This analysis compares TF-IDF top-5 against lexical light reranking over a top-20 candidate pool.",
        "Expected source IDs are used only for evaluation after ranking.",
        "",
        "## Summary",
        "",
        f"- Baseline hit@5: {eval_summary['baseline_hit_at_5']:.3f}",
        f"- Reranked hit@5: {eval_summary['reranked_hit_at_5']:.3f}",
        f"- Baseline MRR: {eval_summary['baseline_mrr']:.3f}",
        f"- Reranked MRR: {eval_summary['reranked_mrr']:.3f}",
        f"- Improved questions: {', '.join(eval_summary['improved_questions']) or '<none>'}",
        f"- Worsened questions: {', '.join(eval_summary['worsened_questions']) or '<none>'}",
        f"- Not fixable by reranking: {', '.join(eval_summary['cases_not_fixable_by_reranking']) or '<none>'}",
        "",
        "## Sensitive Questions",
        "",
    ]
    by_id = {delta["question_id"]: delta for delta in question_deltas}
    for question_id in ("eval_v0_q002", "eval_v0_q005", "eval_v1_q012"):
        delta = by_id.get(question_id)
        if not delta:
            continue
        lines.extend(
            [
                f"### {question_id}",
                "",
                f"- Category: `{delta['delta_category']}`",
                f"- Baseline rank: {delta.get('baseline_rank')}",
                f"- Reranked rank: {delta.get('reranked_rank')}",
                f"- Candidate pool rank: {delta.get('candidate_pool_rank')}",
                f"- Tags: {', '.join(delta.get('error_analysis_tags', [])) or '<none>'}",
                "",
            ]
        )
    lines.extend(["## Categories", ""])
    for category in (
        "fixed_by_reranking",
        "worsened_by_reranking",
        "unchanged_hit",
        "unchanged_miss",
        "expected_source_absent_from_candidate_pool",
        "multi_source_incomplete",
        "very_long_text_context_risk",
        "likely_query_or_eval_issue",
    ):
        ids = [
            delta["question_id"]
            for delta in question_deltas
            if category in delta.get("error_analysis_tags", [])
        ]
        lines.append(f"- `{category}`: {', '.join(ids) if ids else '<none>'}")
    lines.extend(
        [
            "",
            "Light reranking can improve ordering of already retrieved candidates, but it cannot recover sources absent from the candidate pool and does not prove scientific answer quality.",
            "",
        ]
    )
    return "\n".join(lines)


def build_reranking_manifest(
    *,
    rag_ready_dir: Path,
    eval_questions_file: Path,
    output_dir: Path,
    report_path: Path,
    config_path: Path | None,
    output_paths: dict[str, Path],
    candidate_top_n: int,
    final_top_k: int,
    pytest_status: str | None,
) -> dict[str, Any]:
    return {
        "reranking_manifest_version": "reranking_manifest_v1",
        "created_at": utc_now_iso(),
        "config_path": str(config_path) if config_path else None,
        "input_files": {
            "chunks": str(rag_ready_dir / "chunks_v1.jsonl"),
            "rag_manifest": str(rag_ready_dir / "rag_manifest_v1.json"),
            "eval_questions": str(eval_questions_file),
        },
        "output_dir": str(output_dir),
        "output_files": {key: str(path) for key, path in output_paths.items()},
        "report_path": str(report_path),
        "retriever": {
            "name": TfidfRetrieverV1.name,
            "version": TfidfRetrieverV1.version,
        },
        "reranker": {
            "name": LightRerankerV1.name,
            "version": LightRerankerV1.version,
        },
        "candidate_top_n": candidate_top_n,
        "final_top_k": final_top_k,
        "no_ground_truth_used_for_reranking": True,
        "pytest_status": pytest_status,
        "known_limits": [
            "Light lexical reranking cannot recover expected sources absent from the candidate pool.",
            "Evaluation uses expected source IDs only after ranking.",
            "No LLM reranker or LLM judge is used.",
            "No training, parser changes, discovery, or corpus scaling were performed.",
        ],
    }


def build_task_report_markdown(
    *,
    rag_ready_dir: Path,
    eval_questions_file: Path,
    output_dir: Path,
    baseline_records: list[dict[str, Any]],
    question_deltas: list[dict[str, Any]],
    eval_summary: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    output_files = [f"- `{output_dir / filename}`" for filename in RERANKING_OUTPUT_FILES.values()]
    return "\n".join(
        [
            "# Phase 2 Task 06 - Light Reranking v1",
            "",
            f"Date: {utc_now_iso()[:10]}",
            "",
            "## Objective",
            "",
            "Implement and evaluate a lightweight deterministic reranker over TF-IDF candidates for the fixed FastCaloSimulation RAG corpus.",
            "",
            "Light reranking can improve ordering of already retrieved candidates, but it cannot recover sources absent from the candidate pool and does not prove scientific answer quality.",
            "",
            "## Strategy",
            "",
            "- Generate top-20 TF-IDF candidates per hardened eval question.",
            "- Rerank candidates using base score, lexical overlap, title/section overlap, question-type section priors, and quality penalties.",
            "- Use expected source IDs only after ranking for evaluation.",
            "- No training, LLM reranker, LLM judge, discovery, parser change, or corpus scaling is performed.",
            "",
            "## Inputs Used",
            "",
            f"- `{rag_ready_dir / 'chunks_v1.jsonl'}`",
            f"- `{rag_ready_dir / 'rag_manifest_v1.json'}`",
            f"- `{eval_questions_file}`",
            "",
            "## Outputs Produced",
            "",
            *output_files,
            f"- `docs/task_reports/phase2_task06_light_reranking_v1.md`",
            f"- `data/eval/rag_eval_set_v1_hardened.md`",
            "",
            "## Settings",
            "",
            f"- candidate_top_n: {eval_summary['candidate_top_n']}",
            f"- final_top_k: {eval_summary['final_top_k']}",
            "- no ground truth used for reranking: true",
            "",
            "## Metrics",
            "",
            f"- Questions total: {eval_summary['questions_total']}",
            f"- Expected-source questions: {eval_summary['expected_source_questions']}",
            f"- Baseline hit@5: {eval_summary['baseline_hit_at_5']:.3f} ({eval_summary['baseline_hit_at_5_count']})",
            f"- Reranked hit@5: {eval_summary['reranked_hit_at_5']:.3f} ({eval_summary['reranked_hit_at_5_count']})",
            f"- Baseline hit@1: {eval_summary['baseline_hit_at_1']:.3f} ({eval_summary['baseline_hit_at_1_count']})",
            f"- Reranked hit@1: {eval_summary['reranked_hit_at_1']:.3f} ({eval_summary['reranked_hit_at_1_count']})",
            f"- Baseline MRR: {eval_summary['baseline_mrr']:.3f}",
            f"- Reranked MRR: {eval_summary['reranked_mrr']:.3f}",
            f"- Candidate pool contains expected source count: {eval_summary['candidate_pool_contains_expected_source_count']}",
            "",
            "## Question-Level Changes",
            "",
            f"- Improved questions: {', '.join(eval_summary['improved_questions']) or '<none>'}",
            f"- Worsened questions: {', '.join(eval_summary['worsened_questions']) or '<none>'}",
            f"- Unchanged questions: {', '.join(eval_summary['unchanged_questions']) or '<none>'}",
            f"- Baseline retrieval misses: {', '.join(eval_summary['baseline_retrieval_misses']) or '<none>'}",
            f"- Reranked retrieval misses: {', '.join(eval_summary['reranked_retrieval_misses']) or '<none>'}",
            f"- Not fixable by reranking: {', '.join(eval_summary['cases_not_fixable_by_reranking']) or '<none>'}",
            "",
            "## Sensitive Cases",
            "",
            *_sensitive_case_lines(question_deltas),
            "",
            "## Multi-Source Questions",
            "",
            "- Required-all expected-source questions: eval_v0_q005, eval_v0_q006, eval_v1_q012",
            f"- All expected sources hit@5: `{json.dumps(eval_summary['all_expected_sources_hit_at_5'], sort_keys=True)}`",
            "",
            "## Limits",
            "",
            "- Reranking only reorders candidates already found by TF-IDF.",
            "- The reranker is lexical and deterministic; it is not a semantic model comparison.",
            "- Expected source IDs are not used during reranking.",
            "- Retrieval improvement does not validate generated answer quality.",
            f"- Dataset counts: `{json.dumps(_dataset_counts_from_manifest(manifest), sort_keys=True)}`",
            "",
            "## Recommended Next Step",
            "",
            "Inspect question-level deltas, especially multi-source questions, before deciding whether to plug reranked contexts into a new QA-grounded generation run.",
            "",
        ]
    )


def build_eval_set_hardened_markdown(eval_questions: list[dict[str, Any]]) -> str:
    """Build a human-friendly Markdown view from the hardened JSONL."""
    answerable = [q for q in eval_questions if q.get("expected_behavior") == "answerable"]
    insufficient = [q for q in eval_questions if q.get("expected_behavior") == "insufficient_evidence"]
    multi = [q for q in eval_questions if q.get("required_all_expected_sources")]
    manual = [q for q in eval_questions if q.get("needs_manual_question_review")]
    source_review = [q for q in eval_questions if q.get("needs_expected_source_review")]
    lines = [
        "# RAG Eval Set v1 Hardened",
        "",
        "## Summary",
        f"- Total questions: {len(eval_questions)}",
        f"- Answerable: {len(answerable)}",
        f"- Insufficient evidence: {len(insufficient)} ({', '.join(q['question_id'] for q in insufficient)})",
        f"- Multi-source questions: {len(multi)} ({', '.join(q['question_id'] for q in multi)})",
        f"- Questions needing manual review: {len(manual)} ({', '.join(q['question_id'] for q in manual)})",
        f"- Questions needing expected-source review: {len(source_review)} ({', '.join(q['question_id'] for q in source_review)})",
        "",
        "## Question table",
        "",
        "| question_id | question_type | expected_behavior | expected sources | required_all_expected_sources | needs_manual_question_review | question |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for question in eval_questions:
        sources = question.get("expected_arxiv_ids") or question.get("candidate_context_arxiv_ids") or []
        lines.append(
            "| {qid} | {qtype} | {behavior} | {sources} | {required} | {manual} | {text} |".format(
                qid=question.get("question_id", ""),
                qtype=question.get("question_type", ""),
                behavior=question.get("expected_behavior", ""),
                sources=", ".join(sources) if sources else "<none>",
                required=str(bool(question.get("required_all_expected_sources"))).lower(),
                manual=str(bool(question.get("needs_manual_question_review"))).lower(),
                text=str(question.get("question") or "").replace("|", "\\|"),
            )
        )
    lines.extend(["", "## Questions", ""])
    for question in eval_questions:
        sources = question.get("expected_arxiv_ids") or question.get("candidate_context_arxiv_ids") or []
        lines.extend(
            [
                f"### {question.get('question_id')} - {question.get('question_type')}",
                "",
                "Question:",
                str(question.get("question") or ""),
                "",
                "Expected behavior:",
                str(question.get("expected_behavior") or ""),
                "",
                "Expected sources:",
                ", ".join(sources) if sources else "<none>",
                "",
                "Required all expected sources:",
                str(bool(question.get("required_all_expected_sources"))).lower(),
                "",
                "Expected section keywords:",
                ", ".join(question.get("expected_section_keywords", []) or []) or "<none>",
                "",
                "Evaluation criteria:",
                str(question.get("evaluation_criteria") or ""),
                "",
                "Notes:",
                str(question.get("notes") or ""),
                "",
                "Review flags:",
                f"- needs_manual_question_review: {str(bool(question.get('needs_manual_question_review'))).lower()}",
                f"- needs_expected_source_review: {str(bool(question.get('needs_expected_source_review'))).lower()}",
                "",
            ]
        )
    return "\n".join(lines)


def _retrieved_candidate_dict(rank: int, score: float, chunk: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(chunk.get("text") or "").split())
    return {
        "rank": rank,
        "score": round(float(score), 8),
        "chunk_id": chunk.get("chunk_id", ""),
        "paper_id": chunk.get("paper_id", ""),
        "arxiv_id": chunk.get("arxiv_id", ""),
        "title": chunk.get("title", ""),
        "section": chunk.get("section") or chunk.get("section_heading", ""),
        "section_heading": chunk.get("section_heading") or chunk.get("section", ""),
        "source_type": chunk.get("source_type", ""),
        "quality_flags": list(chunk.get("quality_flags", []) or []),
        "text_preview": text[:360],
        "text": text,
    }


def _assign_citation(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    result = dict(chunk)
    result["citation_id"] = f"C{rank}"
    result["citation_label"] = f"[C{rank}]"
    return result


def _candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": candidate.get("chunk_id", ""),
        "paper_id": candidate.get("paper_id", ""),
        "arxiv_id": candidate.get("arxiv_id", ""),
        "title": candidate.get("title", ""),
        "section": candidate.get("section") or candidate.get("section_heading", ""),
        "section_heading": candidate.get("section_heading") or candidate.get("section", ""),
        "source_type": candidate.get("source_type", ""),
        "quality_flags": list(candidate.get("quality_flags", []) or []),
        "text_preview": candidate.get("text_preview", ""),
        "text": candidate.get("text", ""),
    }


def _ranking_question_view(question_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": question_record.get("question", ""),
        "question_type": question_record.get("question_type", ""),
    }


def _content_tokens(text: str) -> list[str]:
    return tokenize_v1(text)


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _overlap_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / len(left_set)


def _section_prior_score(*, question_type: str, section: str) -> float:
    priors = SECTION_PRIORS.get(question_type, ())
    if not priors:
        return 0.0
    section_tokens = set(_content_tokens(section))
    if not section_tokens:
        return 0.0
    return 1.0 if section_tokens & set(priors) else 0.0


def _quality_penalty(candidate: dict[str, Any]) -> float:
    flags = set(candidate.get("quality_flags", []) or [])
    penalty = 0.0
    if "very_long_text" in flags:
        penalty += 0.5
    penalty += 0.15 * max(0, len(flags) - (1 if "very_long_text" in flags else 0))
    return min(1.0, penalty)


def _rank_delta(baseline_rank: Any, reranked_rank: Any, final_top_k: int) -> int | None:
    if baseline_rank is None and reranked_rank is None:
        return None
    baseline_value = int(baseline_rank) if baseline_rank is not None else final_top_k + 1
    reranked_value = int(reranked_rank) if reranked_rank is not None else final_top_k + 1
    return reranked_value - baseline_value


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mrr(deltas: list[dict[str, Any]], key: str) -> float:
    if not deltas:
        return 0.0
    reciprocal = [1 / int(delta[key]) for delta in deltas if delta.get(key)]
    return sum(reciprocal) / len(deltas)


def _unique(values: list[Any]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _sensitive_case_lines(question_deltas: list[dict[str, Any]]) -> list[str]:
    by_id = {delta["question_id"]: delta for delta in question_deltas}
    lines: list[str] = []
    for question_id in ("eval_v0_q002", "eval_v0_q005", "eval_v1_q012"):
        delta = by_id.get(question_id)
        if not delta:
            continue
        lines.extend(
            [
                f"### {question_id}",
                "",
                f"- Category: `{delta['delta_category']}`",
                f"- Baseline rank: {delta.get('baseline_rank')}",
                f"- Reranked rank: {delta.get('reranked_rank')}",
                f"- Candidate pool rank: {delta.get('candidate_pool_rank')}",
                f"- Tags: {', '.join(delta.get('error_analysis_tags', [])) or '<none>'}",
                "",
            ]
        )
    return lines


def _dataset_counts_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "papers": manifest.get("papers_count"),
        "chunks_v1": manifest.get("chunks_v1_count"),
        "equations": manifest.get("equations_count"),
        "eval_questions_v0": manifest.get("eval_questions_count"),
        "chunks_with_quality_flags": (manifest.get("quality_summary") or {}).get("chunks_with_quality_flags"),
    }
