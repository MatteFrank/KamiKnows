"""Phase 2 Task 05 manual grounding review artifacts.

This module prepares deterministic review artifacts for human assessment of
QA-grounded v1 answers. It does not judge scientific correctness. All manual
review fields are initialized to ``pending``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from kamiknows.rag.load_dataset import read_json, read_jsonl
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.run_metadata import utc_now_iso


P2T5_OUTPUT_FILES = {
    "manual_review_queue": "manual_review_queue_v1.jsonl",
    "manual_review_sheet": "manual_review_sheet_v1.csv",
    "manual_review_guidelines": "manual_review_guidelines_v1.md",
    "eval_question_audit": "eval_question_audit_v1.jsonl",
    "eval_question_audit_summary": "eval_question_audit_summary_v1.json",
    "hardened_eval_draft": "rag_eval_set_v1_hardened_draft.jsonl",
    "hardening_notes": "rag_eval_set_hardening_notes_v1.md",
    "manual_review_summary": "manual_review_summary_v1.json",
}

DEFAULT_QA_OUTPUT_DIR = Path("outputs/rag_v1_fastcalosim_qa_grounded_config_v1")
DEFAULT_MANUAL_REVIEW_OUTPUT_DIR = Path("outputs/rag_v1_fastcalosim_manual_review")
DEFAULT_REPORT_PATH = Path("docs/task_reports/phase2_task05_manual_grounding_review_v1.md")

CSV_REVIEW_COLUMNS = [
    "review_id",
    "question_id",
    "question_type",
    "model_name",
    "expected_behavior",
    "answer_status",
    "citation_validation_status",
    "retrieval_expected_source_hit",
    "manual_grounding",
    "manual_answer_correctness",
    "manual_citation_support",
    "manual_abstention_behavior",
    "severity",
    "reviewer_notes",
]


def run_manual_grounding_review_v1(
    *,
    qa_output_dir: Path = DEFAULT_QA_OUTPUT_DIR,
    output_dir: Path = DEFAULT_MANUAL_REVIEW_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    answer_files: list[Path | str | dict[str, Any]] | None = None,
    eval_set_file: Path | str | None = None,
    retrieved_contexts_file: Path | str | None = None,
    citation_audit_file: Path | str | None = None,
    citation_validation_summary_file: Path | str | None = None,
    rag_eval_summary_file: Path | str | None = None,
    chunks_file: Path | str | None = None,
    review_schema_options: dict[str, Any] | None = None,
    hardening_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create all Phase 2 Task 05 review and hardening artifacts."""
    qa_output_dir = Path(qa_output_dir)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_answer_files = _resolve_answer_files(qa_output_dir, answer_files)
    paths = {
        "eval_set": _resolve_path(qa_output_dir, eval_set_file, "rag_eval_set_v1.jsonl"),
        "retrieved_contexts": _resolve_path(
            qa_output_dir,
            retrieved_contexts_file,
            "retrieved_contexts_v1.jsonl",
        ),
        "citation_audit": _resolve_path(qa_output_dir, citation_audit_file, "citation_audit_v1.jsonl"),
        "citation_validation_summary": _resolve_path(
            qa_output_dir,
            citation_validation_summary_file,
            "citation_validation_summary_v1.json",
        ),
        "rag_eval_summary": _resolve_path(qa_output_dir, rag_eval_summary_file, "rag_eval_summary_v1.json"),
        "chunks": Path(chunks_file) if chunks_file is not None else Path("outputs/rag_ready_fastcalo_v1/chunks_v1.jsonl"),
    }

    eval_set = read_jsonl(paths["eval_set"])
    retrieved_contexts = read_jsonl(paths["retrieved_contexts"])
    citation_audit = read_jsonl(paths["citation_audit"])
    citation_summary = read_json(paths["citation_validation_summary"])
    rag_eval_summary = read_json(paths["rag_eval_summary"])
    chunks = read_jsonl(paths["chunks"]) if paths["chunks"].exists() else []
    answer_records = _read_answer_records(resolved_answer_files)

    queue = build_manual_review_queue(
        eval_set=eval_set,
        retrieved_contexts=retrieved_contexts,
        citation_audit=citation_audit,
        answer_records=answer_records,
    )
    eval_audit = build_eval_question_audit(eval_set=eval_set, retrieved_contexts=retrieved_contexts)
    eval_audit_summary = build_eval_question_audit_summary(eval_audit)
    hardened_draft = build_hardened_eval_set_draft(eval_set=eval_set, eval_question_audit=eval_audit)
    manual_summary = build_manual_review_summary(
        queue=queue,
        eval_question_audit=eval_audit,
        citation_summary=citation_summary,
        rag_eval_summary=rag_eval_summary,
    )

    output_paths = {
        key: output_dir / filename
        for key, filename in P2T5_OUTPUT_FILES.items()
    }
    write_jsonl(output_paths["manual_review_queue"], queue)
    write_manual_review_csv(output_paths["manual_review_sheet"], queue)
    output_paths["manual_review_guidelines"].write_text(
        build_manual_review_guidelines_markdown(),
        encoding="utf-8",
    )
    write_jsonl(output_paths["eval_question_audit"], eval_audit)
    write_json(output_paths["eval_question_audit_summary"], eval_audit_summary)
    write_jsonl(output_paths["hardened_eval_draft"], hardened_draft)
    output_paths["hardening_notes"].write_text(
        build_hardening_notes_markdown(eval_audit=eval_audit, hardened_draft=hardened_draft),
        encoding="utf-8",
    )
    write_json(output_paths["manual_review_summary"], manual_summary)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_task05_report_markdown(
            qa_output_dir=qa_output_dir,
            output_dir=output_dir,
            report_path=report_path,
            answer_files=resolved_answer_files,
            input_paths=paths,
            queue=queue,
            eval_question_audit=eval_audit,
            manual_summary=manual_summary,
            citation_summary=citation_summary,
            rag_eval_summary=rag_eval_summary,
            chunks=chunks,
            review_schema_options=review_schema_options or {},
            hardening_settings=hardening_settings or {},
        ),
        encoding="utf-8",
    )

    return {
        "qa_output_dir": str(qa_output_dir),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "input_paths": {key: str(value) for key, value in paths.items()},
        "answer_files": [str(path) for path in resolved_answer_files],
        "manual_review_queue": queue,
        "eval_question_audit": eval_audit,
        "eval_question_audit_summary": eval_audit_summary,
        "hardened_eval_draft": hardened_draft,
        "manual_review_summary": manual_summary,
        "output_paths": {key: str(value) for key, value in output_paths.items()},
    }


def build_manual_review_queue(
    *,
    eval_set: list[dict[str, Any]],
    retrieved_contexts: list[dict[str, Any]],
    citation_audit: list[dict[str, Any]],
    answer_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one pending manual-review record per answer."""
    question_by_id = {str(question.get("question_id") or ""): question for question in eval_set}
    context_by_id = {str(context.get("question_id") or ""): context for context in retrieved_contexts}
    audit_by_key = {
        (str(record.get("question_id") or ""), str(record.get("model_name") or "")): record
        for record in citation_audit
    }

    queue: list[dict[str, Any]] = []
    for answer in answer_records:
        question_id = str(answer.get("question_id") or "")
        model_name = str(answer.get("model_name") or "")
        question = question_by_id.get(question_id, {})
        context = context_by_id.get(question_id, {})
        audit = audit_by_key.get((question_id, model_name), {})
        retrieval_eval = answer.get("retrieval_eval") or context.get("retrieval_eval") or {}
        citation_validation = answer.get("citation_validation") or _citation_validation_from_audit(audit)
        retrieved_chunks = answer.get("retrieved_chunks") or context.get("retrieved_chunks") or []
        warnings = _combined_warnings(answer=answer, audit=audit, retrieval_eval=retrieval_eval)
        model_slug = str(answer.get("model_slug") or _slug(model_name))

        record = {
            "review_id": f"{question_id}__{model_slug}",
            "question_id": question_id,
            "question": answer.get("question") or question.get("question", ""),
            "question_type": answer.get("question_type") or question.get("question_type", ""),
            "expected_behavior": answer.get("expected_behavior") or question.get("expected_behavior", ""),
            "model_name": model_name,
            "answer_status": answer.get("answer_status", ""),
            "answer": answer.get("answer", ""),
            "evidence": answer.get("evidence", ""),
            "citations": list(answer.get("citations", []) or []),
            "limitations": answer.get("limitations", ""),
            "retrieved_chunks": retrieved_chunks,
            "retrieval_eval": retrieval_eval,
            "citation_validation": citation_validation,
            "warnings": warnings,
            "suggested_failure_modes": suggest_failure_modes(
                question=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                retrieval_eval=retrieval_eval,
                citation_validation=citation_validation,
                warnings=warnings,
            ),
            "manual_review_status": "pending",
            "manual_grounding": "pending",
            "manual_answer_correctness": "pending",
            "manual_citation_support": "pending",
            "manual_notes": "",
        }
        queue.append(record)
    return queue


def write_manual_review_csv(path: Path, queue: list[dict[str, Any]]) -> Path:
    """Write a compact CSV sheet for human review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_REVIEW_COLUMNS)
        writer.writeheader()
        for record in queue:
            retrieval_eval = record.get("retrieval_eval", {}) or {}
            citation_validation = record.get("citation_validation", {}) or {}
            writer.writerow(
                {
                    "review_id": record.get("review_id", ""),
                    "question_id": record.get("question_id", ""),
                    "question_type": record.get("question_type", ""),
                    "model_name": record.get("model_name", ""),
                    "expected_behavior": record.get("expected_behavior", ""),
                    "answer_status": record.get("answer_status", ""),
                    "citation_validation_status": citation_validation.get("status", ""),
                    "retrieval_expected_source_hit": _csv_bool_or_na(retrieval_eval),
                    "manual_grounding": "pending",
                    "manual_answer_correctness": "pending",
                    "manual_citation_support": "pending",
                    "manual_abstention_behavior": "pending",
                    "severity": "pending",
                    "reviewer_notes": "",
                }
            )
    return path


def build_eval_question_audit(
    *,
    eval_set: list[dict[str, Any]],
    retrieved_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Audit eval questions for retrieval and hardening concerns."""
    context_by_id = {str(context.get("question_id") or ""): context for context in retrieved_contexts}
    records = []
    for question in eval_set:
        question_id = str(question.get("question_id") or "")
        context = context_by_id.get(question_id, {})
        retrieval_eval = context.get("retrieval_eval", {}) or {}
        expected_ids = [str(value) for value in question.get("expected_arxiv_ids", []) or []]
        retrieved_ids = _unique(
            str(chunk.get("arxiv_id") or "")
            for chunk in context.get("retrieved_chunks", []) or []
            if chunk.get("arxiv_id")
        )
        missing_expected_ids = [value for value in expected_ids if value not in set(retrieved_ids)]
        ambiguous_wording = _has_ambiguous_wording(str(question.get("question") or ""))
        multi_source_required = bool(len(expected_ids) > 1 or question.get("question_type") == "comparison")
        no_answer_question = _is_no_answer_question(question)
        retrieval_hit = bool(retrieval_eval.get("expected_source_hit"))
        needs_expected_source_review = bool(
            (retrieval_eval.get("has_expected_source") and not retrieval_hit)
            or (multi_source_required and missing_expected_ids)
            or (no_answer_question and expected_ids)
        )
        needs_rewrite = bool(
            ambiguous_wording
            or needs_expected_source_review
            or (multi_source_required and missing_expected_ids)
            or (no_answer_question and str(question.get("question_type") or "") == "unanswerable")
        )
        notes = _eval_question_notes(
            question=question,
            retrieval_eval=retrieval_eval,
            missing_expected_ids=missing_expected_ids,
            ambiguous_wording=ambiguous_wording,
            multi_source_required=multi_source_required,
            no_answer_question=no_answer_question,
        )

        records.append(
            {
                "question_id": question_id,
                "question_type": question.get("question_type", ""),
                "expected_behavior": question.get("expected_behavior", ""),
                "expected_arxiv_ids": expected_ids,
                "retrieval_hit": retrieval_hit,
                "retrieval_hit_rank": retrieval_eval.get("expected_source_hit_rank"),
                "ambiguous_wording": ambiguous_wording,
                "multi_source_required": multi_source_required,
                "no_answer_question": no_answer_question,
                "needs_expected_source_review": needs_expected_source_review,
                "needs_rewrite": needs_rewrite,
                "notes": " ".join(notes),
                "retrieved_arxiv_ids": retrieved_ids,
                "missing_expected_arxiv_ids": missing_expected_ids,
                "all_expected_sources_retrieved": bool(expected_ids and not missing_expected_ids),
            }
        )
    return records


def build_eval_question_audit_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize eval-question audit flags."""
    retrieval_misses = [
        record["question_id"]
        for record in records
        if record.get("expected_arxiv_ids") and not record.get("retrieval_hit")
    ]
    return {
        "eval_question_audit_summary_version": "eval_question_audit_summary_v1",
        "created_at": utc_now_iso(),
        "total_questions": len(records),
        "retrieval_miss_questions": retrieval_misses,
        "ambiguous_wording_count": sum(1 for record in records if record.get("ambiguous_wording")),
        "multi_source_required_count": sum(1 for record in records if record.get("multi_source_required")),
        "no_answer_question_count": sum(1 for record in records if record.get("no_answer_question")),
        "needs_expected_source_review_count": sum(
            1 for record in records if record.get("needs_expected_source_review")
        ),
        "needs_rewrite_count": sum(1 for record in records if record.get("needs_rewrite")),
        "questions_needing_rewrite": [
            record["question_id"] for record in records if record.get("needs_rewrite")
        ],
        "questions_needing_expected_source_review": [
            record["question_id"] for record in records if record.get("needs_expected_source_review")
        ],
        "notes": (
            "Audit flags are deterministic review prompts, not final judgments. "
            "eval_v0_q002 and eval_v0_q005 are retrieval misses in the P2T4 run."
        ),
    }


def build_hardened_eval_set_draft(
    *,
    eval_set: list[dict[str, Any]],
    eval_question_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create a draft eval-set hardening file without changing ground truth."""
    audit_by_id = {str(record.get("question_id") or ""): record for record in eval_question_audit}
    draft = []
    for question in eval_set:
        question_copy = json.loads(json.dumps(question))
        question_id = str(question_copy.get("question_id") or "")
        audit = audit_by_id.get(question_id, {})
        suggested_revision = _suggest_question_revision(question_copy, audit)
        flags = []
        for key in (
            "ambiguous_wording",
            "multi_source_required",
            "no_answer_question",
            "needs_expected_source_review",
            "needs_rewrite",
        ):
            if audit.get(key):
                flags.append(key)
        question_copy.update(
            {
                "hardening_status": "draft_not_validated",
                "suggested_revision": suggested_revision,
                "hardening_flags": flags,
                "expected_source_review_required": bool(audit.get("needs_expected_source_review")),
                "ground_truth_change_proposed": False,
                "hardening_notes": audit.get("notes", ""),
            }
        )
        draft.append(question_copy)
    return draft


def build_manual_review_summary(
    *,
    queue: list[dict[str, Any]],
    eval_question_audit: list[dict[str, Any]],
    citation_summary: dict[str, Any],
    rag_eval_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the initial summary while all human review remains pending."""
    status_counts = Counter(str(record.get("manual_review_status") or "unknown") for record in queue)
    failure_mode_counts = Counter(
        mode
        for record in queue
        for mode in record.get("suggested_failure_modes", []) or []
    )
    models = _unique(str(record.get("model_name") or "") for record in queue if record.get("model_name"))
    questions = _unique(str(record.get("question_id") or "") for record in queue if record.get("question_id"))
    retrieval_miss_questions = [
        record["question_id"]
        for record in eval_question_audit
        if record.get("expected_arxiv_ids") and not record.get("retrieval_hit")
    ]
    citation_validation_clean = bool(
        int(citation_summary.get("invalid_citation_count") or 0) == 0
        and int(citation_summary.get("invented_citation_count") or 0) == 0
        and int(citation_summary.get("answered_without_valid_citation_count") or 0) == 0
    )
    return {
        "manual_review_summary_version": "manual_review_summary_v1",
        "created_at": utc_now_iso(),
        "total_review_records": len(queue),
        "models": models,
        "questions": len(questions),
        "question_ids": questions,
        "manual_review_status_counts": dict(sorted(status_counts.items())),
        "pending_count": status_counts.get("pending", 0),
        "retrieval_miss_questions": retrieval_miss_questions,
        "citation_validation_clean": citation_validation_clean,
        "citation_validation_status_counts": citation_summary.get("status_counts", {}),
        "known_failure_modes": dict(sorted(failure_mode_counts.items())),
        "p2t4_retrieval_miss_questions": rag_eval_summary.get("retrieval_miss_question_ids", []),
        "next_actions": [
            "Have a human reviewer complete manual_review_sheet_v1.csv.",
            "Inspect eval_v0_q002 and eval_v0_q005 as P2T4 retrieval misses.",
            "Review comparison questions for coverage of every expected source.",
            "Only after human review, summarize answer quality and grounding outcomes.",
        ],
        "scope": (
            "Initial manual-review artifact summary only; pending human review means "
            "answer quality is not validated."
        ),
    }


def suggest_failure_modes(
    *,
    question: dict[str, Any],
    answer: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    retrieval_eval: dict[str, Any],
    citation_validation: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    """Suggest review prompts without making final scientific judgments."""
    modes: list[str] = []
    expected_behavior = str(answer.get("expected_behavior") or question.get("expected_behavior") or "")
    answer_status = str(answer.get("answer_status") or "")
    citation_status = str(citation_validation.get("status") or "")
    expected_ids = [str(value) for value in question.get("expected_arxiv_ids", []) or []]
    retrieved_ids = {
        str(chunk.get("arxiv_id") or "")
        for chunk in retrieved_chunks
        if chunk.get("arxiv_id")
    }
    question_type = str(question.get("question_type") or answer.get("question_type") or "")

    if retrieval_eval.get("has_expected_source") and not retrieval_eval.get("expected_source_hit"):
        modes.append("retrieval_miss")
    if (question_type == "comparison" or len(expected_ids) > 1) and any(
        expected_id not in retrieved_ids for expected_id in expected_ids
    ):
        modes.append("multi_source_retrieval_incomplete")
    if expected_behavior == "insufficient_evidence" and answer_status == "answered":
        modes.append("possible_overclaim_or_failed_abstention")
    if expected_behavior == "answerable" and answer_status == "insufficient_evidence":
        modes.append("possible_incomplete_answer_or_over_abstention")
    if citation_status not in {"", "valid", "insufficient_evidence_no_citation", "not_applicable_backend_error"}:
        modes.append(f"citation_validation_{citation_status}")
    if any(chunk.get("quality_flags") for chunk in retrieved_chunks):
        modes.append("retrieved_context_quality_flags")
    if warnings:
        modes.append("generation_or_audit_warning")
    return _unique(modes)


def build_manual_review_guidelines_markdown() -> str:
    """Return reviewer-facing guidance for P2T5."""
    return "\n".join(
        [
            "# Manual Review Guidelines v1",
            "",
            "Use this guide to fill `manual_review_sheet_v1.csv`. The automated files prepare evidence for review; they do not validate scientific correctness.",
            "",
            "## Grounded",
            "",
            "Mark grounding as `pass` only when every substantive claim in the answer is supported by the retrieved chunks shown for that review record. Use `partial` when the main idea is supported but some details are missing or too broad. Use `fail` when the answer relies on unsupported claims, outside knowledge, or a wrong source.",
            "",
            "## Citation Support",
            "",
            "Citation support asks whether the cited chunks actually support the sentence they are attached to. A citation can be formally valid because `[C1]` exists in the retrieved context, while still being scientifically unhelpful or mismatched for the claim. Formal citation validation is therefore not the same as manual citation support.",
            "",
            "## Answerable vs Insufficient Evidence",
            "",
            "For `expected_behavior=answerable`, check whether the retrieved chunks contain enough evidence to answer. If the answer abstains despite sufficient evidence, mark abstention behavior as `fail` or `partial`. For `expected_behavior=insufficient_evidence`, a good answer should clearly abstain and avoid policy, deployment, or detector-wide claims not present in the chunks.",
            "",
            "## Overclaim",
            "",
            "Classify overclaim when the answer states more than the retrieved evidence supports, generalizes beyond the cited paper, makes production-readiness claims, or turns a limited validation result into a broad detector-simulation conclusion.",
            "",
            "## Incomplete Answer",
            "",
            "Classify incomplete answer when the answer misses a required method, limitation, comparison side, or named model that is present in the retrieved chunks. For comparison questions, check whether both expected sources are represented and used.",
            "",
            "## Retrieval Miss",
            "",
            "When `retrieval_expected_source_hit=false`, review answer quality cautiously. The generator may cite valid labels from the retrieved context while still answering the wrong expected source. Mark grounding against retrieved chunks separately from answer correctness against the question intent.",
            "",
            "## CSV Fields",
            "",
            "Use only these labels for manual fields: `pass`, `partial`, `fail`, or `pending`. Use severity `none`, `minor`, `major`, `critical`, or `pending`. Put concise rationale in `reviewer_notes`, especially for any `partial`, `fail`, `major`, or `critical` row.",
            "",
        ]
    )


def build_hardening_notes_markdown(
    *,
    eval_audit: list[dict[str, Any]],
    hardened_draft: list[dict[str, Any]],
) -> str:
    """Build Markdown notes for eval-set hardening."""
    retrieval_misses = [
        record["question_id"]
        for record in eval_audit
        if record.get("expected_arxiv_ids") and not record.get("retrieval_hit")
    ]
    rewrite_ids = [record["question_id"] for record in eval_audit if record.get("needs_rewrite")]
    source_review_ids = [
        record["question_id"]
        for record in eval_audit
        if record.get("needs_expected_source_review")
    ]
    lines = [
        "# RAG Eval Set Hardening Notes v1",
        "",
        "This is a draft hardening pass. It preserves original `question_id` values and does not automatically change ground truth.",
        "",
        "## Summary",
        "",
        f"- Questions audited: {len(eval_audit)}",
        f"- Retrieval misses: {', '.join(retrieval_misses) if retrieval_misses else '<none>'}",
        f"- Questions needing expected-source review: {', '.join(source_review_ids) if source_review_ids else '<none>'}",
        f"- Questions with suggested rewrite/review: {', '.join(rewrite_ids) if rewrite_ids else '<none>'}",
        "",
        "## Required Attention",
        "",
        "- `eval_v0_q002` and `eval_v0_q005` are P2T4 retrieval misses and should be rewritten or reviewed before using them as answer-quality evidence.",
        "- Comparison questions should state whether both expected sources must be retrieved and cited.",
        "- No-answer questions should make the expected abstention behavior explicit.",
        "",
        "## Draft Revisions",
        "",
    ]
    for record in hardened_draft:
        revision = record.get("suggested_revision")
        if revision:
            lines.append(f"- `{record.get('question_id')}`: {revision}")
    if not any(record.get("suggested_revision") for record in hardened_draft):
        lines.append("- <none>")
    lines.extend(
        [
            "",
            "Manual review remains required before accepting any rewrite or ground-truth change.",
            "",
        ]
    )
    return "\n".join(lines)


def build_task05_report_markdown(
    *,
    qa_output_dir: Path,
    output_dir: Path,
    report_path: Path,
    answer_files: list[Path],
    input_paths: dict[str, Path],
    queue: list[dict[str, Any]],
    eval_question_audit: list[dict[str, Any]],
    manual_summary: dict[str, Any],
    citation_summary: dict[str, Any],
    rag_eval_summary: dict[str, Any],
    chunks: list[dict[str, Any]],
    review_schema_options: dict[str, Any],
    hardening_settings: dict[str, Any],
) -> str:
    """Build the required P2T5 task report."""
    models = manual_summary.get("models", [])
    retrieval_misses = manual_summary.get("retrieval_miss_questions", [])
    problematic_questions = [
        record["question_id"]
        for record in eval_question_audit
        if record.get("needs_rewrite") or record.get("needs_expected_source_review")
    ]
    output_lines = [f"- `{output_dir / filename}`" for filename in P2T5_OUTPUT_FILES.values()]
    output_lines.extend(
        [
            "- `data/eval/rag_eval_questions_v1_static.jsonl`",
            "- `configs/phase2_p2t5_manual_review_v1.json`",
        ]
    )
    input_lines = [
        f"- `{input_paths['eval_set']}`",
        f"- `{input_paths['retrieved_contexts']}`",
        f"- `{input_paths['citation_audit']}`",
        f"- `{input_paths['citation_validation_summary']}`",
        f"- `{input_paths['rag_eval_summary']}`",
        f"- `{input_paths['chunks']}`",
    ]
    input_lines.extend(f"- `{path}`" for path in answer_files)
    return "\n".join(
        [
            "# Phase 2 Task 05 - Manual Grounding Review and Eval Set Hardening v1",
            "",
            f"Date: {utc_now_iso()[:10]}",
            "",
            "## Objective",
            "",
            "Create manual review artifacts for P2T4 generated answers, classify candidate grounding and answer-quality failure modes for human review, harden the eval set draft, and prepare static v1 eval questions as data.",
            "",
            "Manual review artifacts support groundedness assessment, but pending human review means answer quality is not yet validated.",
            "",
            "## Inputs P2T4 Used",
            "",
            f"QA output directory: `{qa_output_dir}`",
            "",
            *input_lines,
            "",
            "## Outputs Produced",
            "",
            *output_lines,
            f"- `{report_path}`",
            "",
            "## Review Records",
            "",
            f"- Responses to review: {len(queue)}",
            f"- Questions: {manual_summary.get('questions', 0)}",
            f"- Models involved: {', '.join(models) if models else '<none>'}",
            f"- Manual review status: {json.dumps(manual_summary.get('manual_review_status_counts', {}), sort_keys=True)}",
            "",
            "## Citation Status",
            "",
            f"- Citation validation clean: {manual_summary.get('citation_validation_clean')}",
            f"- Invalid citation records: {citation_summary.get('invalid_citation_count', 0)}",
            f"- Invented citation labels: {citation_summary.get('invented_citation_count', 0)}",
            f"- Answered without valid citation: {citation_summary.get('answered_without_valid_citation_count', 0)}",
            f"- Status counts: `{json.dumps(citation_summary.get('status_counts', {}), sort_keys=True)}`",
            "",
            "## Problematic Questions",
            "",
            f"- Questions needing rewrite or expected-source review: {', '.join(problematic_questions) if problematic_questions else '<none>'}",
            f"- Retrieval misses: {', '.join(retrieval_misses) if retrieval_misses else '<none>'}",
            "- Retrieval misses highlighted for P2T5: eval_v0_q002, eval_v0_q005",
            "",
            "## What Was Automated",
            "",
            "- Built a pending review queue with one record per model/question answer.",
            "- Built a CSV sheet for human completion.",
            "- Joined answer records with retrieved chunks, retrieval eval, citation validation, and citation audit data.",
            "- Flagged deterministic candidate failure modes such as retrieval misses, source-coverage gaps, generation warnings, and context quality flags.",
            "- Audited eval questions for ambiguous wording, multi-source requirements, no-answer behavior, and rewrite needs.",
            "- Produced a hardened eval-set draft without changing ground truth automatically.",
            "",
            "## What Remains Manual",
            "",
            "- Scientific groundedness assessment.",
            "- Answer correctness assessment.",
            "- Citation support assessment beyond formal label validity.",
            "- Severity and reviewer notes.",
            "- Acceptance of any eval-set rewrite or expected-source change.",
            "",
            "## Key Metrics",
            "",
            f"- P2T4 expected-source hit rate: {rag_eval_summary.get('expected_source_hit_rate')}",
            f"- P2T4 expected-source hits: {rag_eval_summary.get('expected_source_hits')} / {rag_eval_summary.get('expected_source_questions')}",
            f"- Chunks available for traceability: {len(chunks)}",
            f"- Known candidate failure modes: `{json.dumps(manual_summary.get('known_failure_modes', {}), sort_keys=True)}`",
            "",
            "## Config Notes",
            "",
            f"- Review schema options: `{json.dumps(review_schema_options, sort_keys=True)}`",
            f"- Hardening settings: `{json.dumps(hardening_settings, sort_keys=True)}`",
            "",
            "## Limits",
            "",
            "- No training, LoRA, QLoRA, discovery, corpus scaling, parser changes, or chunks_v1 changes were performed.",
            "- P2T4 outputs and RAG-ready v0/v1 outputs were read, not overwritten.",
            "- Automated citation validation checks labels only, not factual entailment.",
            "- Suggested failure modes are triage hints for a human reviewer, not final judgments.",
            "- All scientific judgments remain pending until a human reviewer fills the sheet.",
            "",
            "## Recommended Next Step",
            "",
            "Complete `manual_review_sheet_v1.csv`, then summarize human labels before claiming any answer-quality or groundedness improvement.",
            "",
        ]
    )


def _resolve_answer_files(
    qa_output_dir: Path,
    answer_files: list[Path | str | dict[str, Any]] | None,
) -> list[Path]:
    if answer_files:
        return [
            Path(item.get("path") if isinstance(item, dict) else item)
            for item in answer_files
        ]
    return sorted(qa_output_dir.glob("rag_answers_*_v1.jsonl"))


def _resolve_path(qa_output_dir: Path, value: Path | str | None, default_name: str) -> Path:
    return Path(value) if value is not None else qa_output_dir / default_name


def _read_answer_records(answer_files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in answer_files:
        records.extend(read_jsonl(path))
    return records


def _citation_validation_from_audit(audit: dict[str, Any]) -> dict[str, Any]:
    if not audit:
        return {}
    return {
        "status": audit.get("citation_validation_status", ""),
        "valid": audit.get("citation_validation_valid"),
        "invented_citation_labels": audit.get("invented_citation_labels", []),
        "used_citation_labels": audit.get("used_citation_labels", []),
        "valid_used_citation_labels": audit.get("valid_used_citation_labels", []),
        "answered_without_valid_citation": audit.get("answered_without_valid_citation"),
        "cited_chunk_ids": audit.get("cited_chunk_ids", []),
    }


def _combined_warnings(
    *,
    answer: dict[str, Any],
    audit: dict[str, Any],
    retrieval_eval: dict[str, Any],
) -> list[str]:
    warnings = []
    warnings.extend(str(value) for value in answer.get("warnings", []) or [])
    warnings.extend(str(value) for value in audit.get("warnings", []) or [])
    if retrieval_eval.get("has_expected_source") and not retrieval_eval.get("expected_source_hit"):
        warnings.append("retrieval_expected_source_miss")
    return _unique(warnings)


def _csv_bool_or_na(retrieval_eval: dict[str, Any]) -> str:
    if not retrieval_eval.get("has_expected_source"):
        return "not_applicable"
    return "true" if retrieval_eval.get("expected_source_hit") else "false"


def _has_ambiguous_wording(question_text: str) -> bool:
    lowered = question_text.lower()
    markers = [
        "first selected",
        "another selected",
        "two selected",
        "two later",
        "later fastcalo",
        "selected source",
        "these five papers",
        "these chunks",
        "another paper",
    ]
    return any(marker in lowered for marker in markers)


def _is_no_answer_question(question: dict[str, Any]) -> bool:
    question_type = str(question.get("question_type") or "")
    expected_behavior = str(question.get("expected_behavior") or "")
    return question_type in {"no_answer", "unanswerable"} or expected_behavior == "insufficient_evidence"


def _eval_question_notes(
    *,
    question: dict[str, Any],
    retrieval_eval: dict[str, Any],
    missing_expected_ids: list[str],
    ambiguous_wording: bool,
    multi_source_required: bool,
    no_answer_question: bool,
) -> list[str]:
    notes: list[str] = []
    question_id = str(question.get("question_id") or "")
    if question_id in {"eval_v0_q002", "eval_v0_q005"}:
        notes.append("P2T4 retrieval miss; inspect expected source and wording.")
    if ambiguous_wording:
        notes.append("Question wording contains relative or corpus-position phrasing.")
    if retrieval_eval.get("has_expected_source") and not retrieval_eval.get("expected_source_hit"):
        notes.append("No retrieved chunk matched the expected source identifiers.")
    if multi_source_required and missing_expected_ids:
        notes.append("Multi-source question is missing expected retrieved sources: " + ", ".join(missing_expected_ids) + ".")
    if no_answer_question:
        notes.append("Expected behavior is abstention/insufficient evidence; do not score answer correctness automatically.")
    if not notes:
        notes.append("No deterministic hardening issue flagged.")
    return notes


def _suggest_question_revision(question: dict[str, Any], audit: dict[str, Any]) -> str | None:
    question_id = str(question.get("question_id") or "")
    expected_ids = [str(value) for value in question.get("expected_arxiv_ids", []) or []]
    if question_id == "eval_v0_q002":
        return (
            "For arXiv 1712.10321v1, which neural or statistical modeling method is used "
            "for calorimeter shower generation? Answer only from that expected source."
        )
    if question_id == "eval_v0_q005":
        return (
            "Compare the modeling goals of the 1705.02355v2 and 1712.10321v1 FastCalo papers. "
            "Cite evidence from both expected sources or state insufficient evidence if either source is missing."
        )
    if question_id == "eval_v1_q012":
        return (
            "Compare CaloGAN in 1705.02355v2 with CaloFlow in 2106.05285v3, citing evidence "
            "from both papers and separating GAN-based simulation from normalizing-flow advantages."
        )
    if question_id == "eval_v0_q008":
        return (
            "Using only the five selected papers, is there enough evidence to recommend an exact "
            "production-ready detector simulation replacement? Expected behavior: insufficient_evidence."
        )
    if question_id == "eval_v1_q013":
        return (
            "Do the retrieved chunks provide an exact ATLAS Run 4 production deployment date and "
            "detector-wide replacement policy? Expected behavior: insufficient_evidence."
        )
    if audit.get("ambiguous_wording") and expected_ids:
        return (
            f"Rewrite to name the expected source arXiv ID(s) explicitly: {', '.join(expected_ids)}."
        )
    if audit.get("multi_source_required") and audit.get("missing_expected_arxiv_ids"):
        return "Clarify that all expected sources must be retrieved and cited for a complete answer."
    return None


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace(":", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _unique(values) -> list[str]:
    seen = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
