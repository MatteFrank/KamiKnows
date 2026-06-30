"""Retrieval baseline v1 orchestration and reporting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from kamiknows.rag.evaluation import evaluate_expected_source_hit
from kamiknows.rag.load_dataset import load_rag_ready_dataset, read_json
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1, tokenize_v1
from kamiknows.run_metadata import utc_now_iso

REQUIRED_OUTPUT_FILES = {
    "retriever_baseline": "retriever_baseline_v1.jsonl",
    "retriever_eval_summary": "retriever_eval_summary_v1.json",
    "retriever_error_analysis": "retriever_error_analysis_v1.md",
    "retrieval_debug_report": "retrieval_debug_report.md",
    "retrieval_run_config": "retrieval_run_config.json",
    "retrieval_manifest": "retrieval_manifest.json",
}

MISS_REASONS = (
    "ambiguous_question",
    "lexical_mismatch",
    "chunk_missing_information",
    "expected_source_too_strict",
    "retriever_weakness",
    "metadata_problem",
    "unknown",
)


def build_baseline_records(
    *,
    chunks: list[dict[str, Any]],
    eval_questions: list[dict[str, Any]],
    top_k_values: list[int],
) -> list[dict[str, Any]]:
    """Run v1 retrieval for every question and top-k setting."""
    retriever = TfidfRetrieverV1(chunks)
    chunk_index = _build_chunk_index(chunks)
    records: list[dict[str, Any]] = []
    for question in eval_questions:
        question_text = str(question.get("question") or "")
        question_type = str(question.get("question_type") or "")
        expected_keywords = [str(value) for value in question.get("expected_section_keywords", []) or []]
        query_terms = retriever.build_query_terms(
            question_text,
            question_type=question_type,
            expected_section_keywords=expected_keywords,
        )
        for top_k in top_k_values:
            retrieved = [
                result.to_baseline_dict()
                for result in retriever.retrieve(
                    question_text,
                    top_k=top_k,
                    question_type=question_type,
                    expected_section_keywords=expected_keywords,
                )
            ]
            expected = [str(value) for value in question.get("expected_source_arxiv_ids", []) or []]
            hit = evaluate_expected_source_hit(retrieved, expected)
            diagnostics = build_retrieval_diagnostics(
                chunks=chunks,
                question=question,
                retrieved_chunks=retrieved,
                query_terms=query_terms,
            )
            records.append(
                {
                    "question_id": question.get("question_id", ""),
                    "question": question_text,
                    "question_type": question_type,
                    "top_k": top_k,
                    "retriever_name": TfidfRetrieverV1.name,
                    "expected_source_arxiv_ids": expected,
                    "expected_source_hit": hit["expected_source_hit"],
                    "expected_source_hit_rank": hit["expected_source_hit_rank"],
                    "retrieved_chunks": retrieved,
                    "diagnostics": {
                        "query_terms": query_terms,
                        "matched_expected_source_chunks_available": diagnostics[
                            "matched_expected_source_chunks_available"
                        ],
                        "notes": diagnostics["notes"],
                        "likely_reason": diagnostics["likely_reason"],
                        "expected_source_chunk_count": diagnostics["expected_source_chunk_count"],
                        "best_expected_source_term_overlap": diagnostics[
                            "best_expected_source_term_overlap"
                        ],
                        "retrieved_expected_source_available": _retrieved_expected_source_available(
                            retrieved, expected, chunk_index
                        ),
                    },
                }
            )
    return records


def build_retrieval_diagnostics(
    *,
    chunks: list[dict[str, Any]],
    question: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    query_terms: list[str],
) -> dict[str, Any]:
    """Build lightweight expected-source and lexical-overlap diagnostics."""
    expected = {str(value) for value in question.get("expected_source_arxiv_ids", []) or []}
    expected_chunks = [chunk for chunk in chunks if str(chunk.get("arxiv_id") or "") in expected]
    expected_source_available = bool(expected_chunks)
    best_overlap = 0
    best_chunk_id = ""
    for chunk in expected_chunks:
        text = " ".join(
            [
                str(chunk.get("title") or ""),
                str(chunk.get("section_heading") or ""),
                str(chunk.get("text") or ""),
            ]
        )
        overlap = len(set(query_terms) & set(tokenize_v1(text)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk_id = str(chunk.get("chunk_id") or "")

    hit = evaluate_expected_source_hit(retrieved_chunks, list(expected))
    likely_reason = classify_miss_reason(
        question=question,
        hit=bool(hit["expected_source_hit"]),
        expected_source_available=expected_source_available,
        best_expected_source_term_overlap=best_overlap,
        retrieved_chunks=retrieved_chunks,
    )
    notes = "Expected source hit." if hit["expected_source_hit"] else f"Likely miss reason: {likely_reason}."
    if not expected:
        notes = "No expected source arXiv IDs were specified."

    return {
        "matched_expected_source_chunks_available": expected_source_available,
        "expected_source_chunk_count": len(expected_chunks),
        "best_expected_source_term_overlap": {
            "chunk_id": best_chunk_id,
            "matching_query_terms_count": best_overlap,
        },
        "likely_reason": likely_reason,
        "notes": notes,
    }


def classify_miss_reason(
    *,
    question: dict[str, Any],
    hit: bool,
    expected_source_available: bool,
    best_expected_source_term_overlap: int,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Return a coarse likely reason for a retrieval miss."""
    if hit:
        return "unknown"
    if not question.get("expected_source_arxiv_ids"):
        return "expected_source_too_strict"
    if not expected_source_available:
        return "metadata_problem"
    question_text = str(question.get("question") or "").lower()
    if any(marker in question_text for marker in ("first selected", "another selected", "selected paper")):
        return "ambiguous_question"
    if best_expected_source_term_overlap == 0:
        return "lexical_mismatch"
    if not retrieved_chunks:
        return "chunk_missing_information"
    top_score = float(retrieved_chunks[0].get("score") or 0.0)
    if top_score == 0.0:
        return "lexical_mismatch"
    return "retriever_weakness"


def build_eval_summary_v1(
    *,
    records: list[dict[str, Any]],
    top_k_values: list[int],
    rag_v0_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build aggregate retrieval metrics for v1."""
    by_top_k = {top_k: [record for record in records if record["top_k"] == top_k] for top_k in top_k_values}
    hit_rate_by_top_k: dict[str, float] = {}
    mean_rank_by_top_k: dict[str, float | None] = {}
    miss_questions_by_top_k: dict[str, list[str]] = {}
    per_question_type_hit_rate: dict[str, dict[str, float]] = defaultdict(dict)

    for top_k, top_records in by_top_k.items():
        expected_records = [record for record in top_records if record.get("expected_source_arxiv_ids")]
        hits = [record for record in expected_records if record.get("expected_source_hit")]
        hit_rate_by_top_k[str(top_k)] = len(hits) / len(expected_records) if expected_records else 0.0
        ranks = [record["expected_source_hit_rank"] for record in hits if record.get("expected_source_hit_rank")]
        mean_rank_by_top_k[str(top_k)] = sum(ranks) / len(ranks) if ranks else None
        miss_questions_by_top_k[str(top_k)] = [
            str(record["question_id"])
            for record in expected_records
            if not record.get("expected_source_hit")
        ]

        question_types = sorted({str(record.get("question_type") or "unknown") for record in top_records})
        for question_type in question_types:
            typed = [
                record
                for record in expected_records
                if str(record.get("question_type") or "unknown") == question_type
            ]
            typed_hits = [record for record in typed if record.get("expected_source_hit")]
            per_question_type_hit_rate[question_type][str(top_k)] = len(typed_hits) / len(typed) if typed else 0.0

    best_top_k = max(top_k_values, key=lambda value: hit_rate_by_top_k[str(value)])
    comparison = build_v0_comparison(
        rag_v0_summary=rag_v0_summary,
        v1_hit_rate=hit_rate_by_top_k[str(best_top_k)],
        best_top_k=best_top_k,
    )
    recommendation = (
        "Use retriever v1 reports to inspect misses before changing generation. "
        f"Best observed expected-source hit rate is {hit_rate_by_top_k[str(best_top_k)]:.3f} at top_k={best_top_k}."
    )
    return {
        "retriever_eval_summary_version": "retriever_eval_summary_v1",
        "questions_total": len({record["question_id"] for record in records}),
        "top_k_values": top_k_values,
        "hit_rate_by_top_k": hit_rate_by_top_k,
        "mean_expected_source_hit_rank_by_top_k": mean_rank_by_top_k,
        "miss_questions_by_top_k": miss_questions_by_top_k,
        "per_question_type_hit_rate": dict(per_question_type_hit_rate),
        "comparison_to_rag_v0_if_available": comparison,
        "recommendation": recommendation,
        "manual_review": {
            "required": True,
            "status": "pending",
        },
    }


def build_v0_comparison(
    *,
    rag_v0_summary: dict[str, Any] | None,
    v1_hit_rate: float,
    best_top_k: int,
) -> dict[str, Any]:
    if not rag_v0_summary:
        return {
            "available": False,
            "notes": "No rag_eval_summary.json was available for comparison.",
        }
    retrieval = rag_v0_summary.get("retrieval", {})
    v0_hit_rate = retrieval.get("expected_source_hit_rate")
    if not isinstance(v0_hit_rate, int | float):
        return {
            "available": False,
            "notes": "rag_eval_summary.json did not include expected_source_hit_rate.",
        }
    return {
        "available": True,
        "v0_retriever_name": "tfidf_local_v0",
        "v0_top_k": retrieval.get("top_k"),
        "v0_expected_source_hit_rate": v0_hit_rate,
        "v1_retriever_name": TfidfRetrieverV1.name,
        "v1_best_top_k": best_top_k,
        "v1_best_expected_source_hit_rate": v1_hit_rate,
        "delta": v1_hit_rate - float(v0_hit_rate),
    }


def build_error_analysis_markdown_v1(records: list[dict[str, Any]], top_k_values: list[int]) -> str:
    """Build retrieval error-analysis Markdown."""
    max_top_k = max(top_k_values)
    final_records = [record for record in records if record["top_k"] == max_top_k]
    misses = [record for record in final_records if record.get("expected_source_arxiv_ids") and not record["expected_source_hit"]]
    ambiguous = [
        record
        for record in misses
        if record.get("diagnostics", {}).get("likely_reason") == "ambiguous_question"
    ]
    expected_unavailable = [
        record
        for record in final_records
        if record.get("expected_source_arxiv_ids")
        and not record.get("diagnostics", {}).get("matched_expected_source_chunks_available")
    ]
    low_overlap = [
        record
        for record in misses
        if record.get("diagnostics", {}).get("likely_reason") == "lexical_mismatch"
    ]
    chunking = [
        record
        for record in misses
        if record.get("diagnostics", {}).get("likely_reason") == "chunk_missing_information"
    ]
    query_wording = [
        record
        for record in misses
        if record.get("diagnostics", {}).get("likely_reason") in {"ambiguous_question", "lexical_mismatch"}
    ]

    lines = [
        "# KamiKnows Retriever Baseline v1 Error Analysis",
        "",
        "This report is retrieval-only. It does not validate generated answers or scientific correctness.",
        "",
        f"Detailed miss analysis below uses top_k={max_top_k}.",
        "",
        "## Retrieval Miss Cases",
        "",
    ]
    lines.extend(_record_bullets(misses, empty="No retrieval misses at the largest top_k."))
    lines.extend(["", "## Ambiguous Questions", ""])
    lines.extend(_record_bullets(ambiguous, empty="No ambiguous-question misses were flagged."))
    lines.extend(["", "## Expected Source Unavailable Cases", ""])
    lines.extend(_record_bullets(expected_unavailable, empty="All expected sources had at least one chunk."))
    lines.extend(["", "## Low Lexical Overlap Cases", ""])
    lines.extend(_record_bullets(low_overlap, empty="No low-overlap misses were flagged."))
    lines.extend(["", "## Chunking Issues", ""])
    lines.extend(_record_bullets(chunking, empty="No chunk-missing-information misses were flagged."))
    lines.extend(["", "## Query Wording Issues", ""])
    lines.extend(_record_bullets(query_wording, empty="No query-wording misses were flagged."))
    lines.extend(
        [
            "",
            "## Recommended Fixes",
            "",
            "- Inspect miss questions before changing generation.",
            "- Clarify evaluation questions that rely on ordinal phrasing such as first or another selected paper.",
            "- Add targeted query expansions only when they are deterministic and documented.",
            "- Review chunk boundaries for expected papers whose key terms are split across chunks.",
            "- Keep citation validation separate from scientific correctness review.",
            "",
        ]
    )
    return "\n".join(lines)


def build_debug_report_markdown(records: list[dict[str, Any]], top_k_values: list[int]) -> str:
    """Build detailed retrieval miss debug report."""
    max_top_k = max(top_k_values)
    misses = [
        record
        for record in records
        if record["top_k"] == max_top_k
        and record.get("expected_source_arxiv_ids")
        and not record["expected_source_hit"]
    ]
    lines = [
        "# KamiKnows Retrieval Debug Report v1",
        "",
        f"Miss details use top_k={max_top_k}, the largest requested retrieval depth.",
        "",
    ]
    if not misses:
        lines.extend(["No retrieval misses at the largest top-k.", ""])
        return "\n".join(lines)

    for record in misses:
        diagnostics = record.get("diagnostics", {})
        lines.extend(
            [
                f"## {record['question_id']}",
                "",
                f"Question: {record['question']}",
                "",
                "Expected source: " + ", ".join(record.get("expected_source_arxiv_ids", []) or ["<none>"]),
                "",
                "Top retrieved sources:",
            ]
        )
        source_seen: set[str] = set()
        for chunk in record.get("retrieved_chunks", []):
            source = str(chunk.get("arxiv_id") or "<missing>")
            if source in source_seen:
                continue
            source_seen.add(source)
            lines.append(f"- `{source}`")
        lines.extend(["", "Best retrieved chunks:"])
        for chunk in record.get("retrieved_chunks", [])[:5]:
            lines.append(
                "- rank {rank}: `{chunk_id}` / `{arxiv_id}` / score {score:.4f} / {section}".format(
                    rank=chunk.get("rank"),
                    chunk_id=chunk.get("chunk_id", ""),
                    arxiv_id=chunk.get("arxiv_id", ""),
                    score=float(chunk.get("score") or 0.0),
                    section=chunk.get("section_heading", ""),
                )
            )
        overlap = diagnostics.get("best_expected_source_term_overlap", {})
        lines.extend(
            [
                "",
                "Expected paper had chunks containing key terms: "
                + ("yes" if overlap.get("matching_query_terms_count", 0) > 0 else "no"),
                f"Best expected-source overlap chunk: `{overlap.get('chunk_id', '')}`",
                f"Likely reason: `{diagnostics.get('likely_reason', 'unknown')}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_run_config(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    top_k_values: list[int],
    rag_v0_summary_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_config_version": "retrieval_run_config_v1",
        "rag_ready_dir": str(rag_ready_dir),
        "output_dir": str(output_dir),
        "top_k_values": top_k_values,
        "rag_v0_summary_path": str(rag_v0_summary_path) if rag_v0_summary_path else None,
        "retriever": {
            "name": TfidfRetrieverV1.name,
            "version": TfidfRetrieverV1.version,
            "features": [
                "deterministic tokenization",
                "stopword filtering",
                "title and section heading weighting",
                "question-type and expected-keyword query expansion",
                "expected-source diagnostics",
            ],
        },
        "scope": "retrieval baseline v1; no generation, no fine-tuning, no vector DB, no network",
    }


def build_retrieval_manifest(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    dataset_counts: dict[str, int],
    top_k_values: list[int],
) -> dict[str, Any]:
    files = {
        key: str(output_dir / filename)
        for key, filename in REQUIRED_OUTPUT_FILES.items()
        if key != "retrieval_manifest"
    }
    missing_outputs = [path for path in files.values() if not Path(path).exists()]
    validation_status = "PASS" if not missing_outputs else "FAIL"
    return {
        "retrieval_run_name": output_dir.name or "rag_v1_fastcalosim_retrieval",
        "source_rag_ready_dir": str(rag_ready_dir),
        "output_dir": str(output_dir),
        "created_at": utc_now_iso(),
        "dataset_counts": dataset_counts,
        "retriever": {
            "name": TfidfRetrieverV1.name,
            "version": TfidfRetrieverV1.version,
            "top_k_values": top_k_values,
        },
        "files": files,
        "validation": {
            "status": validation_status,
            "missing_outputs": missing_outputs,
            "warnings": ["Manual review remains required."],
        },
        "scope": "retriever baseline v1; no generation, no scientific correctness claim",
    }


def run_retriever_baseline_v1(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    top_k_values: list[int],
    rag_v0_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Run retrieval baseline v1 and write all requested artifacts."""
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(value < 1 for value in top_k_values):
        raise ValueError("all top_k_values must be >= 1")
    top_k_values = sorted(dict.fromkeys(top_k_values))
    dataset = load_rag_ready_dataset(rag_ready_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config(
        rag_ready_dir=rag_ready_dir,
        output_dir=output_dir,
        top_k_values=top_k_values,
        rag_v0_summary_path=rag_v0_summary_path,
    )
    write_json(output_dir / "retrieval_run_config.json", run_config)

    records = build_baseline_records(
        chunks=dataset["chunks"],
        eval_questions=dataset["eval_questions"],
        top_k_values=top_k_values,
    )
    write_jsonl(output_dir / "retriever_baseline_v1.jsonl", records)

    rag_v0_summary = None
    if rag_v0_summary_path and rag_v0_summary_path.exists():
        rag_v0_summary = read_json(rag_v0_summary_path)
    summary = build_eval_summary_v1(
        records=records,
        top_k_values=top_k_values,
        rag_v0_summary=rag_v0_summary,
    )
    write_json(output_dir / "retriever_eval_summary_v1.json", summary)

    (output_dir / "retriever_error_analysis_v1.md").write_text(
        build_error_analysis_markdown_v1(records, top_k_values),
        encoding="utf-8",
    )
    (output_dir / "retrieval_debug_report.md").write_text(
        build_debug_report_markdown(records, top_k_values),
        encoding="utf-8",
    )

    manifest = build_retrieval_manifest(
        rag_ready_dir=rag_ready_dir,
        output_dir=output_dir,
        dataset_counts=dataset["counts"],
        top_k_values=top_k_values,
    )
    write_json(output_dir / "retrieval_manifest.json", manifest)
    return {
        "dataset": dataset,
        "records": records,
        "eval_summary": summary,
        "manifest": manifest,
    }


def default_v0_summary_path(rag_ready_dir: Path) -> Path:
    """Return the default v0 summary path for the current output layout."""
    return Path(rag_ready_dir).parent / "rag_v0_fastcalosim" / "rag_eval_summary.json"


def _record_bullets(records: list[dict[str, Any]], *, empty: str) -> list[str]:
    if not records:
        return [empty]
    return [
        "- `{question_id}`: {question} Expected: {expected}. Reason: `{reason}`.".format(
            question_id=record.get("question_id", ""),
            question=record.get("question", ""),
            expected=", ".join(record.get("expected_source_arxiv_ids", []) or ["<none>"]),
            reason=record.get("diagnostics", {}).get("likely_reason", "unknown"),
        )
        for record in records
    ]


def _build_chunk_index(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks}


def _retrieved_expected_source_available(
    retrieved: list[dict[str, Any]],
    expected_arxiv_ids: list[str],
    chunk_index: dict[str, dict[str, Any]],
) -> bool:
    expected = {str(value) for value in expected_arxiv_ids if value}
    for retrieved_chunk in retrieved:
        chunk_id = str(retrieved_chunk.get("chunk_id") or "")
        indexed = chunk_index.get(chunk_id)
        if indexed and str(indexed.get("arxiv_id") or "") in expected:
            return True
    return False
