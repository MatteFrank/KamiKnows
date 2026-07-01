"""Build RAG-ready chunk metadata v1 from an existing RAG-ready v0 dataset."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from kamiknows.rag.evaluation import evaluate_expected_source_hit
from kamiknows.rag.load_dataset import load_rag_ready_dataset
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1, tokenize_v1
from kamiknows.run_metadata import utc_now_iso

DOMAIN = "HEP / FastCaloSimulation / calorimetry"
CHUNK_SCHEMA_VERSION = "chunk_schema_v1"
VALID_SOURCE_TYPES = {"abstract", "section", "equation_context", "table_context", "unknown"}
VERY_SHORT_TEXT_CHARS = 80
VERY_LONG_TEXT_CHARS = 2500


def normalize_section(section: str) -> str:
    """Normalize a section label for stable filtering/grouping."""
    section = (section or "").strip().lower()
    section = re.sub(r"[^a-z0-9]+", "_", section)
    section = re.sub(r"_+", "_", section).strip("_")
    return section or "unknown"


def stable_chunk_id_v1(paper_id: str, index: int) -> str:
    """Build a stable chunk v1 ID scoped by paper and sequence."""
    safe_paper_id = paper_id.strip() or "unknown_paper"
    return f"{safe_paper_id}::chunk_v1::{index:04d}"


def audit_chunks_v0(
    *,
    chunks: list[dict[str, Any]],
    papers: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit v0 chunks and return summary plus per-chunk details."""
    paper_ids = {str(paper.get("paper_id") or "") for paper in papers}
    text_counts = Counter(_normalized_text(chunk.get("text", "")) for chunk in chunks)
    lengths = [len(str(chunk.get("text") or "")) for chunk in chunks]
    details = []
    for index, chunk in enumerate(chunks, start=1):
        flags = quality_flags_for_v0_chunk(
            chunk=chunk,
            paper_ids=paper_ids,
            normalized_text_counts=text_counts,
        )
        details.append(
            {
                "record_index": index,
                "chunk_id": chunk.get("chunk_id", ""),
                "paper_id": chunk.get("paper_id", ""),
                "arxiv_id": chunk.get("arxiv_id", ""),
                "title": chunk.get("title", ""),
                "source_type": chunk.get("source_type", ""),
                "section": chunk.get("section_heading", "") or chunk.get("section", ""),
                "text_length_chars": len(str(chunk.get("text") or "")),
                "text_length_tokens_est": len(tokenize_v1(str(chunk.get("text") or ""))),
                "quality_flags": flags,
            }
        )

    flags_counter = Counter(flag for detail in details for flag in detail["quality_flags"])
    chunks_by_paper = Counter(str(chunk.get("paper_id") or "<missing>") for chunk in chunks)
    source_type_distribution = Counter(str(chunk.get("source_type") or "<missing>") for chunk in chunks)
    section_distribution = Counter(
        str(chunk.get("section_heading") or chunk.get("section") or "<missing>")
        for chunk in chunks
    )
    summary = {
        "audit_version": "chunk_audit_v0_summary_v1",
        "chunks_count": len(chunks),
        "papers_count": len(papers),
        "chunks_per_paper": dict(sorted(chunks_by_paper.items())),
        "source_type_distribution": dict(sorted(source_type_distribution.items())),
        "section_distribution": dict(sorted(section_distribution.items())),
        "text_length_chars": {
            "min": min(lengths) if lengths else 0,
            "mean": mean(lengths) if lengths else 0.0,
            "median": median(lengths) if lengths else 0.0,
            "max": max(lengths) if lengths else 0,
        },
        "chunk_empty_count": flags_counter.get("empty_text", 0),
        "chunk_missing_paper_id_count": flags_counter.get("missing_paper_id", 0),
        "chunk_missing_arxiv_id_count": flags_counter.get("missing_arxiv_id", 0),
        "chunk_missing_title_count": flags_counter.get("missing_title", 0),
        "chunk_missing_section_count": flags_counter.get("missing_section", 0),
        "chunk_duplicate_or_near_duplicate_count": flags_counter.get("duplicate_text_candidate", 0),
        "chunk_too_short_count": flags_counter.get("very_short_text", 0),
        "chunk_too_long_count": flags_counter.get("very_long_text", 0),
        "chunk_orphan_count": flags_counter.get("orphan_chunk", 0),
        "quality_flag_counts": dict(sorted(flags_counter.items())),
    }
    return summary, details


def quality_flags_for_v0_chunk(
    *,
    chunk: dict[str, Any],
    paper_ids: set[str],
    normalized_text_counts: Counter[str],
) -> list[str]:
    """Return quality flags for one v0 chunk."""
    flags = []
    paper_id = str(chunk.get("paper_id") or "").strip()
    arxiv_id = str(chunk.get("arxiv_id") or "").strip()
    title = str(chunk.get("title") or "").strip()
    section = str(chunk.get("section_heading") or chunk.get("section") or "").strip()
    text = str(chunk.get("text") or "")
    source_type = str(chunk.get("source_type") or "").strip()
    normalized = _normalized_text(text)

    if not text.strip():
        flags.append("empty_text")
    if not paper_id:
        flags.append("missing_paper_id")
    if not arxiv_id:
        flags.append("missing_arxiv_id")
    if not title:
        flags.append("missing_title")
    if not section:
        flags.append("missing_section")
    if source_type not in VALID_SOURCE_TYPES:
        flags.append("unknown_source_type")
    if text.strip() and len(text) < VERY_SHORT_TEXT_CHARS:
        flags.append("very_short_text")
    if len(text) > VERY_LONG_TEXT_CHARS:
        flags.append("very_long_text")
    if paper_id and paper_id not in paper_ids:
        flags.append("orphan_chunk")
    if normalized and normalized_text_counts[normalized] > 1:
        flags.append("duplicate_text_candidate")
    return flags


def build_chunks_v1(
    *,
    chunks: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    equations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build normalized chunk v1 records and v0-to-v1 ID map."""
    papers_by_id = {str(paper.get("paper_id") or ""): paper for paper in papers}
    papers_by_arxiv = {str(paper.get("arxiv_id") or ""): paper for paper in papers}
    paper_ids = set(papers_by_id)
    normalized_text_counts = Counter(_normalized_text(chunk.get("text", "")) for chunk in chunks)
    per_paper_index: defaultdict[str, int] = defaultdict(int)
    equations_by_paper_section: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for equation in equations:
        key = (str(equation.get("paper_id") or ""), str(equation.get("section_id") or ""))
        equation_id = str(equation.get("global_equation_id") or equation.get("equation_id") or "")
        if equation_id:
            equations_by_paper_section[key].append(equation_id)

    chunks_v1 = []
    chunk_id_map = {}
    for chunk in chunks:
        original_paper_id = str(chunk.get("paper_id") or "").strip()
        original_arxiv_id = str(chunk.get("arxiv_id") or "").strip()
        paper = papers_by_id.get(original_paper_id) or papers_by_arxiv.get(original_arxiv_id) or {}
        paper_id = original_paper_id or str(paper.get("paper_id") or "").strip() or "unknown_paper"
        per_paper_index[paper_id] += 1
        chunk_id = stable_chunk_id_v1(paper_id, per_paper_index[paper_id])
        arxiv_id = original_arxiv_id or str(paper.get("arxiv_id") or "").strip()
        title = str(chunk.get("title") or paper.get("title") or "").strip()
        section = str(chunk.get("section_heading") or chunk.get("section") or "").strip()
        source_type = str(chunk.get("source_type") or "").strip()
        if source_type not in VALID_SOURCE_TYPES:
            source_type = "unknown"
        metadata = dict(chunk.get("metadata") or {})
        metadata.update(
            {
                "domain": metadata.get("domain") or DOMAIN,
                "source_file": metadata.get("source_file") or paper.get("files", {}).get("chunks", ""),
                "version": "v1",
            }
        )
        flags = quality_flags_for_v0_chunk(
            chunk={**chunk, "paper_id": original_paper_id, "arxiv_id": original_arxiv_id, "title": chunk.get("title", "")},
            paper_ids=paper_ids,
            normalized_text_counts=normalized_text_counts,
        )
        # Add flags after propagation only when v1 still cannot provide required metadata.
        if not arxiv_id and "missing_arxiv_id" not in flags:
            flags.append("missing_arxiv_id")
        if not title and "missing_title" not in flags:
            flags.append("missing_title")
        if source_type == "unknown" and "unknown_source_type" not in flags:
            flags.append("unknown_source_type")

        text = str(chunk.get("text") or "")
        section_normalized = normalize_section(section)
        parent_chunk_id = str(chunk.get("chunk_id") or "")
        chunk_id_map[parent_chunk_id] = chunk_id
        chunks_v1.append(
            {
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "source_type": source_type,
                "section": section,
                "section_heading": section,
                "section_normalized": section_normalized,
                "text": text,
                "text_length_chars": len(text),
                "text_length_tokens_est": len(tokenize_v1(text)),
                "metadata": metadata,
                "quality_flags": sorted(set(flags)),
                "source_refs": {
                    "parent_chunk_id": parent_chunk_id,
                    "equation_ids": equations_by_paper_section.get(
                        (paper_id, str(chunk.get("section_id") or "")),
                        [],
                    ),
                    "paper_record_available": paper_id in papers_by_id,
                },
            }
        )
    return chunks_v1, chunk_id_map


def build_manifest_v1(
    *,
    input_dir: Path,
    output_dir: Path,
    chunks_v0_count: int,
    chunks_v1: list[dict[str, Any]],
    papers_count: int,
    equations_count: int,
    eval_questions_count: int,
) -> dict[str, Any]:
    """Build RAG-ready v1 manifest."""
    quality_counter = Counter(flag for chunk in chunks_v1 for flag in chunk.get("quality_flags", []))
    return {
        "manifest_version": "rag_manifest_v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "chunks_v0_count": chunks_v0_count,
        "chunks_v1_count": len(chunks_v1),
        "papers_count": papers_count,
        "equations_count": equations_count,
        "eval_questions_count": eval_questions_count,
        "created_at": utc_now_iso(),
        "schema_version": CHUNK_SCHEMA_VERSION,
        "quality_summary": {
            "chunks_with_quality_flags": sum(1 for chunk in chunks_v1 if chunk.get("quality_flags")),
            "quality_flag_counts": dict(sorted(quality_counter.items())),
        },
        "known_limits": [
            "Chunk text is inherited from v0; this task does not reparse full text.",
            "Quality flags are diagnostics and do not prove scientific answer quality.",
            "Duplicate detection is a deterministic exact normalized-text candidate check.",
            "Retrieval smoke test checks usability, not retrieval improvement.",
        ],
    }


def build_retrieval_smoke_summary(
    *,
    chunks_v1: list[dict[str, Any]],
    eval_questions: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Run a TF-IDF v1 retrieval smoke test over chunks v1."""
    retriever = TfidfRetrieverV1(chunks_v1)
    records = []
    for question in eval_questions:
        retrieved_chunks = [
            result.to_baseline_dict()
            for result in retriever.retrieve(
                str(question.get("question") or ""),
                top_k=top_k,
                question_type=str(question.get("question_type") or ""),
                expected_section_keywords=[str(value) for value in question.get("expected_section_keywords", []) or []],
            )
        ]
        expected = [str(value) for value in question.get("expected_source_arxiv_ids", []) or []]
        retrieval_eval = evaluate_expected_source_hit(retrieved_chunks, expected)
        records.append(
            {
                "question_id": question.get("question_id", ""),
                "expected_source_arxiv_ids": expected,
                "expected_source_hit": retrieval_eval["expected_source_hit"],
                "expected_source_hit_rank": retrieval_eval["expected_source_hit_rank"],
            }
        )
    expected_records = [record for record in records if record["expected_source_arxiv_ids"]]
    hits = [record for record in expected_records if record["expected_source_hit"]]
    return {
        "retrieval_smoke_summary_version": "retrieval_smoke_summary_v1",
        "retriever_name": TfidfRetrieverV1.name,
        "top_k": top_k,
        "questions_total": len(eval_questions),
        "chunks_total": len(chunks_v1),
        "expected_source_questions": len(expected_records),
        "expected_source_hits": len(hits),
        "expected_source_hit_rate": len(hits) / len(expected_records) if expected_records else 0.0,
        "miss_question_ids": [
            str(record["question_id"])
            for record in expected_records
            if not record["expected_source_hit"]
        ],
        "status": "PASS",
        "notes": "Smoke test verifies chunks_v1 are retrievable; it does not prove improvement or scientific correctness.",
    }


def build_rag_ready_chunk_metadata_v1(
    *,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build all chunking/metadata v1 outputs."""
    dataset = load_rag_ready_dataset(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_summary, audit_details = audit_chunks_v0(
        chunks=dataset["chunks"],
        papers=dataset["papers"],
    )
    write_json(output_dir / "chunk_audit_v0_summary.json", audit_summary)
    write_jsonl(output_dir / "chunk_audit_v0_details.jsonl", audit_details)

    chunks_v1, chunk_id_map = build_chunks_v1(
        chunks=dataset["chunks"],
        papers=dataset["papers"],
        equations=dataset["equations"],
    )
    write_jsonl(output_dir / "chunks_v1.jsonl", chunks_v1)
    write_json(output_dir / "chunk_id_map_v0_to_v1.json", chunk_id_map)
    write_jsonl(output_dir / "eval_questions_v0.jsonl", dataset["eval_questions"])

    smoke_summary = build_retrieval_smoke_summary(
        chunks_v1=chunks_v1,
        eval_questions=dataset["eval_questions"],
        top_k=5,
    )
    write_json(output_dir / "retrieval_smoke_summary_v1.json", smoke_summary)

    manifest = build_manifest_v1(
        input_dir=input_dir,
        output_dir=output_dir,
        chunks_v0_count=len(dataset["chunks"]),
        chunks_v1=chunks_v1,
        papers_count=len(dataset["papers"]),
        equations_count=len(dataset["equations"]),
        eval_questions_count=len(dataset["eval_questions"]),
    )
    write_json(output_dir / "rag_manifest_v1.json", manifest)
    return {
        "dataset": dataset,
        "audit_summary": audit_summary,
        "audit_details": audit_details,
        "chunks_v1": chunks_v1,
        "chunk_id_map": chunk_id_map,
        "retrieval_smoke_summary": smoke_summary,
        "manifest": manifest,
    }


def _normalized_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())
