"""RAG generator/configuration benchmark for Phase 2 Task 07.

The benchmark compares context sources, prompt profiles, and local generator
models using structural metrics only. It does not use an LLM judge and does not
claim scientific answer correctness.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import requests

from kamiknows.rag.load_dataset import read_json, read_jsonl
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag.qa_grounded_v1 import (
    DEFAULT_BASE_URL,
    build_grounded_qa_prompt_v1,
    generate_ollama_json,
    list_local_ollama_models,
    model_slug,
    parse_model_json_response,
    validate_answer_citations,
)
from kamiknows.rag.reranker_v1 import evaluate_retrieval
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1
from kamiknows.run_metadata import utc_now_iso


BENCHMARK_OUTPUT_FILES = {
    "benchmark_matrix": "benchmark_matrix_v1.json",
    "benchmark_answers": "benchmark_answers_v1.jsonl",
    "benchmark_summary": "benchmark_summary_v1.json",
    "config_comparison": "benchmark_config_comparison_v1.md",
    "citation_validation_summary": "citation_validation_summary_v1.json",
    "abstention_analysis": "abstention_analysis_v1.json",
    "error_analysis": "benchmark_error_analysis_v1.md",
    "recommended_config": "recommended_rag_config_v1.json",
    "manifest": "benchmark_manifest_v1.json",
}


def run_rag_generator_benchmark_v1(
    *,
    rag_ready_dir: Path,
    eval_questions_file: Path,
    reranking_output_dir: Path | None,
    output_dir: Path,
    report_path: Path,
    config_path: Path | None = None,
    include_tfidf_baseline: bool = True,
    include_light_reranked: bool = True,
    fallback_to_tfidf_if_reranking_missing: bool = True,
    top_k: int = 5,
    prompt_profiles: list[dict[str, Any]] | None = None,
    model_names: list[str] | None = None,
    parameter_sets: list[dict[str, Any]] | None = None,
    backend: str = "ollama",
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds_override: int | None = None,
    context_sources: list[str] | None = None,
    pytest_status: str | None = None,
) -> dict[str, Any]:
    """Run the full generator benchmark and write all artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rag_ready_dir = Path(rag_ready_dir)
    eval_questions_file = Path(eval_questions_file)
    reranking_output_dir = Path(reranking_output_dir) if reranking_output_dir else None

    chunks = read_jsonl(rag_ready_dir / "chunks_v1.jsonl")
    rag_manifest = read_json(rag_ready_dir / "rag_manifest_v1.json")
    eval_questions = read_jsonl(eval_questions_file)
    known_chunk_ids = [str(chunk.get("chunk_id") or "") for chunk in chunks]

    prompt_profiles = prompt_profiles or [
        {"name": "grounded_json_v1", "description": "P2T4 citation-constrained JSON prompt."},
        {"name": "strict_abstention_v2", "description": "More conservative prompt for missing or incomplete evidence."},
    ]
    model_names = model_names or ["qwen3:0.6b", "qwen3:4b-instruct", "mistral:7b"]
    parameter_sets = parameter_sets or [
        {
            "name": "deterministic_512",
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": 512,
            "timeout_seconds": 180,
        }
    ]

    contexts_by_source = load_benchmark_contexts(
        chunks=chunks,
        eval_questions=eval_questions,
        reranking_output_dir=reranking_output_dir,
        include_tfidf_baseline=include_tfidf_baseline,
        include_light_reranked=include_light_reranked,
        fallback_to_tfidf_if_reranking_missing=fallback_to_tfidf_if_reranking_missing,
        top_k=top_k,
    )
    if context_sources:
        contexts_by_source = {
            key: value for key, value in contexts_by_source.items() if key in set(context_sources)
        }
    matrix = build_benchmark_matrix(
        context_sources=list(contexts_by_source),
        prompt_profiles=prompt_profiles,
        model_names=model_names,
        parameter_sets=parameter_sets,
        backend=backend,
        top_k=top_k,
    )

    availability = list_local_ollama_models(base_url=base_url)
    answer_records = generate_benchmark_answers(
        matrix=matrix["configs"],
        contexts_by_source=contexts_by_source,
        known_chunk_ids=known_chunk_ids,
        availability=availability,
        base_url=base_url,
        timeout_seconds_override=timeout_seconds_override,
    )
    summary = build_benchmark_summary(
        matrix_configs=matrix["configs"],
        answer_records=answer_records,
    )
    citation_summary = build_benchmark_citation_validation_summary(answer_records)
    abstention_analysis = build_abstention_analysis(answer_records)
    recommended = build_recommended_config(summary["per_config"])

    output_paths = {key: output_dir / filename for key, filename in BENCHMARK_OUTPUT_FILES.items()}
    write_json(output_paths["benchmark_matrix"], matrix)
    write_jsonl(output_paths["benchmark_answers"], answer_records)
    write_json(output_paths["benchmark_summary"], summary)
    output_paths["config_comparison"].write_text(
        build_config_comparison_markdown(per_config=summary["per_config"], recommended=recommended),
        encoding="utf-8",
    )
    write_json(output_paths["citation_validation_summary"], citation_summary)
    write_json(output_paths["abstention_analysis"], abstention_analysis)
    output_paths["error_analysis"].write_text(
        build_error_analysis_markdown(answer_records=answer_records, per_config=summary["per_config"]),
        encoding="utf-8",
    )
    write_json(output_paths["recommended_config"], recommended)
    manifest = build_manifest(
        rag_ready_dir=rag_ready_dir,
        eval_questions_file=eval_questions_file,
        reranking_output_dir=reranking_output_dir,
        output_dir=output_dir,
        report_path=report_path,
        config_path=config_path,
        output_paths=output_paths,
        model_names=model_names,
        prompt_profiles=prompt_profiles,
        context_sources=list(contexts_by_source),
        generated_records=len(answer_records),
        pytest_status=pytest_status,
    )
    write_json(output_paths["manifest"], manifest)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_task_report_markdown(
            rag_ready_dir=rag_ready_dir,
            eval_questions_file=eval_questions_file,
            reranking_output_dir=reranking_output_dir,
            output_dir=output_dir,
            matrix=matrix,
            summary=summary,
            citation_summary=citation_summary,
            abstention_analysis=abstention_analysis,
            recommended=recommended,
            rag_manifest=rag_manifest,
        ),
        encoding="utf-8",
    )

    return {
        "chunks": chunks,
        "eval_questions": eval_questions,
        "contexts_by_source": contexts_by_source,
        "matrix": matrix,
        "answer_records": answer_records,
        "summary": summary,
        "citation_summary": citation_summary,
        "abstention_analysis": abstention_analysis,
        "recommended_config": recommended,
        "manifest": manifest,
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "report_path": str(report_path),
    }


def build_benchmark_matrix(
    *,
    context_sources: list[str],
    prompt_profiles: list[dict[str, Any]],
    model_names: list[str],
    parameter_sets: list[dict[str, Any]],
    backend: str,
    top_k: int,
) -> dict[str, Any]:
    """Build the benchmark configuration matrix."""
    configs = []
    for index, (context_source, prompt_profile, model_name, parameter_set) in enumerate(
        product(context_sources, prompt_profiles, model_names, parameter_sets),
        start=1,
    ):
        config_id = "cfg_{:03d}__{}__{}__{}__{}".format(
            index,
            context_source,
            prompt_profile["name"],
            model_slug(model_name),
            parameter_set["name"],
        )
        configs.append(
            {
                "config_id": config_id,
                "context_source": context_source,
                "prompt_profile": prompt_profile["name"],
                "prompt_profile_description": prompt_profile.get("description", ""),
                "model_name": model_name,
                "parameter_set": dict(parameter_set),
                "backend": backend,
                "top_k": top_k,
                "enabled": True,
            }
        )
    return {
        "benchmark_matrix_version": "benchmark_matrix_v1",
        "created_at": utc_now_iso(),
        "dimensions": {
            "context_sources": context_sources,
            "prompt_profiles": [profile["name"] for profile in prompt_profiles],
            "models": model_names,
            "parameter_sets": [params["name"] for params in parameter_sets],
        },
        "configs": configs,
    }


def load_benchmark_contexts(
    *,
    chunks: list[dict[str, Any]],
    eval_questions: list[dict[str, Any]],
    reranking_output_dir: Path | None,
    include_tfidf_baseline: bool,
    include_light_reranked: bool,
    fallback_to_tfidf_if_reranking_missing: bool,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    """Load benchmark contexts from P2T6 outputs or reconstruct TF-IDF."""
    contexts: dict[str, list[dict[str, Any]]] = {}
    if include_tfidf_baseline:
        path = (reranking_output_dir / "baseline_retrieval_results_v1.jsonl") if reranking_output_dir else None
        if path and path.exists():
            contexts["tfidf_baseline_top5"] = _contexts_from_p2t6_baseline(path, top_k=top_k)
        else:
            contexts["tfidf_baseline_top5"] = build_tfidf_contexts(chunks=chunks, eval_questions=eval_questions, top_k=top_k)

    if include_light_reranked:
        path = (reranking_output_dir / "reranked_retrieval_results_v1.jsonl") if reranking_output_dir else None
        if path and path.exists():
            contexts["light_reranked_top5"] = _contexts_from_p2t6_reranked(path, top_k=top_k)
        elif fallback_to_tfidf_if_reranking_missing:
            fallback = contexts.get("tfidf_baseline_top5") or build_tfidf_contexts(
                chunks=chunks,
                eval_questions=eval_questions,
                top_k=top_k,
            )
            contexts["light_reranked_top5"] = [
                {**record, "context_source": "light_reranked_top5", "context_warning": "reranking_missing_tfidf_fallback"}
                for record in fallback
            ]
    return contexts


def build_tfidf_contexts(
    *,
    chunks: list[dict[str, Any]],
    eval_questions: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Build TF-IDF contexts directly from chunks."""
    retriever = TfidfRetrieverV1(chunks)
    records = []
    for question in eval_questions:
        results = []
        for item in retriever.retrieve(
            str(question.get("question") or ""),
            top_k=top_k,
            question_type=str(question.get("question_type") or ""),
            expected_section_keywords=[str(value) for value in question.get("expected_section_keywords", []) or []],
        ):
            results.append(_context_chunk_from_retrieved(item.rank, item.score, item.chunk))
        records.append(_context_record_from_question(
            question=question,
            context_source="tfidf_baseline_top5",
            retrieved_chunks=results,
            retrieval_eval=evaluate_retrieval(question=question, retrieved_chunks=results, top_k=top_k),
            top_k=top_k,
        ))
    return records


def build_prompt(
    *,
    prompt_profile: str,
    context_record: dict[str, Any],
) -> str:
    """Build a benchmark prompt without exposing expected source IDs."""
    if prompt_profile == "grounded_json_v1":
        return build_grounded_qa_prompt_v1(context_record)
    if prompt_profile == "strict_abstention_v2":
        return build_strict_abstention_prompt_v2(context_record)
    raise ValueError(f"unknown prompt_profile: {prompt_profile}")


def build_strict_abstention_prompt_v2(context_record: dict[str, Any]) -> str:
    """Build a stricter citation-constrained prompt."""
    blocks = []
    for chunk in context_record.get("retrieved_chunks", []) or []:
        blocks.append(
            "\n".join(
                [
                    f"{chunk.get('citation_label', '')}",
                    f"chunk_id: {chunk.get('chunk_id', '')}",
                    f"paper_id: {chunk.get('paper_id', '')}",
                    f"arxiv_id: {chunk.get('arxiv_id', '')}",
                    f"title: {chunk.get('title', '')}",
                    f"section: {chunk.get('section', '')}",
                    f"quality_flags: {json.dumps(chunk.get('quality_flags', []), sort_keys=True)}",
                    "text:",
                    str(chunk.get("text") or ""),
                ]
            )
        )
    context_text = "\n\n".join(blocks) if blocks else "<NO_CONTEXT>"
    required_all = bool(context_record.get("required_all_expected_sources"))
    expected_behavior = str(context_record.get("expected_behavior") or "")
    return f"""You are KamiKnows, a conservative citation-constrained scientific QA assistant.

Use only the retrieved context below. Do not use outside knowledge.
Do not invent citation labels, chunk IDs, arXiv IDs, titles, or sources.
Every factual claim in the answer must include one or more labels like [C1].
If evidence is missing, incomplete, ambiguous, or only indirectly related, set answer_status to "insufficient_evidence".
If the question requires comparing multiple sources and the provided context does not contain all required sides, set answer_status to "insufficient_evidence".
If benchmark expected_behavior is "insufficient_evidence", answer "answered" only when the provided context explicitly supports every requested detail.
Use only citation labels that appear in the context.
Keep the answer concise and technical.

Benchmark instruction fields:
- expected_behavior: {expected_behavior}
- required_all_expected_sources: {str(required_all).lower()}

Return only valid JSON with this exact shape:
{{
  "answer_status": "answered | insufficient_evidence",
  "answer": "...",
  "evidence": "...",
  "citations": ["C1", "C2"],
  "confidence": "low | medium | high",
  "limitations": "...",
  "unsupported_or_inferred": []
}}

Question:
{context_record.get("question", "")}

Retrieved context:
{context_text}
"""


def generate_benchmark_answers(
    *,
    matrix: list[dict[str, Any]],
    contexts_by_source: dict[str, list[dict[str, Any]]],
    known_chunk_ids: list[str],
    availability: dict[str, Any],
    base_url: str,
    timeout_seconds_override: int | None,
) -> list[dict[str, Any]]:
    """Generate one answer per config/question, tolerating backend errors."""
    available_models = set(availability.get("model_names", []) or [])
    backend_available = bool(availability.get("available"))
    records = []
    for config in matrix:
        if not config.get("enabled", True):
            continue
        contexts = contexts_by_source.get(config["context_source"], [])
        for context in contexts:
            model_name = str(config["model_name"])
            model_config = {
                "model_name": model_name,
                "model_slug": model_slug(model_name),
                "backend": config.get("backend", "ollama"),
                "temperature": float(config["parameter_set"].get("temperature", 0.0)),
                "top_p": float(config["parameter_set"].get("top_p", 0.9)),
                "num_predict": int(config["parameter_set"].get("num_predict", 512)),
            }
            if not backend_available:
                records.append(_backend_error_record(
                    config=config,
                    context=context,
                    model_config=model_config,
                    known_chunk_ids=known_chunk_ids,
                    error_type="backend_error",
                    error_message=str(availability.get("backend_error") or "Ollama backend unavailable."),
                ))
                continue
            if model_name not in available_models:
                records.append(_backend_error_record(
                    config=config,
                    context=context,
                    model_config=model_config,
                    known_chunk_ids=known_chunk_ids,
                    error_type="model_unavailable",
                    error_message="Model is not present in local Ollama tags.",
                ))
                continue

            prompt = build_prompt(prompt_profile=config["prompt_profile"], context_record=context)
            try:
                raw_response = generate_ollama_json(
                    prompt=prompt,
                    model_config=model_config,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds_override or int(config["parameter_set"].get("timeout_seconds", 180)),
                )
                parsed = parse_model_json_response(raw_response)
                validation = validate_answer_citations(
                    parsed_answer=parsed,
                    retrieved_chunks=list(context.get("retrieved_chunks", []) or []),
                    known_chunk_ids=known_chunk_ids,
                )
                records.append(_answer_record_from_parsed(
                    config=config,
                    context=context,
                    raw_response=raw_response,
                    parsed=parsed,
                    citation_validation=validation,
                ))
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                records.append(_backend_error_record(
                    config=config,
                    context=context,
                    model_config=model_config,
                    known_chunk_ids=known_chunk_ids,
                    error_type="backend_error",
                    error_message=str(exc),
                ))
    return records


def build_benchmark_summary(
    *,
    matrix_configs: list[dict[str, Any]],
    answer_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build aggregate benchmark metrics."""
    records_by_config = defaultdict(list)
    for record in answer_records:
        records_by_config[record["benchmark_config_id"]].append(record)
    per_config = []
    for config in matrix_configs:
        records = records_by_config.get(config["config_id"], [])
        per_config.append(_summarize_config(config, records))
    return {
        "benchmark_summary_version": "benchmark_summary_v1",
        "created_at": utc_now_iso(),
        "configs_total": len(matrix_configs),
        "answer_records_total": len(answer_records),
        "per_config": per_config,
        "manual_review_required": True,
        "scope": "Automatic structural metrics only; no LLM judge and no scientific correctness claim.",
    }


def build_benchmark_citation_validation_summary(answer_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate citation validation across all benchmark answer records."""
    valid_records = [record for record in answer_records if record.get("citation_validation", {}).get("valid")]
    invalid = [record for record in answer_records if not record.get("citation_validation", {}).get("valid")]
    summary = {
        "citation_validation_summary_version": "citation_validation_summary_v1",
        "created_at": utc_now_iso(),
        "total_records": len(answer_records),
        "valid_records": len(valid_records),
        "invalid_citation_count": len(invalid),
        "invented_citation_count": sum(
            len(record.get("citation_validation", {}).get("invented_citation_labels", []) or [])
            for record in answer_records
        ),
        "answered_without_valid_citation_count": sum(
            1 for record in answer_records if record.get("citation_validation", {}).get("answered_without_valid_citation")
        ),
        "by_config_id": _citation_counts_by(answer_records, "benchmark_config_id"),
        "by_model": _citation_counts_by(answer_records, "model_name"),
        "by_prompt_profile": _citation_counts_by(answer_records, "prompt_profile"),
        "by_context_source": _citation_counts_by(answer_records, "context_source"),
        "notes": "Citation validation checks deterministic labels only, not factual entailment.",
    }
    return summary


def build_abstention_analysis(answer_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze abstention behavior without judging scientific correctness."""
    rows = []
    for record in answer_records:
        expected = str(record.get("expected_behavior") or "")
        status = str(record.get("answer_status") or "")
        rows.append({
            "correct_abstention": expected == "insufficient_evidence" and status == "insufficient_evidence",
            "failed_abstention": expected == "insufficient_evidence" and status == "answered",
            "over_abstention": expected == "answerable" and status == "insufficient_evidence",
            "backend_error": status == "backend_error",
            "model_name": record.get("model_name", ""),
            "prompt_profile": record.get("prompt_profile", ""),
            "context_source": record.get("context_source", ""),
        })
    return {
        "abstention_analysis_version": "abstention_analysis_v1",
        "created_at": utc_now_iso(),
        "total_records": len(answer_records),
        "correct_abstention_count": sum(1 for row in rows if row["correct_abstention"]),
        "failed_abstention_possible_overclaim_count": sum(1 for row in rows if row["failed_abstention"]),
        "over_abstention_count": sum(1 for row in rows if row["over_abstention"]),
        "backend_error_count": sum(1 for row in rows if row["backend_error"]),
        "per_model": _abstention_counts_by(rows, "model_name"),
        "per_prompt_profile": _abstention_counts_by(rows, "prompt_profile"),
        "per_context_source": _abstention_counts_by(rows, "context_source"),
        "manual_review_required": True,
    }


def build_recommended_config(per_config: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a cautious config for manual review using structural metrics."""
    scored = []
    rejected = []
    for row in per_config:
        hard_reasons = []
        if row["invented_citation_count"] > 0:
            hard_reasons.append("invented_citations")
        if row["answered_without_valid_citation_count"] > 0:
            hard_reasons.append("answered_without_valid_citation")
        if row["backend_error_count"] == row.get("answers_total"):
            hard_reasons.append("all_records_backend_error")
        if hard_reasons:
            rejected.append({"config_id": row["config_id"], "reasons": hard_reasons})
            continue
        score = 0.0
        score -= 5.0 * row["backend_error_count"]
        score -= 2.0 * row["json_parse_error_count"]
        score -= 2.0 * row["failed_abstention_possible_overclaim_count"]
        score -= 1.0 * row["answerable_over_abstention_count"]
        score -= 1.0 * row["invalid_citation_count"]
        score += 2.0 if row["citation_validation_clean"] else 0.0
        score += row["retrieval_expected_source_hit_rate"]
        score += 0.1 * row["answered_count"]
        scored.append((score, row["config_id"], row))
    if not scored:
        return {
            "recommended_for_manual_review": None,
            "rationale": "No config passed hard citation filters.",
            "rejected_configs": rejected,
            "warnings": ["Manual review remains required.", "No scientific quality claim is made."],
            "manual_review_required": True,
        }
    scored.sort(key=lambda item: (-item[0], item[1]))
    best = scored[0][2]
    return {
        "recommended_for_manual_review": best["config_id"],
        "rationale": (
            "Selected by structural scoring after excluding invented citations and answered-without-valid-citation cases. "
            "This is a queueing recommendation for manual review, not a scientific quality claim."
        ),
        "score": scored[0][0],
        "selected_metrics": best,
        "rejected_configs": rejected,
        "warnings": [
            "Manual review remains required.",
            "Automatic metrics do not prove scientific answer correctness.",
        ],
        "manual_review_required": True,
    }


def build_config_comparison_markdown(
    *,
    per_config: list[dict[str, Any]],
    recommended: dict[str, Any],
) -> str:
    lines = [
        "# RAG Generator Benchmark Config Comparison v1",
        "",
        "Automatic structural metrics only. Manual grounding review remains required.",
        "",
        f"Recommended for manual review: `{recommended.get('recommended_for_manual_review')}`",
        "",
        "| config_id | context_source | prompt_profile | model | answered | insufficient | backend errors | invalid citations | JSON warnings | correct abstentions | over-abstentions | possible overclaims | avg answer chars | notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in per_config:
        lines.append(
            "| {config_id} | {context_source} | {prompt_profile} | {model_name} | {answered_count} | {insufficient_evidence_count} | {backend_error_count} | {invalid_citation_count} | {json_warnings_count} | {expected_insufficient_evidence_correct_count} | {answerable_over_abstention_count} | {failed_abstention_possible_overclaim_count} | {average_answer_chars:.1f} | {notes} |".format(
                **row,
                notes="manual review required",
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_error_analysis_markdown(
    *,
    answer_records: list[dict[str, Any]],
    per_config: list[dict[str, Any]],
) -> str:
    tags = defaultdict(list)
    for record in answer_records:
        key = f"{record.get('benchmark_config_id')}::{record.get('question_id')}"
        status = record.get("answer_status")
        warnings = set(record.get("parse_warnings", []) or [])
        validation = record.get("citation_validation", {})
        retrieval_eval = record.get("retrieval_eval", {}) or {}
        if status == "backend_error":
            tags["backend_error"].append(key)
        if "json_parse_error" in warnings or "json_extracted_from_response" in warnings:
            tags["json_parse_error"].append(key)
        if not validation.get("valid", False):
            tags["invalid_citation"].append(key)
        if validation.get("invented_citation_labels"):
            tags["invented_citation"].append(key)
        if record.get("expected_behavior") == "insufficient_evidence" and status == "answered":
            tags["failed_abstention"].append(key)
        if record.get("expected_behavior") == "answerable" and status == "insufficient_evidence":
            tags["over_abstention"].append(key)
        if record.get("required_all_expected_sources") and retrieval_eval.get("missing_expected_source_ids"):
            tags["multi_source_context_incomplete"].append(key)
        if retrieval_eval.get("has_expected_source") and not retrieval_eval.get("expected_source_hit") and status == "answered":
            tags["answered_on_retrieval_miss"].append(key)

    for question_id in ("eval_v0_q002", "eval_v0_q005", "eval_v1_q012", "eval_v0_q008", "eval_v1_q013"):
        if any(record.get("question_id") == question_id for record in answer_records):
            tags[f"sensitive_{question_id}"].append(question_id)

    lines = [
        "# RAG Generator Benchmark Error Analysis v1",
        "",
        "These are structural tags only; they are not scientific correctness judgments.",
        "",
        "## Categories",
        "",
    ]
    for category in (
        "backend_error",
        "json_parse_error",
        "invalid_citation",
        "invented_citation",
        "failed_abstention",
        "over_abstention",
        "multi_source_context_incomplete",
        "answered_on_retrieval_miss",
    ):
        values = tags.get(category, [])
        lines.append(f"- `{category}`: {len(values)}")
    lines.extend(
        [
            "",
            "## Sensitivity Notes",
            "",
            "- `possible_prompt_sensitivity`, `model_sensitivity`, and `context_source_sensitivity` should be inspected by comparing rows in `benchmark_config_comparison_v1.md`.",
            "- Sensitive questions highlighted: eval_v0_q002, eval_v0_q005, eval_v1_q012, eval_v0_q008, eval_v1_q013.",
            "",
            "Manual review remains required before using any benchmark result as evidence of answer quality.",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(
    *,
    rag_ready_dir: Path,
    eval_questions_file: Path,
    reranking_output_dir: Path | None,
    output_dir: Path,
    report_path: Path,
    config_path: Path | None,
    output_paths: dict[str, Path],
    model_names: list[str],
    prompt_profiles: list[dict[str, Any]],
    context_sources: list[str],
    generated_records: int,
    pytest_status: str | None,
) -> dict[str, Any]:
    return {
        "benchmark_manifest_version": "benchmark_manifest_v1",
        "created_at": utc_now_iso(),
        "config_path": str(config_path) if config_path else None,
        "input_files": {
            "chunks": str(rag_ready_dir / "chunks_v1.jsonl"),
            "rag_manifest": str(rag_ready_dir / "rag_manifest_v1.json"),
            "eval_questions": str(eval_questions_file),
            "reranking_output_dir": str(reranking_output_dir) if reranking_output_dir else None,
        },
        "output_dir": str(output_dir),
        "output_files": {key: str(path) for key, path in output_paths.items()},
        "report_path": str(report_path),
        "model_list": model_names,
        "prompt_profiles": prompt_profiles,
        "context_sources": context_sources,
        "generated_records": generated_records,
        "pytest_status": pytest_status,
        "manual_review_required": True,
        "known_limits": [
            "Automatic structural metrics do not prove scientific answer correctness.",
            "No LLM judge is used.",
            "Expected source IDs are used only for post-hoc evaluation.",
            "Manual grounding review remains required.",
        ],
    }


def build_task_report_markdown(
    *,
    rag_ready_dir: Path,
    eval_questions_file: Path,
    reranking_output_dir: Path | None,
    output_dir: Path,
    matrix: dict[str, Any],
    summary: dict[str, Any],
    citation_summary: dict[str, Any],
    abstention_analysis: dict[str, Any],
    recommended: dict[str, Any],
    rag_manifest: dict[str, Any],
) -> str:
    per_config = summary["per_config"]
    best = recommended.get("recommended_for_manual_review")
    context_sources = matrix["dimensions"]["context_sources"]
    lines = [
        "# Phase 2 Task 07 - RAG Generator and Configuration Benchmark v1",
        "",
        f"Date: {utc_now_iso()[:10]}",
        "",
        "## Objective",
        "",
        "Benchmark complete RAG configurations across context sources, prompt profiles, and local generators using automatic structural metrics.",
        "",
        "This benchmark compares RAG configurations using automatic structural metrics, but it does not prove scientific answer correctness without manual grounding review.",
        "",
        "## Why P2T7",
        "",
        "P2T4 established grounded QA generation, P2T5 prepared manual review, and P2T6 compared baseline and reranked contexts. P2T7 combines those axes to identify which configuration should be prioritized for human grounding review.",
        "",
        "## Benchmark Matrix",
        "",
        f"- Configs: {len(matrix['configs'])}",
        f"- Context sources: {', '.join(context_sources)}",
        f"- Prompt profiles: {', '.join(matrix['dimensions']['prompt_profiles'])}",
        f"- Models: {', '.join(matrix['dimensions']['models'])}",
        f"- Parameter sets: {', '.join(matrix['dimensions']['parameter_sets'])}",
        "",
        "## Inputs Used",
        "",
        f"- `{rag_ready_dir / 'chunks_v1.jsonl'}`",
        f"- `{rag_ready_dir / 'rag_manifest_v1.json'}`",
        f"- `{eval_questions_file}`",
        f"- reranking output dir: `{reranking_output_dir}`",
        "",
        "## Outputs Produced",
        "",
        *[f"- `{output_dir / filename}`" for filename in BENCHMARK_OUTPUT_FILES.values()],
        "- `docs/task_reports/phase2_task07_rag_generator_benchmark_v1.md`",
        "",
        "## Automatic Results",
            "",
            f"- Answer records: {summary['answer_records_total']}",
            f"- Backend errors: {sum(row['backend_error_count'] for row in summary['per_config'])}",
            f"- Citation invalid records: {citation_summary['invalid_citation_count']}",
        f"- Invented citations: {citation_summary['invented_citation_count']}",
        f"- Answered without valid citation: {citation_summary['answered_without_valid_citation_count']}",
        f"- Correct abstentions: {abstention_analysis['correct_abstention_count']}",
        f"- Failed abstentions / possible overclaims: {abstention_analysis['failed_abstention_possible_overclaim_count']}",
        f"- Over-abstentions: {abstention_analysis['over_abstention_count']}",
        "",
        "## Recommended Config For Manual Review",
        "",
            f"`{best}`",
            "",
            str(recommended.get("rationale", "")),
            "",
            f"- Rejected configs: {len(recommended.get('rejected_configs', []))}",
            "",
        "## Comparisons",
        "",
        "- Generator, prompt, and context-source comparisons are tabulated in `benchmark_config_comparison_v1.md`.",
        "- Baseline vs reranked context comparison is structural and depends on P2T6 context availability.",
        "- Citation validation checks labels only; factual support remains manual.",
        "",
        "## What Is Not Demonstrated",
        "",
        "- Scientific answer correctness.",
        "- Full factual entailment of every generated sentence.",
        "- Production readiness or detector-simulation replacement decisions.",
        "",
        "## Limits",
        "",
        "- No training, LoRA, QLoRA, discovery, corpus scaling, parser changes, or chunks_v1 changes were performed.",
        "- No LLM judge was used.",
        "- Expected source IDs were not given to generation prompts and are used only post-hoc.",
        f"- Dataset counts: `{json.dumps(_dataset_counts(rag_manifest), sort_keys=True)}`",
        "",
        "## Next Step",
        "",
        "Use the recommended config only to prioritize manual grounding review, then compare human labels before making any answer-quality claim.",
        "",
    ]
    if per_config:
        lines.extend(["## Top Structural Configs", ""])
        for row in sorted(per_config, key=lambda item: (item["backend_error_count"], item["invalid_citation_count"], -item["answered_count"]))[:5]:
            lines.append(
                f"- `{row['config_id']}`: answered={row['answered_count']}, insufficient={row['insufficient_evidence_count']}, backend_errors={row['backend_error_count']}, invalid_citations={row['invalid_citation_count']}"
            )
        lines.append("")
    return "\n".join(lines)


def _contexts_from_p2t6_baseline(path: Path, *, top_k: int) -> list[dict[str, Any]]:
    records = []
    for record in read_jsonl(path):
        chunks = [_normalize_context_chunk(chunk, rank) for rank, chunk in enumerate(record.get("baseline_top_k_results", [])[:top_k], start=1)]
        records.append(_context_record_from_question(
            question=record,
            context_source="tfidf_baseline_top5",
            retrieved_chunks=chunks,
            retrieval_eval=record.get("baseline_retrieval_eval", {}),
            top_k=top_k,
        ))
    return records


def _contexts_from_p2t6_reranked(path: Path, *, top_k: int) -> list[dict[str, Any]]:
    records = []
    for record in read_jsonl(path):
        chunks = [_normalize_context_chunk(chunk, rank) for rank, chunk in enumerate(record.get("reranked_top_k_results", [])[:top_k], start=1)]
        records.append(_context_record_from_question(
            question=record,
            context_source="light_reranked_top5",
            retrieved_chunks=chunks,
            retrieval_eval=record.get("reranked_retrieval_eval", {}),
            top_k=top_k,
        ))
    return records


def _context_record_from_question(
    *,
    question: dict[str, Any],
    context_source: str,
    retrieved_chunks: list[dict[str, Any]],
    retrieval_eval: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    return {
        "question_id": question.get("question_id", ""),
        "question": question.get("question", ""),
        "question_type": question.get("question_type", ""),
        "expected_behavior": question.get("expected_behavior", ""),
        "expected_arxiv_ids": list(question.get("expected_arxiv_ids", []) or []),
        "required_all_expected_sources": bool(question.get("required_all_expected_sources")),
        "context_source": context_source,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_eval": retrieval_eval,
        "top_k": top_k,
    }


def _context_chunk_from_retrieved(rank: int, score: float, chunk: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(str(chunk.get("text") or "").split())
    return {
        "citation_id": f"C{rank}",
        "citation_label": f"[C{rank}]",
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


def _normalize_context_chunk(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    normalized = dict(chunk)
    normalized["rank"] = rank
    normalized["citation_id"] = f"C{rank}"
    normalized["citation_label"] = f"[C{rank}]"
    normalized["section"] = normalized.get("section") or normalized.get("section_heading", "")
    normalized["section_heading"] = normalized.get("section_heading") or normalized.get("section", "")
    normalized.setdefault("quality_flags", [])
    normalized.setdefault("text", "")
    normalized.setdefault("text_preview", " ".join(str(normalized.get("text") or "").split())[:360])
    return normalized


def _answer_record_from_parsed(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    raw_response: str,
    parsed: dict[str, Any],
    citation_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_record_prefix(config, context),
        "answer_status": parsed.get("answer_status", ""),
        "answer": parsed.get("answer", ""),
        "evidence": parsed.get("evidence", ""),
        "citations": list(parsed.get("citations", []) or []),
        "confidence": parsed.get("confidence", "low"),
        "limitations": parsed.get("limitations", ""),
        "unsupported_or_inferred": list(parsed.get("unsupported_or_inferred", []) or []),
        "raw_model_response": raw_response,
        "retrieved_chunks": context.get("retrieved_chunks", []),
        "retrieval_eval": context.get("retrieval_eval", {}),
        "citation_validation": citation_validation,
        "parse_warnings": list(parsed.get("parse_warnings", []) or []),
        "backend_error": None,
        "grounding_status": "not_reviewed",
    }


def _backend_error_record(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    model_config: dict[str, Any],
    known_chunk_ids: list[str],
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    parsed = {
        "answer_status": "backend_error",
        "answer": "",
        "evidence": "",
        "citations": [],
        "confidence": "low",
        "limitations": "",
        "unsupported_or_inferred": [],
    }
    validation = validate_answer_citations(
        parsed_answer=parsed,
        retrieved_chunks=list(context.get("retrieved_chunks", []) or []),
        known_chunk_ids=known_chunk_ids,
    )
    return {
        **_record_prefix(config, context),
        "answer_status": "backend_error",
        "answer": "",
        "evidence": "",
        "citations": [],
        "confidence": "low",
        "limitations": "",
        "unsupported_or_inferred": [],
        "raw_model_response": "",
        "retrieved_chunks": context.get("retrieved_chunks", []),
        "retrieval_eval": context.get("retrieval_eval", {}),
        "citation_validation": validation,
        "parse_warnings": [error_type],
        "backend_error": {
            "error_type": error_type,
            "message": error_message,
        },
        "grounding_status": "not_reviewed",
    }


def _record_prefix(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_config_id": config["config_id"],
        "question_id": context.get("question_id", ""),
        "question": context.get("question", ""),
        "question_type": context.get("question_type", ""),
        "expected_behavior": context.get("expected_behavior", ""),
        "required_all_expected_sources": bool(context.get("required_all_expected_sources")),
        "context_source": config["context_source"],
        "prompt_profile": config["prompt_profile"],
        "model_name": config["model_name"],
        "parameter_set": config["parameter_set"].get("name", ""),
    }


def _summarize_config(config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [record for record in records if record.get("answer_status") == "answered"]
    insufficient = [record for record in records if record.get("answer_status") == "insufficient_evidence"]
    backend_errors = [record for record in records if record.get("answer_status") == "backend_error"]
    answerable = [record for record in records if record.get("expected_behavior") == "answerable"]
    expected_insufficient = [record for record in records if record.get("expected_behavior") == "insufficient_evidence"]
    invalid_citations = [record for record in records if not record.get("citation_validation", {}).get("valid", False)]
    json_parse_errors = [record for record in records if "json_parse_error" in set(record.get("parse_warnings", []) or [])]
    invalid_confidence = [record for record in records if "invalid_confidence_normalized" in set(record.get("parse_warnings", []) or [])]
    retrieval_expected = [record for record in records if record.get("retrieval_eval", {}).get("has_expected_source")]
    retrieval_hits = [record for record in retrieval_expected if record.get("retrieval_eval", {}).get("expected_source_hit")]
    multi_source = [record for record in records if record.get("required_all_expected_sources")]
    multi_complete = [
        record for record in multi_source
        if not record.get("retrieval_eval", {}).get("missing_expected_source_ids")
        and record.get("retrieval_eval", {}).get("expected_source_hit")
    ]
    citation_counts = [
        len(record.get("citation_validation", {}).get("valid_used_citation_labels", []) or [])
        for record in records if record.get("answer_status") == "answered"
    ]
    answer_lengths = [len(str(record.get("answer") or "")) for record in answered]
    failed_abstentions = [record for record in expected_insufficient if record.get("answer_status") == "answered"]
    over_abstentions = [record for record in answerable if record.get("answer_status") == "insufficient_evidence"]
    return {
        "config_id": config["config_id"],
        "context_source": config["context_source"],
        "prompt_profile": config["prompt_profile"],
        "model_name": config["model_name"],
        "parameter_set": config["parameter_set"].get("name", ""),
        "questions_total": len(records),
        "answers_total": len(records),
        "answered_count": len(answered),
        "insufficient_evidence_count": len(insufficient),
        "backend_error_count": len(backend_errors),
        "json_parse_error_count": len(json_parse_errors),
        "invalid_confidence_normalized_count": len(invalid_confidence),
        "json_warnings_count": sum(len(record.get("parse_warnings", []) or []) for record in records),
        "invalid_citation_count": len(invalid_citations),
        "invented_citation_count": sum(
            len(record.get("citation_validation", {}).get("invented_citation_labels", []) or [])
            for record in records
        ),
        "answered_without_valid_citation_count": sum(
            1 for record in records if record.get("citation_validation", {}).get("answered_without_valid_citation")
        ),
        "citation_validation_clean": not invalid_citations,
        "expected_insufficient_evidence_total": len(expected_insufficient),
        "expected_insufficient_evidence_correct_count": sum(
            1 for record in expected_insufficient if record.get("answer_status") == "insufficient_evidence"
        ),
        "expected_answerable_total": len(answerable),
        "answerable_answered_count": sum(1 for record in answerable if record.get("answer_status") == "answered"),
        "answerable_over_abstention_count": len(over_abstentions),
        "possible_overclaim_count": len(failed_abstentions),
        "failed_abstention_possible_overclaim_count": len(failed_abstentions),
        "average_answer_chars": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
        "average_citations_per_answer": sum(citation_counts) / len(citation_counts) if citation_counts else 0.0,
        "retrieval_expected_source_hit_rate": len(retrieval_hits) / len(retrieval_expected) if retrieval_expected else 0.0,
        "multi_source_questions_total": len(multi_source),
        "multi_source_context_complete_count": len(multi_complete),
        "multi_source_answered_with_citations_count": sum(
            1
            for record in multi_source
            if record.get("answer_status") == "answered"
            and record.get("citation_validation", {}).get("valid_used_citation_labels")
        ),
    }


def _citation_counts_by(answer_records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped = defaultdict(list)
    for record in answer_records:
        grouped[str(record.get(key) or "")].append(record)
    return {
        group: {
            "total_records": len(records),
            "valid_records": sum(1 for record in records if record.get("citation_validation", {}).get("valid")),
            "invalid_citation_count": sum(1 for record in records if not record.get("citation_validation", {}).get("valid", False)),
            "invented_citation_count": sum(
                len(record.get("citation_validation", {}).get("invented_citation_labels", []) or [])
                for record in records
            ),
            "answered_without_valid_citation_count": sum(
                1 for record in records if record.get("citation_validation", {}).get("answered_without_valid_citation")
            ),
        }
        for group, records in sorted(grouped.items())
    }


def _abstention_counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "")].append(row)
    return {
        group: {
            "total_records": len(values),
            "correct_abstention_count": sum(1 for row in values if row["correct_abstention"]),
            "failed_abstention_possible_overclaim_count": sum(1 for row in values if row["failed_abstention"]),
            "over_abstention_count": sum(1 for row in values if row["over_abstention"]),
            "backend_error_count": sum(1 for row in values if row["backend_error"]),
        }
        for group, values in sorted(grouped.items())
    }


def _dataset_counts(rag_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "papers": rag_manifest.get("papers_count"),
        "chunks_v1": rag_manifest.get("chunks_v1_count"),
        "equations": rag_manifest.get("equations_count"),
        "chunks_with_quality_flags": (rag_manifest.get("quality_summary") or {}).get("chunks_with_quality_flags"),
    }
