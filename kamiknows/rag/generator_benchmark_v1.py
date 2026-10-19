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
    extract_citation_labels,
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

DEBUG_OUTPUT_FILES = {
    "citation_failure_audit": "citation_failure_audit_v1.jsonl",
    "citation_failure_summary": "citation_failure_summary_v1.json",
    "prompt_profile_failure_summary": "prompt_profile_failure_summary_v1.json",
    "context_label_mapping_summary": "context_label_mapping_summary_v1.json",
    "manifest": "benchmark_debug_manifest_v1.json",
}

FAILURE_CAUSES = {
    "model_returned_empty_citations_array",
    "citations_in_text_but_not_array",
    "citation_label_format_mismatch",
    "citation_label_not_in_context",
    "answer_status_answered_but_no_citation",
    "json_parse_fallback_lost_citations",
    "context_label_mapping_error",
    "prompt_noncompliance",
    "unknown",
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
    limit_questions: int | None = None,
    limit_configs: int | None = None,
    question_ids: list[str] | None = None,
    config_ids: list[str] | None = None,
    dry_run_matrix: bool = False,
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
    eval_questions = filter_eval_questions(
        eval_questions,
        question_ids=question_ids,
        limit_questions=limit_questions,
    )
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
    matrix["configs"] = filter_benchmark_configs(
        matrix["configs"],
        config_ids=config_ids,
        limit_configs=limit_configs,
    )
    matrix["selection"] = {
        "question_ids": list(question_ids or []),
        "limit_questions": limit_questions,
        "config_ids": list(config_ids or []),
        "limit_configs": limit_configs,
        "dry_run_matrix": dry_run_matrix,
    }
    if dry_run_matrix:
        return {
            "chunks": chunks,
            "eval_questions": eval_questions,
            "contexts_by_source": contexts_by_source,
            "matrix": matrix,
            "answer_records": [],
            "summary": None,
            "citation_summary": None,
            "abstention_analysis": None,
            "recommended_config": None,
            "manifest": None,
            "output_paths": {},
            "report_path": str(report_path),
        }

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


def filter_eval_questions(
    eval_questions: list[dict[str, Any]],
    *,
    question_ids: list[str] | None = None,
    limit_questions: int | None = None,
) -> list[dict[str, Any]]:
    """Apply deterministic question selection for reduced benchmark runs."""
    selected = list(eval_questions)
    if question_ids:
        wanted = set(str(question_id) for question_id in question_ids)
        selected = [question for question in selected if str(question.get("question_id") or "") in wanted]
    if limit_questions is not None:
        if limit_questions < 1:
            raise ValueError("limit_questions must be >= 1")
        selected = selected[:limit_questions]
    return selected


def filter_benchmark_configs(
    configs: list[dict[str, Any]],
    *,
    config_ids: list[str] | None = None,
    limit_configs: int | None = None,
) -> list[dict[str, Any]]:
    """Apply deterministic config selection for reduced benchmark runs."""
    selected = list(configs)
    if config_ids:
        wanted = set(str(config_id) for config_id in config_ids)
        selected = [config for config in selected if str(config.get("config_id") or "") in wanted]
    if limit_configs is not None:
        if limit_configs < 1:
            raise ValueError("limit_configs must be >= 1")
        selected = selected[:limit_configs]
    return selected


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
For answer_status="answered", citations must contain at least one valid citation_id from the retrieved context, such as "C1".
If you cannot include a valid retrieved citation_id, set answer_status to "insufficient_evidence".
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


def run_benchmark_citation_debug_v1(
    *,
    benchmark_output_dir: Path,
    output_dir: Path,
    report_path: Path,
    pytest_status: str | None = None,
) -> dict[str, Any]:
    """Diagnose P2T7 citation failures without rerunning the full benchmark."""
    benchmark_output_dir = Path(benchmark_output_dir)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    answer_records = read_jsonl(benchmark_output_dir / BENCHMARK_OUTPUT_FILES["benchmark_answers"])

    audit = build_citation_failure_audit(answer_records)
    context_mapping_summary = validate_context_label_mapping(answer_records)
    failure_summary = build_citation_failure_summary(
        answer_records=answer_records,
        audit_records=audit,
        context_label_mapping_summary=context_mapping_summary,
    )
    axis_summary = build_prompt_profile_failure_summary(
        answer_records=answer_records,
        audit_records=audit,
    )
    output_paths = {key: output_dir / filename for key, filename in DEBUG_OUTPUT_FILES.items()}
    write_jsonl(output_paths["citation_failure_audit"], audit)
    write_json(output_paths["citation_failure_summary"], failure_summary)
    write_json(output_paths["prompt_profile_failure_summary"], axis_summary)
    write_json(output_paths["context_label_mapping_summary"], context_mapping_summary)
    manifest = {
        "benchmark_debug_manifest_version": "benchmark_debug_manifest_v1",
        "created_at": utc_now_iso(),
        "source_benchmark_output_dir": str(benchmark_output_dir),
        "source_answer_file": str(benchmark_output_dir / BENCHMARK_OUTPUT_FILES["benchmark_answers"]),
        "output_dir": str(output_dir),
        "output_files": {key: str(path) for key, path in output_paths.items()},
        "report_path": str(report_path),
        "full_benchmark_rerun": False,
        "pytest_status": pytest_status,
        "manual_review_required": True,
        "scope": "Static citation-failure diagnostics over existing P2T7 answer records.",
    }
    write_json(output_paths["manifest"], manifest)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_citation_debug_report_markdown(
            benchmark_output_dir=benchmark_output_dir,
            output_dir=output_dir,
            failure_summary=failure_summary,
            axis_summary=axis_summary,
            context_label_mapping_summary=context_mapping_summary,
            pytest_status=pytest_status,
        ),
        encoding="utf-8",
    )
    return {
        "answer_records": answer_records,
        "audit_records": audit,
        "failure_summary": failure_summary,
        "prompt_profile_failure_summary": axis_summary,
        "context_label_mapping_summary": context_mapping_summary,
        "manifest": manifest,
        "output_paths": {key: str(path) for key, path in output_paths.items()},
        "report_path": str(report_path),
    }


def build_citation_failure_audit(answer_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build detailed audit rows for records with invalid citation validation."""
    audit = []
    for record in answer_records:
        validation = record.get("citation_validation", {}) or {}
        if validation.get("valid", False):
            continue
        extracted = extract_used_citation_labels_from_record(record)
        available_labels = available_retrieved_citation_labels(record)
        mapping_issues = _context_label_mapping_issues_for_record(record)
        audit.append(
            {
                "benchmark_config_id": record.get("benchmark_config_id", ""),
                "question_id": record.get("question_id", ""),
                "question_type": record.get("question_type", ""),
                "model_name": record.get("model_name", ""),
                "prompt_profile": record.get("prompt_profile", ""),
                "context_source": record.get("context_source", ""),
                "answer_status": record.get("answer_status", ""),
                "expected_behavior": record.get("expected_behavior", ""),
                "answer": record.get("answer", ""),
                "evidence": record.get("evidence", ""),
                "limitations": record.get("limitations", ""),
                "citations": list(record.get("citations", []) or []),
                "raw_model_response": record.get("raw_model_response", ""),
                "available_retrieved_labels": available_labels,
                "used_citation_labels_by_field": extracted["by_field"],
                "used_citation_labels": extracted["all"],
                "invented_citation_labels": list(validation.get("invented_citation_labels", []) or []),
                "answered_without_valid_citation": bool(validation.get("answered_without_valid_citation")),
                "parse_warnings": list(record.get("parse_warnings", []) or []),
                "citation_validation_status": validation.get("status", ""),
                "citation_validation_valid": bool(validation.get("valid")),
                "valid_used_citation_labels": list(validation.get("valid_used_citation_labels", []) or []),
                "context_label_mapping_issues": mapping_issues,
                "suspected_failure_cause": classify_citation_failure(record),
                "grounding_status": record.get("grounding_status", "not_reviewed"),
            }
        )
    return audit


def extract_used_citation_labels_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Extract citation labels from structured and text answer fields."""
    by_field = {
        "citations": _extract_labels_from_value(record.get("citations", [])),
        "answer": extract_citation_labels(str(record.get("answer") or "")),
        "evidence": extract_citation_labels(str(record.get("evidence") or "")),
        "limitations": extract_citation_labels(str(record.get("limitations") or "")),
        "unsupported_or_inferred": _extract_labels_from_value(record.get("unsupported_or_inferred", [])),
    }
    labels: list[str] = []
    for values in by_field.values():
        labels.extend(values)
    return {
        "by_field": {key: _dedupe_labels(values) for key, values in by_field.items()},
        "all": _dedupe_labels(labels),
    }


def available_retrieved_citation_labels(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the final context label map visible to the generator."""
    labels = []
    for chunk in record.get("retrieved_chunks", []) or []:
        labels.append(
            {
                "rank": chunk.get("rank"),
                "citation_id": str(chunk.get("citation_id") or ""),
                "citation_label": str(chunk.get("citation_label") or ""),
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "paper_id": str(chunk.get("paper_id") or ""),
                "arxiv_id": str(chunk.get("arxiv_id") or ""),
            }
        )
    return labels


def classify_citation_failure(record: dict[str, Any]) -> str:
    """Classify a citation-validation failure into a deterministic cause bucket."""
    validation = record.get("citation_validation", {}) or {}
    extracted = extract_used_citation_labels_from_record(record)
    by_field = extracted["by_field"]
    available = {item["citation_id"] for item in available_retrieved_citation_labels(record) if item["citation_id"]}
    structured_labels = set(by_field.get("citations", []) or [])
    text_labels = set(by_field.get("answer", []) or [])
    text_labels.update(by_field.get("evidence", []) or [])
    text_labels.update(by_field.get("limitations", []) or [])
    warnings = set(str(warning) for warning in record.get("parse_warnings", []) or [])
    answer_status = str(record.get("answer_status") or "")
    has_answer_text = bool(str(record.get("answer") or "").strip() or str(record.get("evidence") or "").strip())

    if _context_label_mapping_issues_for_record(record):
        return "context_label_mapping_error"
    if text_labels & available and not structured_labels:
        return "citations_in_text_but_not_array"
    if validation.get("invented_citation_labels") or (set(extracted["all"]) - available):
        return "citation_label_not_in_context"
    if record.get("citations") and not structured_labels:
        return "citation_label_format_mismatch"
    if "json_parse_error" in warnings and not extracted["all"]:
        return "json_parse_fallback_lost_citations"
    if answer_status == "answered" and not validation.get("valid_used_citation_labels"):
        if not record.get("citations") and not has_answer_text:
            return "answer_status_answered_but_no_citation"
        if not record.get("citations"):
            return "model_returned_empty_citations_array"
        return "prompt_noncompliance"
    if not validation.get("valid", True):
        return "prompt_noncompliance"
    return "unknown"


def validate_context_label_mapping(answer_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check final retrieved context labels for deterministic C1... mapping."""
    issues = []
    chunks_checked = 0
    for record in answer_records:
        record_issues = _context_label_mapping_issues_for_record(record)
        chunks_checked += len(record.get("retrieved_chunks", []) or [])
        for issue in record_issues:
            issues.append(
                {
                    "benchmark_config_id": record.get("benchmark_config_id", ""),
                    "question_id": record.get("question_id", ""),
                    "context_source": record.get("context_source", ""),
                    **issue,
                }
            )
    return {
        "context_label_mapping_summary_version": "context_label_mapping_summary_v1",
        "created_at": utc_now_iso(),
        "records_checked": len(answer_records),
        "chunks_checked": chunks_checked,
        "issue_count": len(issues),
        "status": "valid" if not issues else "invalid",
        "issues": issues[:100],
        "notes": "Labels are checked against the final context order passed to each generator record.",
    }


def build_citation_failure_summary(
    *,
    answer_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
    context_label_mapping_summary: dict[str, Any],
) -> dict[str, Any]:
    """Summarize invalid citation records and suspected causes."""
    cause_counts = Counter(str(record.get("suspected_failure_cause") or "unknown") for record in audit_records)
    status_counts = Counter(str(record.get("citation_validation_status") or "unknown") for record in audit_records)
    answered_without_valid = sum(1 for record in audit_records if record.get("answered_without_valid_citation"))
    invented = sum(len(record.get("invented_citation_labels", []) or []) for record in audit_records)
    main_cause = cause_counts.most_common(1)[0][0] if cause_counts else None
    return {
        "citation_failure_summary_version": "citation_failure_summary_v1",
        "created_at": utc_now_iso(),
        "source_answer_records": len(answer_records),
        "invalid_citation_records": len(audit_records),
        "invented_citation_labels": invented,
        "answered_without_valid_citation_records": answered_without_valid,
        "citation_validation_status_counts": dict(sorted(status_counts.items())),
        "suspected_failure_cause_counts": dict(sorted(cause_counts.items())),
        "main_suspected_failure_cause": main_cause,
        "context_label_mapping_status": context_label_mapping_summary.get("status", ""),
        "context_label_mapping_issue_count": context_label_mapping_summary.get("issue_count", 0),
        "failure_cause_taxonomy": sorted(FAILURE_CAUSES),
        "manual_review_required": True,
        "notes": "Cause classification is deterministic debugging metadata; it is not a scientific correctness judgment.",
    }


def build_prompt_profile_failure_summary(
    *,
    answer_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize citation failures by the required benchmark axes."""
    axes = ["model_name", "prompt_profile", "context_source", "question_type", "benchmark_config_id"]
    return {
        "prompt_profile_failure_summary_version": "prompt_profile_failure_summary_v1",
        "created_at": utc_now_iso(),
        "axes": {
            axis: _failure_counts_by_axis(
                answer_records=answer_records,
                audit_records=audit_records,
                axis=axis,
            )
            for axis in axes
        },
        "notes": "Rates use all benchmark answer records as the denominator for each axis value.",
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


def build_citation_debug_report_markdown(
    *,
    benchmark_output_dir: Path,
    output_dir: Path,
    failure_summary: dict[str, Any],
    axis_summary: dict[str, Any],
    context_label_mapping_summary: dict[str, Any],
    pytest_status: str | None,
) -> str:
    """Build the Phase 2 Task 07b report."""
    causes = failure_summary.get("suspected_failure_cause_counts", {}) or {}
    statuses = failure_summary.get("citation_validation_status_counts", {}) or {}
    prompt_axis = (axis_summary.get("axes", {}) or {}).get("prompt_profile", {})
    model_axis = (axis_summary.get("axes", {}) or {}).get("model_name", {})
    lines = [
        "# Phase 2 Task 07b - Benchmark Citation Failure Debug v1",
        "",
        f"Date: {utc_now_iso()[:10]}",
        "",
        "## Objective",
        "",
        "Diagnose P2T7 benchmark citation-validation failures and harden the reduced benchmark controls and future prompt/validator behavior.",
        "",
        "The full P2T7 benchmark was not rerun in this task; only reduced validation or static diagnostics were performed.",
        "",
        "## Inputs Used",
        "",
        f"- `{benchmark_output_dir / BENCHMARK_OUTPUT_FILES['benchmark_answers']}`",
        f"- existing P2T7 output dir: `{benchmark_output_dir}`",
        "",
        "## Outputs Produced",
        "",
        *[f"- `{output_dir / filename}`" for filename in DEBUG_OUTPUT_FILES.values()],
        "- `docs/task_reports/phase2_task07b_benchmark_citation_debug_v1.md`",
        "",
        "## Commands Run",
        "",
        "- Static diagnostics over existing `benchmark_answers_v1.jsonl`",
        "- `pytest -q`" if pytest_status else "- `pytest -q` was pending when this report was written.",
        "",
        "## Key Metrics",
        "",
        f"- Source answer records: {failure_summary.get('source_answer_records')}",
        f"- Invalid citation records: {failure_summary.get('invalid_citation_records')}",
        f"- Answered without valid citation records: {failure_summary.get('answered_without_valid_citation_records')}",
        f"- Invented citation labels: {failure_summary.get('invented_citation_labels')}",
        f"- Main suspected failure cause: `{failure_summary.get('main_suspected_failure_cause')}`",
        f"- Context label mapping status: `{context_label_mapping_summary.get('status')}`",
        f"- Context label mapping issues: {context_label_mapping_summary.get('issue_count')}",
        "",
        "## Failure Cause Counts",
        "",
    ]
    for cause, count in sorted(causes.items()):
        lines.append(f"- `{cause}`: {count}")
    lines.extend(["", "## Citation Validation Status Counts", ""])
    for status, count in sorted(statuses.items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Axis Highlights", ""])
    for name, values in (("prompt_profile", prompt_axis), ("model_name", model_axis)):
        lines.append(f"### {name}")
        for value, row in sorted(values.items()):
            lines.append(
                f"- `{value}`: invalid={row.get('invalid_citation_records')}/{row.get('total_records')}, causes={json.dumps(row.get('suspected_failure_cause_counts', {}), sort_keys=True)}"
            )
        lines.append("")
    lines.extend(
        [
            "## Code Hardening",
            "",
            "- Future P2T7 prompts now state that `answer_status=\"answered\"` requires a non-empty `citations` array containing at least one valid retrieved `citation_id`.",
            "- Citation extraction now uses safer label boundaries while still accepting `[C1]` and `C1` forms.",
            "- The benchmark CLI now supports reduced question/config selection and dry-run matrix inspection.",
            "",
            "## Limits And Non-Claims",
            "",
            "- No training, LoRA, QLoRA, discovery generation, corpus scaling, parser changes to source full text, or chunks_v1 changes were performed.",
            "- Expected source IDs were not added to generation prompts.",
            "- These diagnostics classify structural citation failures only; they do not establish factual entailment or scientific answer correctness.",
            "- Manual grounding review remains required.",
            "",
            "## Next Recommended Step",
            "",
            "Use the debug audit to inspect the top invalid configs, then run at most a small reduced benchmark before choosing any manual-review queue.",
            "",
            f"Pytest status: {pytest_status or 'pending'}",
            "",
        ]
    )
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


def _extract_labels_from_value(value: Any) -> list[str]:
    labels: list[str] = []
    if value is None:
        return labels
    if isinstance(value, dict):
        for inner in value.values():
            labels.extend(_extract_labels_from_value(inner))
        return labels
    if isinstance(value, (list, tuple, set)):
        for inner in value:
            labels.extend(_extract_labels_from_value(inner))
        return labels
    return extract_citation_labels(str(value))


def _context_label_mapping_issues_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, chunk in enumerate(record.get("retrieved_chunks", []) or [], start=1):
        expected_id = f"C{index}"
        citation_id = str(chunk.get("citation_id") or "")
        citation_label = str(chunk.get("citation_label") or "")
        rank = chunk.get("rank")
        if citation_id != expected_id:
            issues.append(
                {
                    "rank": rank,
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "issue": "citation_id_not_sequential_final_context",
                    "expected_citation_id": expected_id,
                    "actual_citation_id": citation_id,
                }
            )
        if citation_label != f"[{expected_id}]":
            issues.append(
                {
                    "rank": rank,
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "issue": "citation_label_not_sequential_final_context",
                    "expected_citation_label": f"[{expected_id}]",
                    "actual_citation_label": citation_label,
                }
            )
        if citation_id and citation_id in seen:
            issues.append(
                {
                    "rank": rank,
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "issue": "duplicate_citation_id",
                    "actual_citation_id": citation_id,
                }
            )
        seen.add(citation_id)
    return issues


def _failure_counts_by_axis(
    *,
    answer_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
    axis: str,
) -> dict[str, Any]:
    totals = Counter(str(record.get(axis) or "") for record in answer_records)
    grouped = defaultdict(list)
    for record in audit_records:
        grouped[str(record.get(axis) or "")].append(record)
    values = sorted(set(totals) | set(grouped))
    summary = {}
    for value in values:
        invalid = grouped.get(value, [])
        cause_counts = Counter(str(record.get("suspected_failure_cause") or "unknown") for record in invalid)
        status_counts = Counter(str(record.get("citation_validation_status") or "unknown") for record in invalid)
        total = totals.get(value, 0)
        summary[value] = {
            "total_records": total,
            "invalid_citation_records": len(invalid),
            "invalid_citation_rate": len(invalid) / total if total else 0.0,
            "answered_without_valid_citation_records": sum(
                1 for record in invalid if record.get("answered_without_valid_citation")
            ),
            "invented_citation_labels": sum(
                len(record.get("invented_citation_labels", []) or []) for record in invalid
            ),
            "suspected_failure_cause_counts": dict(sorted(cause_counts.items())),
            "citation_validation_status_counts": dict(sorted(status_counts.items())),
        }
    return summary


def _dedupe_labels(values: list[str]) -> list[str]:
    seen: set[str] = set()
    labels = []
    for value in values:
        label = str(value or "").upper()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


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
