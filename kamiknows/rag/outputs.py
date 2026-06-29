"""Output writers and orchestration for mini-RAG v0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kamiknows.rag.evaluation import (
    build_error_analysis_markdown,
    build_eval_summary,
    build_retrieved_context_record,
)
from kamiknows.rag.generator import generate_answer_records
from kamiknows.rag.load_dataset import load_rag_ready_dataset
from kamiknows.rag.retriever import TfidfRetriever
from kamiknows.run_metadata import utc_now_iso


REQUIRED_OUTPUT_FILES = {
    "rag_run_config": "rag_run_config.json",
    "retrieved_contexts": "retrieved_contexts.jsonl",
    "rag_answers": "rag_answers.jsonl",
    "rag_eval_summary": "rag_eval_summary.json",
    "rag_error_analysis": "rag_error_analysis.md",
    "rag_manual_review_checklist": "rag_manual_review_checklist.md",
    "rag_manifest": "rag_manifest.json",
}


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def build_run_config(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    backend: str,
    model: str,
    top_k: int,
    retrieval_only: bool,
    base_url: str,
    temperature: float,
) -> dict[str, Any]:
    return {
        "run_config_version": "rag_run_config_v0",
        "rag_ready_dir": str(rag_ready_dir),
        "output_dir": str(output_dir),
        "backend": backend,
        "model": model,
        "top_k": top_k,
        "retrieval_only": retrieval_only,
        "base_url": base_url,
        "temperature": temperature,
        "retriever": {
            "name": "tfidf_local_v0",
            "version": "retriever_v0",
        },
        "scope": "mini-RAG v0; no fine-tuning, no discovery, no automatic scientific correctness judgment",
    }


def retrieve_contexts(
    *,
    chunks: list[dict[str, Any]],
    eval_questions: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    retriever = TfidfRetriever(chunks)
    records = []
    for question in eval_questions:
        retrieved_chunks = [
            retrieved.to_context_dict()
            for retrieved in retriever.retrieve(str(question.get("question") or ""), top_k=top_k)
        ]
        records.append(
            build_retrieved_context_record(
                question=question,
                retrieved_chunks=retrieved_chunks,
                top_k=top_k,
            )
        )
    return records


def build_manual_review_checklist(
    *,
    retrieved_contexts: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> str:
    answers_by_id = {answer["question_id"]: answer for answer in answers}
    lines = [
        "# KamiKnows Mini-RAG v0 Manual Review Checklist",
        "",
        "Manual review is required. Do not treat retrieval or generation as scientific QA until reviewed.",
        "",
    ]
    for context in retrieved_contexts:
        answer = answers_by_id.get(context["question_id"], {})
        lines.extend(
            [
                f"## {context['question_id']}",
                "",
                f"Question: {context['question']}",
                "",
                "Expected source arXiv IDs: "
                + ", ".join(context.get("expected_source_arxiv_ids", []) or ["<none>"]),
                "",
                "Retrieved chunks:",
            ]
        )
        for chunk in context.get("retrieved_chunks", []):
            lines.append(
                f"- rank {chunk['rank']}: `{chunk['chunk_id']}` / `{chunk['arxiv_id']}` / {chunk.get('section_heading', '')}"
            )
        lines.extend(
            [
                "",
                "Generated answer:",
                "",
                answer.get("answer") or "<no answer generated>",
                "",
                "Checklist:",
                "",
                "- retrieval_useful: yes/no/unclear",
                "- answer_grounded: yes/no/unclear",
                "- citations_correct: yes/no/unclear",
                "- no_hallucination: yes/no/unclear",
                "- abstained_when_needed: yes/no/unclear/not_applicable",
                "- outcome: pass/revise/reject/unclear",
                "- notes:",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_rag_manifest(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    dataset_counts: dict[str, int],
    backend: str,
    model: str,
    top_k: int,
    retrieval_only: bool,
) -> dict[str, Any]:
    files = {
        key: str(output_dir / filename)
        for key, filename in REQUIRED_OUTPUT_FILES.items()
        if key != "rag_manifest"
    }
    missing_outputs = [
        path
        for path in files.values()
        if not Path(path).exists()
    ]
    warnings = []
    if retrieval_only:
        warnings.append("Generation skipped because retrieval-only mode was used.")
    validation_status = "PASS" if not missing_outputs else "FAIL"
    return {
        "rag_run_name": output_dir.name or "rag_v0_fastcalosim",
        "source_rag_ready_dir": str(rag_ready_dir),
        "output_dir": str(output_dir),
        "created_at": utc_now_iso(),
        "dataset_counts": dataset_counts,
        "retriever": {
            "name": "tfidf_local_v0",
            "top_k": top_k,
        },
        "generator": {
            "backend": backend,
            "model": model,
            "enabled": not retrieval_only,
        },
        "files": files,
        "validation": {
            "status": validation_status,
            "missing_outputs": missing_outputs,
            "warnings": warnings,
        },
        "scope": "mini-RAG v0; no fine-tuning, no discovery, no automatic scientific correctness judgment",
    }


def run_mini_rag(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    backend: str,
    model: str,
    top_k: int,
    retrieval_only: bool,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run retrieval plus optional generation and write all output artifacts."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    dataset = load_rag_ready_dataset(rag_ready_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config(
        rag_ready_dir=rag_ready_dir,
        output_dir=output_dir,
        backend=backend,
        model=model,
        top_k=top_k,
        retrieval_only=retrieval_only,
        base_url=base_url,
        temperature=temperature,
    )
    write_json(output_dir / "rag_run_config.json", run_config)

    retrieved_contexts = retrieve_contexts(
        chunks=dataset["chunks"],
        eval_questions=dataset["eval_questions"],
        top_k=top_k,
    )
    write_jsonl(output_dir / "retrieved_contexts.jsonl", retrieved_contexts)

    answers = generate_answer_records(
        retrieved_contexts=retrieved_contexts,
        backend=backend,
        model=model,
        top_k=top_k,
        retrieval_only=retrieval_only,
        base_url=base_url,
        temperature=temperature,
    )
    write_jsonl(output_dir / "rag_answers.jsonl", answers)

    eval_summary = build_eval_summary(
        retrieved_contexts=retrieved_contexts,
        answers=answers,
        top_k=top_k,
        retrieval_only=retrieval_only,
    )
    write_json(output_dir / "rag_eval_summary.json", eval_summary)

    (output_dir / "rag_error_analysis.md").write_text(
        build_error_analysis_markdown(
            answers=answers,
            retrieved_contexts=retrieved_contexts,
        ),
        encoding="utf-8",
    )
    (output_dir / "rag_manual_review_checklist.md").write_text(
        build_manual_review_checklist(
            retrieved_contexts=retrieved_contexts,
            answers=answers,
        ),
        encoding="utf-8",
    )

    manifest = build_rag_manifest(
        rag_ready_dir=rag_ready_dir,
        output_dir=output_dir,
        dataset_counts=dataset["counts"],
        backend=backend,
        model=model,
        top_k=top_k,
        retrieval_only=retrieval_only,
    )
    write_json(output_dir / "rag_manifest.json", manifest)
    return {
        "dataset": dataset,
        "retrieved_contexts": retrieved_contexts,
        "answers": answers,
        "eval_summary": eval_summary,
        "manifest": manifest,
    }
