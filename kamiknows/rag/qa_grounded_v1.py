"""Phase 2 QA-grounded RAG v1 orchestration.

This module runs deterministic TF-IDF retrieval over ``chunks_v1`` and then
optionally asks local Ollama models to produce citation-constrained answers.
Generated citations are validated only against deterministic retrieval labels
such as ``C1``/``[C1]``.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

from kamiknows.rag.load_dataset import RagDatasetError, read_json, read_jsonl
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1
from kamiknows.run_metadata import utc_now_iso

DEFAULT_MODEL_NAMES = ("qwen3:0.6b", "qwen3:4b-instruct", "mistral:7b")
DEFAULT_TOP_K = 5
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_HARDENED_EVAL_QUESTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data/eval/rag_eval_set_v1_hardened.jsonl"
)

QA_OUTPUT_FILES = {
    "qa_model_registry": "qa_model_registry_v1.json",
    "rag_eval_set": "rag_eval_set_v1.jsonl",
    "rag_eval_set_card": "rag_eval_set_card_v1.md",
    "retrieved_contexts": "retrieved_contexts_v1.jsonl",
    "citation_validation_summary": "citation_validation_summary_v1.json",
    "citation_audit": "citation_audit_v1.jsonl",
    "rag_eval_summary": "rag_eval_summary_v1.json",
    "rag_model_comparison": "rag_model_comparison_v1.md",
    "rag_error_analysis": "rag_error_analysis_v1.md",
    "qa_run_config": "qa_run_config_v1.json",
    "qa_manifest": "qa_manifest_v1.json",
}

VALID_ANSWER_STATUSES = {"answered", "insufficient_evidence", "backend_error"}
VALID_CONFIDENCE = {"low", "medium", "high"}
LABEL_RE = re.compile(r"\[?(C[1-9][0-9]*)\]?", flags=re.IGNORECASE)


def load_rag_ready_v1_dataset(
    rag_ready_dir: Path,
    *,
    include_eval_questions_v0: bool = True,
) -> dict[str, Any]:
    """Load the Phase 2 v1 RAG-ready dataset files."""
    rag_ready_dir = Path(rag_ready_dir)
    if not rag_ready_dir.exists():
        raise RagDatasetError(f"RAG-ready v1 directory does not exist: {rag_ready_dir}")

    paths = {
        "chunks": rag_ready_dir / "chunks_v1.jsonl",
        "manifest": rag_ready_dir / "rag_manifest_v1.json",
    }
    if include_eval_questions_v0:
        paths["eval_questions"] = rag_ready_dir / "eval_questions_v0.jsonl"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RagDatasetError("missing RAG-ready v1 file(s): " + ", ".join(missing))

    chunks = read_jsonl(paths["chunks"])
    eval_questions = read_jsonl(paths["eval_questions"]) if include_eval_questions_v0 else []
    manifest = read_json(paths["manifest"])
    return {
        "rag_ready_dir": str(rag_ready_dir),
        "paths": {key: str(path) for key, path in paths.items()},
        "chunks": chunks,
        "eval_questions_v0": eval_questions,
        "manifest": manifest,
        "counts": {
            "papers": int(manifest.get("papers_count") or 0),
            "chunks": len(chunks),
            "equations": int(manifest.get("equations_count") or 0),
            "eval_questions_v0": len(eval_questions),
        },
    }


def model_slug(model_name: str) -> str:
    """Return a stable filesystem slug for an Ollama model name."""
    return (
        model_name.lower()
        .replace(":", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def build_default_model_registry(
    *,
    model_names: list[str] | None = None,
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_predict: int = 512,
) -> dict[str, Any]:
    """Build the generator registry requested for QA v1."""
    names = model_names or list(DEFAULT_MODEL_NAMES)
    models = []
    for name in names:
        models.append(
            {
                "model_name": name,
                "model_slug": model_slug(name),
                "backend": "ollama",
                "role": "rag_grounded_qa_generator",
                "enabled": True,
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
                "max_tokens": num_predict,
                "notes": (
                    "Local Ollama generator. The run marks this model as model_unavailable "
                    "if it is not present in the local Ollama tag list."
                ),
            }
        )
    return {
        "qa_model_registry_version": "qa_model_registry_v1",
        "created_at": utc_now_iso(),
        "models": models,
        "scope": "local RAG QA generation only; no training, no discovery, no scientific reliability claim",
    }


def build_eval_set_v1(
    eval_questions_v0: list[dict[str, Any]],
    *,
    eval_questions_file: Path | str | None = None,
    static_v1_questions_enabled: bool = False,
    llm_question_generation_enabled: bool = False,
    static_v1_questions_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Build the QA eval set from a canonical file or legacy v0 inputs.

    LLM-based question generation is intentionally not implemented for P2T4.
    When ``eval_questions_file`` is provided, only that JSONL file is used.
    The legacy v0-plus-static composition path is retained for explicit
    compatibility only and requires a static question file.
    """
    if llm_question_generation_enabled:
        raise ValueError("LLM question generation is not implemented for QA grounded v1.")
    if eval_questions_file is not None:
        return load_eval_questions_from_file(eval_questions_file)
    records = [_normalize_eval_question_v0(question) for question in eval_questions_v0]
    if static_v1_questions_enabled:
        if static_v1_questions_path is None:
            raise ValueError(
                "static_v1_questions_path is required when static_v1_questions_enabled is true; "
                "no operational eval questions are hard-coded in QA grounded v1."
            )
        records.extend(load_static_eval_questions_v1(static_v1_questions_path))
    return records


def load_eval_questions_from_file(path: Path | str) -> list[dict[str, Any]]:
    """Load and validate a JSONL eval-question file."""
    questions_path = Path(path)
    if not questions_path.exists():
        raise RagDatasetError(f"eval questions file does not exist: {questions_path}")

    records = read_jsonl(questions_path)
    if not records:
        raise RagDatasetError(f"eval questions file is empty: {questions_path}")

    required_fields = ("question_id", "question", "question_type", "expected_behavior")
    seen_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        for field in required_fields:
            if not str(record.get(field) or "").strip():
                raise RagDatasetError(
                    f"{questions_path}:{index} missing required eval-question field: {field}"
                )
        question_id = str(record["question_id"])
        if question_id in seen_ids:
            raise RagDatasetError(
                f"{questions_path}:{index} duplicate question_id: {question_id}"
            )
        seen_ids.add(question_id)
    return records


def load_static_eval_questions_v1(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load legacy static extension questions from an explicit data file."""
    if path is None:
        raise ValueError("static_v1_questions_path is required for legacy static eval loading")
    return load_eval_questions_from_file(path)


def build_eval_set_card_markdown(eval_set: list[dict[str, Any]]) -> str:
    """Build a concise dataset card for the QA eval set."""
    type_counts = Counter(str(question.get("question_type") or "unknown") for question in eval_set)
    behavior_counts = Counter(str(question.get("expected_behavior") or "unknown") for question in eval_set)
    lines = [
        "# KamiKnows RAG Eval Set v1",
        "",
        "This lightweight eval set is intended for grounded QA pipeline validation over the fixed chunks_v1 corpus.",
        "It is not a scientific benchmark and does not prove answer reliability.",
        "",
        "## Counts",
        "",
        f"- Total questions: {len(eval_set)}",
        "- Question types: " + json.dumps(dict(sorted(type_counts.items())), sort_keys=True),
        "- Expected behavior: " + json.dumps(dict(sorted(behavior_counts.items())), sort_keys=True),
        "",
        "## Questions",
        "",
    ]
    for question in eval_set:
        expected_ids = question.get("expected_arxiv_ids") or question.get("expected_paper_ids") or []
        lines.append(
            "- `{qid}` / `{qtype}` / `{behavior}` / expected `{expected}`: {text}".format(
                qid=question.get("question_id", ""),
                qtype=question.get("question_type", ""),
                behavior=question.get("expected_behavior", ""),
                expected=", ".join(expected_ids) if expected_ids else "<none>",
                text=question.get("question", ""),
            )
        )
    lines.extend(
        [
            "",
            "Manual review remains required for answer quality and scientific interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def retrieve_contexts_v1(
    *,
    chunks: list[dict[str, Any]],
    eval_set: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    source_policy: str = "none",
) -> list[dict[str, Any]]:
    """Retrieve deterministic contexts with C1/C2/... citation labels."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    retriever = TfidfRetrieverV1(chunks)
    from kamiknows.rag.source_constraints import plan_source_constraints, select_balanced_results
    records = []
    for question in eval_set:
        question_text = str(question.get("question") or "")
        question_type = str(question.get("question_type") or "")
        expected_keywords = [str(value) for value in question.get("expected_section_keywords", []) or []]
        indices, ids, diagnostics = plan_source_constraints(chunks, question_text, top_k, source_policy)
        retrieved = [
            build_retrieved_qa_chunk(chunk=item.chunk, rank=item.rank, score=item.score)
            for item in retriever.retrieve(
                question_text,
                top_k=max(1, len(indices)) if ids else top_k,
                question_type=question_type,
                expected_section_keywords=expected_keywords,
                candidate_indices=indices,
            )
        ]
        retrieved = select_balanced_results(retrieved, ids, top_k) if indices else []
        records.append(
            {
                "question_id": question.get("question_id", ""),
                "question": question_text,
                "question_type": question_type,
                "expected_behavior": question.get("expected_behavior", "answerable"),
                "expected_arxiv_ids": list(question.get("expected_arxiv_ids", []) or []),
                "expected_paper_ids": list(question.get("expected_paper_ids", []) or []),
                "top_k": top_k,
                "retriever": {
                    "name": TfidfRetrieverV1.name,
                    "version": TfidfRetrieverV1.version,
                },
                "retrieved_chunks": retrieved,
                "source_constraint": diagnostics,
                "retrieval_eval": evaluate_retrieval_for_question(
                    question=question,
                    retrieved_chunks=retrieved,
                ),
            }
        )
    return records


def build_retrieved_qa_chunk(
    *,
    chunk: dict[str, Any],
    rank: int,
    score: float,
) -> dict[str, Any]:
    """Build one retrieved chunk with deterministic citation metadata."""
    text = " ".join(str(chunk.get("text") or "").split())
    return {
        "citation_label": f"[C{rank}]",
        "citation_id": f"C{rank}",
        "rank": rank,
        "retrieved_rank": rank,
        "retrieval_score": round(float(score), 8),
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


def evaluate_retrieval_for_question(
    *,
    question: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate retrieved chunks against expected arXiv or paper IDs."""
    expected_arxiv_ids = [
        str(value)
        for value in _as_list(question.get("expected_arxiv_ids") or question.get("expected_source_arxiv_ids"))
        if value
    ]
    expected_paper_ids = [
        str(value)
        for value in _as_list(question.get("expected_paper_ids") or question.get("expected_paper_id"))
        if value
    ]
    has_expected = bool(expected_arxiv_ids or expected_paper_ids)
    if not has_expected:
        return {
            "has_expected_source": False,
            "expected_source_hit": False,
            "expected_source_hit_rank": None,
            "notes": "No expected source identifier was specified.",
        }

    required_all = bool(question.get("required_all_expected_sources"))
    matched_identifiers: list[str] = []
    hit_ranks: list[int] = []
    for chunk in retrieved_chunks:
        chunk_arxiv_id = str(chunk.get("arxiv_id") or "")
        chunk_paper_id = str(chunk.get("paper_id") or "")
        matched = []
        if chunk_arxiv_id in expected_arxiv_ids:
            matched.append(chunk_arxiv_id)
        if chunk_paper_id in expected_paper_ids:
            matched.append(chunk_paper_id)
        if matched:
            matched_identifiers.extend(matched)
            hit_ranks.append(int(chunk.get("rank") or 0) or 0)
        if matched and not required_all:
            return {
                "has_expected_source": True,
                "expected_source_hit": True,
                "expected_source_hit_rank": int(chunk.get("rank") or 0) or None,
                "required_all_expected_sources": False,
                "matched_expected_source_ids": _unique_preserve_order(matched_identifiers),
                "missing_expected_source_ids": [],
                "notes": "At least one expected source identifier was retrieved.",
            }

    if required_all:
        expected_groups = [
            [arxiv_id, str(arxiv_id).replace(".", "_")]
            for arxiv_id in expected_arxiv_ids
        ]
        if not expected_groups:
            expected_groups = [[paper_id] for paper_id in expected_paper_ids]
        matched_set = set(matched_identifiers)
        missing_groups = [
            group[0]
            for group in expected_groups
            if not any(identifier in matched_set for identifier in group)
        ]
        if not missing_groups:
            return {
                "has_expected_source": True,
                "expected_source_hit": True,
                "expected_source_hit_rank": max(hit_ranks) if hit_ranks else None,
                "required_all_expected_sources": True,
                "matched_expected_source_ids": _unique_preserve_order(matched_identifiers),
                "missing_expected_source_ids": [],
                "notes": "All required expected source identifiers were retrieved.",
            }
        return {
            "has_expected_source": True,
            "expected_source_hit": False,
            "expected_source_hit_rank": None,
            "required_all_expected_sources": True,
            "matched_expected_source_ids": _unique_preserve_order(matched_identifiers),
            "missing_expected_source_ids": missing_groups,
            "notes": "Not all required expected source identifiers were retrieved.",
        }

    return {
        "has_expected_source": True,
        "expected_source_hit": False,
        "expected_source_hit_rank": None,
        "required_all_expected_sources": False,
        "matched_expected_source_ids": [],
        "missing_expected_source_ids": _unique_preserve_order(expected_arxiv_ids or expected_paper_ids),
        "notes": "No retrieved chunk matched the expected source identifiers.",
    }


def build_grounded_qa_prompt_v1(context_record: dict[str, Any], prompt_profile: str = "grounded_json_v1") -> str:
    """Build the strict grounded prompt for one retrieved context."""
    if prompt_profile not in {"grounded_json_v1", "synthesis_v2"}:
        raise ValueError(f"Unknown QA prompt profile: {prompt_profile}")
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
                    f"source_type: {chunk.get('source_type', '')}",
                    f"quality_flags: {json.dumps(chunk.get('quality_flags', []), sort_keys=True)}",
                    "text:",
                    str(chunk.get("text") or ""),
                ]
            )
        )
    context_text = "\n\n".join(blocks) if blocks else "<NO_CONTEXT>"
    constraint = context_record.get("source_constraint", {})
    source_instruction = ""
    if constraint.get("source_policy") == "explicit_arxiv" and constraint.get("requested_arxiv_ids"):
        source_instruction = (
            "\nRequested sources: " + ", ".join(constraint["requested_arxiv_ids"]) +
            ". Attribute each claim to its source; comparisons must distinguish each paper.\n"
        )
        if constraint.get("status") != "available":
            source_instruction += "Requested evidence is unavailable or source budget is insufficient. Set answer_status to insufficient_evidence.\n"
    prompt = f"""You are KamiKnows, a citation-constrained scientific QA assistant.

Use only the retrieved context below. Do not use outside knowledge.
Use only citation labels that appear in the context: [C1], [C2], ...
Do not invent citation labels, chunk IDs, arXiv IDs, titles, or sources.
If the context does not support the answer, set answer_status to "insufficient_evidence".
Every factual claim in answer must include one or more labels like [C1].
Keep the answer concise and technical.
{source_instruction}

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
    if prompt_profile == "synthesis_v2":
        from kamiknows.rag.prompt_profiles import SYNTHESIS_INSTRUCTIONS_V2
        marker = "Return only valid JSON with this exact shape:"
        # Cambia solo le istruzioni: schema, domanda e contesto restano identici.
        prompt = SYNTHESIS_INSTRUCTIONS_V2 + "\n" + source_instruction + "\n" + marker + prompt.split(marker, 1)[1]
    return prompt


def list_local_ollama_models(base_url: str = DEFAULT_BASE_URL, timeout_seconds: int = 5) -> dict[str, Any]:
    """Return local Ollama tag names or a backend error."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {
            "available": False,
            "model_names": [],
            "backend_error": f"Ollama tag request failed: {exc}",
        }
    data = response.json()
    names = sorted(
        str(model.get("name") or "")
        for model in data.get("models", [])
        if isinstance(model, dict) and model.get("name")
    )
    return {
        "available": True,
        "model_names": names,
        "backend_error": "",
    }


def annotate_registry_availability(
    *,
    registry: dict[str, Any],
    base_url: str,
) -> dict[str, Any]:
    """Annotate registry models with local Ollama availability."""
    tags = list_local_ollama_models(base_url)
    model_names = set(tags.get("model_names", []))
    annotated = json.loads(json.dumps(registry))
    annotated["ollama"] = {
        "base_url": base_url,
        "tags_available": bool(tags.get("available")),
        "available_model_names": sorted(model_names),
        "backend_error": tags.get("backend_error", ""),
    }
    for model in annotated["models"]:
        if not model.get("enabled", True):
            model["availability_status"] = "disabled"
        elif not tags.get("available"):
            model["availability_status"] = "backend_error"
            model["availability_notes"] = tags.get("backend_error", "")
        elif model["model_name"] in model_names:
            model["availability_status"] = "available"
            model["availability_notes"] = "Model is present in local Ollama tags."
        else:
            model["availability_status"] = "model_unavailable"
            model["availability_notes"] = "Model is not present in local Ollama tags."
    return annotated


def generate_model_answer_records(
    *,
    model_config: dict[str, Any],
    retrieved_contexts: list[dict[str, Any]],
    known_chunk_ids: list[str],
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: int = 120,
) -> list[dict[str, Any]]:
    """Generate answer records for one model, tolerating unavailable backends."""
    availability = str(model_config.get("availability_status") or "")
    if availability in {"backend_error", "model_unavailable", "disabled"}:
        return [
            build_backend_error_answer_record(
                context_record=context,
                model_config=model_config,
                top_k=int(context.get("top_k") or DEFAULT_TOP_K),
                error_type=availability or "backend_error",
                error_message=str(model_config.get("availability_notes") or "Model is not available."),
                known_chunk_ids=known_chunk_ids,
            )
            for context in retrieved_contexts
        ]

    records = []
    for context in retrieved_contexts:
        prompt = build_grounded_qa_prompt_v1(context, model_config.get("prompt_profile", "grounded_json_v1"))
        try:
            raw_response = generate_ollama_json(
                prompt=prompt,
                model_config=model_config,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            records.append(
                build_backend_error_answer_record(
                    context_record=context,
                    model_config=model_config,
                    top_k=int(context.get("top_k") or DEFAULT_TOP_K),
                    error_type="backend_error",
                    error_message=str(exc),
                    known_chunk_ids=known_chunk_ids,
                )
            )
            continue

        parsed = parse_model_json_response(raw_response)
        validation = validate_answer_citations(
            parsed_answer=parsed,
            retrieved_chunks=list(context.get("retrieved_chunks", []) or []),
            known_chunk_ids=known_chunk_ids,
        )
        records.append(
            build_answer_record_from_parsed(
                context_record=context,
                model_config=model_config,
                raw_response=raw_response,
                parsed=parsed,
                citation_validation=validation,
            )
        )
    return records


def generate_ollama_json(
    *,
    prompt: str,
    model_config: dict[str, Any],
    base_url: str,
    timeout_seconds: int,
) -> str:
    """Call Ollama's generate endpoint with deterministic QA options."""
    text = _ollama_generate(
        prompt=prompt,
        model_config=model_config,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        request_json_format=True,
    )
    if text:
        return text
    text = _ollama_generate(
        prompt=prompt,
        model_config=model_config,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        request_json_format=False,
    )
    if text:
        return text
    raise RuntimeError("Ollama returned an empty or invalid response.")


def _ollama_generate(
    *,
    prompt: str,
    model_config: dict[str, Any],
    base_url: str,
    timeout_seconds: int,
    request_json_format: bool,
) -> str:
    """Call Ollama once and return the raw text response."""
    payload = {
        "model": model_config["model_name"],
        "prompt": prompt.strip(),
        "stream": False,
        "options": {
            "temperature": float(model_config.get("temperature", 0.0)),
            "top_p": float(model_config.get("top_p", 0.9)),
            "num_predict": int(model_config.get("num_predict") or model_config.get("max_tokens") or 512),
        },
    }
    if request_json_format:
        payload["format"] = "json"
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    text = data.get("response")
    if not isinstance(text, str):
        return ""
    return text.strip()


def parse_model_json_response(raw_response: str) -> dict[str, Any]:
    """Parse model JSON, with a conservative fallback for malformed output."""
    warnings = []
    text = raw_response.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    parsed: dict[str, Any] | None = None
    try:
        parsed_data = json.loads(text)
        if isinstance(parsed_data, dict):
            parsed = parsed_data
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(text)
        if extracted:
            try:
                parsed_data = json.loads(extracted)
                if isinstance(parsed_data, dict):
                    parsed = parsed_data
                    warnings.append("json_extracted_from_response")
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        warnings.append("json_parse_error")
        labels = sorted(set(extract_citation_labels(text)))
        parsed = {
            "answer_status": "answered" if text else "insufficient_evidence",
            "answer": text,
            "evidence": "",
            "citations": labels,
            "confidence": "low",
            "limitations": "Response was not valid JSON; parsed conservatively.",
            "unsupported_or_inferred": [],
        }

    status = str(parsed.get("answer_status") or "").strip()
    if status not in VALID_ANSWER_STATUSES:
        warnings.append("invalid_answer_status_normalized")
        status = "insufficient_evidence" if "insufficient" in status.lower() else "answered"

    confidence = str(parsed.get("confidence") or "low").strip().lower()
    if confidence not in VALID_CONFIDENCE:
        warnings.append("invalid_confidence_normalized")
        confidence = "low"

    citations = normalize_citation_list(parsed.get("citations", []))
    return {
        "answer_status": status,
        "answer": str(parsed.get("answer") or ""),
        "evidence": str(parsed.get("evidence") or ""),
        "citations": citations,
        "confidence": confidence,
        "limitations": str(parsed.get("limitations") or ""),
        "unsupported_or_inferred": _as_list(parsed.get("unsupported_or_inferred")),
        "parse_warnings": warnings,
        "parsed_json": parsed,
    }


def normalize_citation_list(citations: Any) -> list[str]:
    """Normalize model citation values to C1/C2 labels."""
    labels = []
    for value in _as_list(citations):
        if isinstance(value, dict):
            value = value.get("citation_label") or value.get("label") or value.get("citation")
        text = str(value or "")
        labels.extend(extract_citation_labels(text))
    return _dedupe(labels)


def extract_citation_labels(text: str) -> list[str]:
    """Extract C-number citation labels from text."""
    labels = []
    for match in LABEL_RE.finditer(text or ""):
        labels.append(match.group(1).upper())
    return labels


def validate_answer_citations(
    *,
    parsed_answer: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    known_chunk_ids: list[str],
) -> dict[str, Any]:
    """Validate generated citation labels against retrieved chunks."""
    label_to_chunk = {
        str(chunk.get("citation_id") or "").upper(): str(chunk.get("chunk_id") or "")
        for chunk in retrieved_chunks
        if chunk.get("citation_id") and chunk.get("chunk_id")
    }
    valid_labels = set(label_to_chunk)
    used_labels = set(parsed_answer.get("citations", []) or [])
    for key in ("answer", "evidence", "limitations"):
        used_labels.update(extract_citation_labels(str(parsed_answer.get(key) or "")))
    used_labels = {label.upper() for label in used_labels if label}
    status = str(parsed_answer.get("answer_status") or "answered")
    if status == "backend_error":
        return {
            "status": "not_applicable_backend_error",
            "valid": True,
            "answer_status": status,
            "valid_labels": sorted(valid_labels),
            "used_citation_labels": sorted(used_labels),
            "valid_used_citation_labels": [],
            "invented_citation_labels": [],
            "cited_chunk_ids": [],
            "unknown_citation_chunk_ids": [],
            "answered_without_valid_citation": False,
            "notes": "Citation validation is not applicable because generation failed or was unavailable.",
        }
    invented = sorted(label for label in used_labels if label not in valid_labels)
    valid_used = sorted(label for label in used_labels if label in valid_labels)
    cited_chunk_ids = [label_to_chunk[label] for label in valid_used]
    unknown_chunk_ids = sorted(chunk_id for chunk_id in cited_chunk_ids if chunk_id not in set(known_chunk_ids))
    answered_without_valid = status == "answered" and not valid_used
    invalid = bool(invented or unknown_chunk_ids or answered_without_valid)
    if invented:
        validation_status = "invented_citation"
    elif unknown_chunk_ids:
        validation_status = "unknown_chunk_id"
    elif answered_without_valid:
        validation_status = "answered_without_valid_citation"
    elif status == "insufficient_evidence" and not valid_used:
        validation_status = "insufficient_evidence_no_citation"
    else:
        validation_status = "valid"
    return {
        "status": validation_status,
        "valid": not invalid,
        "answer_status": status,
        "valid_labels": sorted(valid_labels),
        "used_citation_labels": sorted(used_labels),
        "valid_used_citation_labels": valid_used,
        "invented_citation_labels": invented,
        "cited_chunk_ids": cited_chunk_ids,
        "unknown_citation_chunk_ids": unknown_chunk_ids,
        "answered_without_valid_citation": answered_without_valid,
        "notes": _citation_validation_note(validation_status),
    }


def build_answer_record_from_parsed(
    *,
    context_record: dict[str, Any],
    model_config: dict[str, Any],
    raw_response: str,
    parsed: dict[str, Any],
    citation_validation: dict[str, Any],
) -> dict[str, Any]:
    """Build one structured QA answer record."""
    warnings = list(parsed.get("parse_warnings", []) or [])
    if not citation_validation.get("valid", False):
        warnings.append(str(citation_validation.get("status") or "citation_validation_failed"))
    answer_status = str(parsed.get("answer_status") or "answered")
    return {
        "question_id": context_record.get("question_id", ""),
        "question": context_record.get("question", ""),
        "question_type": context_record.get("question_type", ""),
        "expected_behavior": context_record.get("expected_behavior", ""),
        "model_name": model_config.get("model_name", ""),
        "model_slug": model_config.get("model_slug", model_slug(str(model_config.get("model_name") or ""))),
        "backend": model_config.get("backend", "ollama"),
        "retriever": context_record.get("retriever", {}).get("name", TfidfRetrieverV1.name),
        "top_k": context_record.get("top_k", DEFAULT_TOP_K),
        "answer_status": answer_status,
        "answer": parsed.get("answer", ""),
        "evidence": parsed.get("evidence", ""),
        "citations": list(parsed.get("citations", []) or []),
        "confidence": parsed.get("confidence", "low"),
        "limitations": parsed.get("limitations", ""),
        "unsupported_or_inferred": list(parsed.get("unsupported_or_inferred", []) or []),
        "raw_model_response": raw_response,
        "retrieved_chunks": context_record.get("retrieved_chunks", []),
        "retrieval_eval": context_record.get("retrieval_eval", {}),
        "citation_validation": citation_validation,
        "warnings": _dedupe(warnings),
        "backend_error": None,
        "grounding_status": "not_reviewed",
        "prompt_profile": model_config.get("prompt_profile", "grounded_json_v1"),
    }


def build_backend_error_answer_record(
    *,
    context_record: dict[str, Any],
    model_config: dict[str, Any],
    top_k: int,
    error_type: str,
    error_message: str,
    known_chunk_ids: list[str],
) -> dict[str, Any]:
    """Build an answer record when generation is not available."""
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
        retrieved_chunks=list(context_record.get("retrieved_chunks", []) or []),
        known_chunk_ids=known_chunk_ids,
    )
    record = build_answer_record_from_parsed(
        context_record=context_record,
        model_config=model_config,
        raw_response="",
        parsed=parsed,
        citation_validation=validation,
    )
    record["top_k"] = top_k
    record["warnings"] = _dedupe(list(record.get("warnings", []) or []) + [error_type])
    record["backend_error"] = {
        "error_type": error_type,
        "message": error_message,
    }
    return record


def build_citation_audit_records(answer_records_by_model: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build one citation audit record per model-question answer."""
    audit = []
    for model_name, records in answer_records_by_model.items():
        for record in records:
            validation = record.get("citation_validation", {})
            audit.append(
                {
                    "model_name": model_name,
                    "question_id": record.get("question_id", ""),
                    "answer_status": record.get("answer_status", ""),
                    "citation_validation_status": validation.get("status", ""),
                    "citation_validation_valid": validation.get("valid", False),
                    "used_citation_labels": validation.get("used_citation_labels", []),
                    "valid_used_citation_labels": validation.get("valid_used_citation_labels", []),
                    "invented_citation_labels": validation.get("invented_citation_labels", []),
                    "cited_chunk_ids": validation.get("cited_chunk_ids", []),
                    "retrieved_chunk_ids": [
                        chunk.get("chunk_id", "")
                        for chunk in record.get("retrieved_chunks", []) or []
                    ],
                    "answered_without_valid_citation": validation.get(
                        "answered_without_valid_citation",
                        False,
                    ),
                    "warnings": record.get("warnings", []),
                    "grounding_status": record.get("grounding_status", "not_reviewed"),
                }
            )
    return audit


def build_citation_validation_summary(audit_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize generated citation validation."""
    status_counts = Counter(str(record.get("citation_validation_status") or "unknown") for record in audit_records)
    invented_count = sum(len(record.get("invented_citation_labels", []) or []) for record in audit_records)
    answered_without_valid = sum(
        1 for record in audit_records if record.get("answered_without_valid_citation")
    )
    invalid_records = [
        {
            "model_name": record.get("model_name", ""),
            "question_id": record.get("question_id", ""),
            "status": record.get("citation_validation_status", ""),
            "invented_citation_labels": record.get("invented_citation_labels", []),
            "answered_without_valid_citation": record.get("answered_without_valid_citation", False),
        }
        for record in audit_records
        if not record.get("citation_validation_valid", False)
    ]
    return {
        "citation_validation_summary_version": "citation_validation_summary_v1",
        "created_at": utc_now_iso(),
        "total_answer_records": len(audit_records),
        "status_counts": dict(sorted(status_counts.items())),
        "invalid_citation_count": len(invalid_records),
        "invented_citation_count": invented_count,
        "answered_without_valid_citation_count": answered_without_valid,
        "invalid_records": invalid_records,
        "notes": (
            "Only deterministic labels associated with retrieved chunks are accepted. "
            "This validation does not prove factual or scientific correctness."
        ),
    }


def build_eval_summary_v1(
    *,
    eval_set: list[dict[str, Any]],
    retrieved_contexts: list[dict[str, Any]],
    registry: dict[str, Any],
    answer_records_by_model: dict[str, list[dict[str, Any]]],
    citation_summary: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Build the comparative QA summary."""
    all_records = [record for records in answer_records_by_model.values() for record in records]
    available_models = [
        model["model_name"]
        for model in registry.get("models", [])
        if model.get("availability_status") == "available"
    ]
    model_summaries = [
        summarize_model_answers(model_config=model, records=answer_records_by_model.get(model["model_name"], []))
        for model in registry.get("models", [])
    ]
    warnings_by_type = Counter()
    for record in all_records:
        warnings_by_type.update(str(warning) for warning in record.get("warnings", []) or [])
    retrieval_expected = [
        context
        for context in retrieved_contexts
        if context.get("retrieval_eval", {}).get("has_expected_source")
    ]
    retrieval_hits = [
        context
        for context in retrieval_expected
        if context.get("retrieval_eval", {}).get("expected_source_hit")
    ]
    answer_lengths = [len(str(record.get("answer") or "")) for record in all_records if record.get("answer_status") == "answered"]
    return {
        "rag_eval_summary_version": "rag_eval_summary_v1",
        "created_at": utc_now_iso(),
        "questions_total": len(eval_set),
        "top_k": top_k,
        "retriever": {
            "name": TfidfRetrieverV1.name,
            "version": TfidfRetrieverV1.version,
        },
        "models_attempted": [model.get("model_name", "") for model in registry.get("models", [])],
        "models_available": available_models,
        "models_unavailable": [
            {
                "model_name": model.get("model_name", ""),
                "availability_status": model.get("availability_status", ""),
                "availability_notes": model.get("availability_notes", ""),
            }
            for model in registry.get("models", [])
            if model.get("availability_status") != "available"
        ],
        "answers_attempted": len(all_records),
        "answers_succeeded": sum(1 for record in all_records if record.get("answer_status") == "answered"),
        "insufficient_evidence_count": sum(
            1 for record in all_records if record.get("answer_status") == "insufficient_evidence"
        ),
        "backend_errors": sum(1 for record in all_records if record.get("answer_status") == "backend_error"),
        "model_unavailable_records": sum(
            1
            for record in all_records
            if (record.get("backend_error") or {}).get("error_type") == "model_unavailable"
        ),
        "invalid_citation_count": citation_summary["invalid_citation_count"],
        "invented_citation_count": citation_summary["invented_citation_count"],
        "answered_without_valid_citation_count": citation_summary[
            "answered_without_valid_citation_count"
        ],
        "average_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
        "expected_source_hit_rate": len(retrieval_hits) / len(retrieval_expected) if retrieval_expected else 0.0,
        "expected_source_hits": len(retrieval_hits),
        "expected_source_questions": len(retrieval_expected),
        "retrieval_miss_question_ids": [
            context["question_id"]
            for context in retrieval_expected
            if not context.get("retrieval_eval", {}).get("expected_source_hit")
        ],
        "warnings_by_type": dict(sorted(warnings_by_type.items())),
        "per_model": model_summaries,
        "manual_review": {
            "required": True,
            "status": "pending",
        },
        "scope": (
            "QA grounded v1 tests citation-constrained answer generation, "
            "but does not prove scientific reliability."
        ),
    }


def summarize_model_answers(
    *,
    model_config: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize answer records for one model."""
    warnings_by_type = Counter()
    for record in records:
        warnings_by_type.update(str(warning) for warning in record.get("warnings", []) or [])
    answer_lengths = [len(str(record.get("answer") or "")) for record in records if record.get("answer_status") == "answered"]
    return {
        "model_name": model_config.get("model_name", ""),
        "availability_status": model_config.get("availability_status", ""),
        "answers_total": len(records),
        "answers_succeeded": sum(1 for record in records if record.get("answer_status") == "answered"),
        "insufficient_evidence_count": sum(
            1 for record in records if record.get("answer_status") == "insufficient_evidence"
        ),
        "backend_errors": sum(1 for record in records if record.get("answer_status") == "backend_error"),
        "invalid_citation_count": sum(
            1 for record in records if not record.get("citation_validation", {}).get("valid", False)
        ),
        "invented_citation_count": sum(
            len(record.get("citation_validation", {}).get("invented_citation_labels", []) or [])
            for record in records
        ),
        "answered_without_valid_citation_count": sum(
            1
            for record in records
            if record.get("citation_validation", {}).get("answered_without_valid_citation")
        ),
        "average_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
        "warnings_by_type": dict(sorted(warnings_by_type.items())),
    }


def build_model_comparison_markdown(
    *,
    eval_summary: dict[str, Any],
) -> str:
    """Build the model comparison Markdown report."""
    lines = [
        "# KamiKnows RAG Model Comparison v1",
        "",
        "QA grounded v1 tests citation-constrained answer generation, but does not prove scientific reliability.",
        "",
        "## Summary",
        "",
        f"- Questions: {eval_summary['questions_total']}",
        f"- Top-k: {eval_summary['top_k']}",
        f"- Retriever expected-source hit rate: {eval_summary['expected_source_hit_rate']:.3f}",
        f"- Models available: {', '.join(eval_summary['models_available']) or '<none>'}",
        "",
        "## Per-Model Metrics",
        "",
        "| Model | Availability | Answered | Insufficient | Backend errors | Invalid citations | Invented labels | Answered without valid citation | Avg answer chars |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in eval_summary["per_model"]:
        lines.append(
            "| {model_name} | {availability_status} | {answers_succeeded} | {insufficient_evidence_count} | {backend_errors} | {invalid_citation_count} | {invented_citation_count} | {answered_without_valid_citation_count} | {average_answer_length:.1f} |".format(
                **model
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `backend_error` and `model_unavailable` records mean the run completed without treating missing local models as fatal.",
            "- Citation validation only checks generated labels against retrieved chunks.",
            "- Manual review remains required for grounding and scientific interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_error_analysis_markdown_v1(
    *,
    retrieved_contexts: list[dict[str, Any]],
    answer_records_by_model: dict[str, list[dict[str, Any]]],
) -> str:
    """Build the requested QA error-analysis Markdown."""
    all_records = [record for records in answer_records_by_model.values() for record in records]
    categorized: dict[str, list[str]] = {category: [] for category in _error_categories()}
    retrieval_by_id = {context["question_id"]: context for context in retrieved_contexts}
    for context in retrieved_contexts:
        question_id = str(context.get("question_id") or "")
        if context.get("retrieval_eval", {}).get("has_expected_source") and not context.get("retrieval_eval", {}).get(
            "expected_source_hit"
        ):
            categorized["retrieval_miss"].append(question_id)
        if _is_ambiguous_question(context):
            categorized["ambiguous_question"].append(question_id)

    for record in all_records:
        qid = str(record.get("question_id") or "")
        validation = record.get("citation_validation", {})
        if record.get("answer_status") == "backend_error":
            categorized["backend_error"].append(_model_question(record))
            continue
        if not validation.get("valid", False):
            categorized["answer_not_grounded"].append(_model_question(record))
        if validation.get("status") in {"invented_citation", "unknown_chunk_id", "answered_without_valid_citation"}:
            categorized["invalid_citation"].append(_model_question(record))
        if validation.get("invented_citation_labels"):
            categorized["invented_citation"].append(_model_question(record))
        expected_behavior = str(record.get("expected_behavior") or "")
        if expected_behavior == "insufficient_evidence" and record.get("answer_status") == "insufficient_evidence":
            categorized["insufficient_evidence_expected"].append(_model_question(record))
        if expected_behavior == "answerable" and record.get("answer_status") == "insufficient_evidence":
            categorized["insufficient_evidence_unexpected"].append(_model_question(record))
        if expected_behavior == "insufficient_evidence" and record.get("answer_status") == "answered":
            categorized["overclaim"].append(_model_question(record))
        if record.get("answer_status") == "answered" and not str(record.get("answer") or "").strip():
            categorized["incomplete_answer"].append(_model_question(record))
        if (
            record.get("answer_status") == "answered"
            and retrieval_by_id.get(qid, {}).get("retrieval_eval", {}).get("has_expected_source")
            and not retrieval_by_id.get(qid, {}).get("retrieval_eval", {}).get("expected_source_hit")
        ):
            categorized["incomplete_answer"].append(_model_question(record))

    lines = [
        "# KamiKnows QA Grounded v1 Error Analysis",
        "",
        "This analysis classifies retrieval, generation, and citation issues. It does not judge scientific correctness.",
        "",
    ]
    for category in _error_categories():
        items = _dedupe(categorized[category])
        lines.extend([f"## {category}", ""])
        if items:
            lines.extend(f"- `{item}`" for item in items)
        else:
            lines.append("No cases flagged.")
        lines.append("")
    lines.extend(
        [
            "## Review Notes",
            "",
            "- `retrieval_miss` is based on expected source IDs only.",
            "- `answer_not_grounded` means citation-label validation failed, not necessarily that the prose is scientifically false.",
            "- `overclaim` flags an answered response for an eval item expected to require insufficient evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def build_task_report_markdown(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    dataset: dict[str, Any],
    eval_set: list[dict[str, Any]],
    eval_questions_file: Path | str | None,
    registry: dict[str, Any],
    eval_summary: dict[str, Any],
    citation_summary: dict[str, Any],
) -> str:
    """Build the mandatory Phase 2 Task 04 report."""
    configured = [model["model_name"] for model in registry.get("models", [])]
    available = eval_summary.get("models_available", [])
    unavailable = eval_summary.get("models_unavailable", [])
    per_model = eval_summary.get("per_model", [])
    generation_backend_errors = [
        model for model in per_model if int(model.get("backend_errors") or 0) > 0
    ]
    retrieval_misses = eval_summary.get("retrieval_miss_question_ids", [])
    input_files = [
        str(Path(rag_ready_dir) / "chunks_v1.jsonl"),
        str(Path(rag_ready_dir) / "rag_manifest_v1.json"),
    ]
    if eval_questions_file is not None:
        input_files.append(str(eval_questions_file))
    else:
        input_files.append(str(Path(rag_ready_dir) / "eval_questions_v0.jsonl"))

    question_source_lines = [
        f"- Complete eval questions are loaded from `{eval_questions_file}`."
        if eval_questions_file is not None
        else "- Eval questions are built from `outputs/rag_ready_fastcalo_v1/eval_questions_v0.jsonl` plus any explicitly configured static question file.",
        "- No operational eval questions are hard-coded in the QA runtime.",
        "- No LLM generates questions in QA grounded v1.",
    ]
    lines = [
        "# Phase 2 Task 04 - QA Grounded v1",
        "",
        "Date: 2026-07-02",
        "",
        "## Objective",
        "",
        "Build a QA RAG grounded v1 loop over chunks_v1 with deterministic retrieved-chunk citations and local Ollama generator comparison.",
        "",
        "QA grounded v1 tests citation-constrained answer generation, but does not prove scientific reliability.",
        "",
        "## Inputs Used",
        "",
        "```text",
        *input_files,
        "```",
        "",
        "The existing `outputs/rag_ready_fastcalo_v0/` and `outputs/rag_ready_fastcalo_v1/` directories were not overwritten.",
        "",
        "## Chunks v1 Used",
        "",
        f"- Papers: {dataset['counts']['papers']}",
        f"- Chunks: {dataset['counts']['chunks']}",
        f"- Equations: {dataset['counts']['equations']}",
        f"- v0 eval questions imported: {dataset['counts']['eval_questions_v0']}",
        f"- Known chunk quality flags: {json.dumps(dataset['manifest'].get('quality_summary', {}), sort_keys=True)}",
        "",
        "## Models Configured",
        "",
        "- " + "\n- ".join(configured),
        "",
        "## Models Effectively Available",
        "",
        "- " + ("\n- ".join(available) if available else "<none>"),
        "",
        "Unavailable models:",
        "",
        "- "
        + (
            "\n- ".join(
                f"{item['model_name']}: {item['availability_status']} ({item.get('availability_notes', '')})"
                for item in unavailable
            )
            if unavailable
            else "<none>"
        ),
        "",
        "Models with generation backend errors:",
        "",
        "- "
        + (
            "\n- ".join(
                f"{model['model_name']}: {model['backend_errors']} backend-error records"
                for model in generation_backend_errors
            )
            if generation_backend_errors
            else "<none>"
        ),
        "",
        "## Eval Set Used",
        "",
        f"- Total questions: {len(eval_set)}",
        f"- Answerable questions: {sum(1 for q in eval_set if q.get('expected_behavior') == 'answerable')}",
        f"- Insufficient-evidence questions: {sum(1 for q in eval_set if q.get('expected_behavior') == 'insufficient_evidence')}",
        "- Includes fact lookup, method, limitation, comparison, and no-answer/insufficient-evidence questions.",
        "",
        "Question source:",
        "",
        *question_source_lines,
        "",
        "## Top-k",
        "",
        f"`top_k={eval_summary['top_k']}` using `{eval_summary['retriever']['name']}`.",
        "",
        "## Summary Retrieval",
        "",
        f"- Expected-source hit rate: {eval_summary['expected_source_hit_rate']:.3f}",
        f"- Expected-source hits: {eval_summary['expected_source_hits']} / {eval_summary['expected_source_questions']}",
        f"- Retrieval misses: {', '.join(retrieval_misses) if retrieval_misses else '<none>'}",
        "",
        "## Summary QA Per Model",
        "",
        "| Model | Availability | Answered | Insufficient | Backend errors | Invalid citations |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in per_model:
        lines.append(
            "| {model_name} | {availability_status} | {answers_succeeded} | {insufficient_evidence_count} | {backend_errors} | {invalid_citation_count} |".format(
                **model
            )
        )
    lines.extend(
        [
            "",
            "## Citation Validation",
            "",
            f"- Invalid citation records: {citation_summary['invalid_citation_count']}",
            f"- Invented citation labels: {citation_summary['invented_citation_count']}",
            f"- Answered without valid citation: {citation_summary['answered_without_valid_citation_count']}",
            f"- Status counts: `{json.dumps(citation_summary['status_counts'], sort_keys=True)}`",
            "",
            "## Principal Failure Cases",
            "",
            f"- Retrieval misses: {', '.join(retrieval_misses) if retrieval_misses else '<none>'}",
            f"- Backend/model availability warnings: `{json.dumps(eval_summary['warnings_by_type'], sort_keys=True)}`",
            "- Manual grounding review remains pending for all generated answers.",
            "",
            "## Model Comparison",
            "",
            _comparison_sentence(per_model),
            "",
            "## Improvements Compared To Mini-RAG v0",
            "",
            "- Uses v1 chunk IDs and quality flags.",
            "- Uses deterministic `[C1]`, `[C2]` citation labels tied to retrieved chunks.",
            "- Validates generated citation labels automatically.",
            "- Produces per-model answer files and a comparative summary.",
            "- Separates retrieval evidence, generated answer, citation audit, and manual grounding status.",
            "",
            "## What Is Not Yet Demonstrated",
            "",
            "- Scientific answer correctness.",
            "- Robust semantic retrieval beyond the TF-IDF local baseline.",
            "- Reliable abstention behavior across models.",
            "- Production readiness for detector simulation replacement decisions.",
            "",
            "## Limits",
            "",
            "- No training, LoRA, QLoRA, discovery, corpus scaling, or parser changes were performed.",
            "- Chunk text is inherited from v1; 31 chunks still carry `very_long_text` quality flags.",
            "- Citation validation checks labels, not the full factual entailment of every sentence.",
            "- Local Ollama availability determines whether model comparisons contain generated answers or backend-error records.",
            "",
            "## Recommended Next Step",
            "",
            "Run manual review on generated answers for available models, then address the retrieval misses and very-long chunk policy before claiming any answer-quality improvement.",
            "",
            "## Output Directory",
            "",
            f"`{output_dir}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_run_config(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    top_k: int,
    base_url: str,
    registry: dict[str, Any],
    eval_set_mode: str,
    eval_questions_file: Path | str | None,
    static_v1_questions_enabled: bool,
    static_v1_questions_path: Path | str | None,
    llm_question_generation_enabled: bool,
) -> dict[str, Any]:
    """Build run config for traceability."""
    return {
        "qa_run_config_version": "qa_run_config_v1",
        "created_at": utc_now_iso(),
        "rag_ready_dir": str(rag_ready_dir),
        "output_dir": str(output_dir),
        "top_k": top_k,
        "base_url": base_url,
        "retriever": {
            "name": TfidfRetrieverV1.name,
            "version": TfidfRetrieverV1.version,
        },
        "models": registry.get("models", []),
        "eval_set": {
            "mode": eval_set_mode,
            "questions_file": str(eval_questions_file) if eval_questions_file is not None else None,
            "static_v1_questions_enabled": static_v1_questions_enabled,
            "static_v1_questions_path": (
                str(static_v1_questions_path)
                if static_v1_questions_path is not None
                else None
            ),
            "llm_question_generation_enabled": llm_question_generation_enabled,
            "notes": (
                "The complete hardened eval set is loaded from a data file when questions_file "
                "is set. No operational questions are hard-coded in the QA runtime."
            ),
        },
        "scope": "QA grounded v1; no training, no discovery, no parser changes, no scientific reliability claim",
    }


def build_manifest(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    dataset: dict[str, Any],
    registry: dict[str, Any],
    eval_set_mode: str,
    eval_questions_file: Path | str | None,
) -> dict[str, Any]:
    """Build QA v1 output manifest."""
    files = {
        key: str(output_dir / filename)
        for key, filename in QA_OUTPUT_FILES.items()
        if key != "qa_manifest"
    }
    for model in registry.get("models", []):
        slug = model.get("model_slug", model_slug(str(model.get("model_name") or "")))
        files[f"rag_answers_{slug}"] = str(output_dir / f"rag_answers_{slug}_v1.jsonl")
    missing = [path for path in files.values() if not Path(path).exists()]
    return {
        "qa_manifest_version": "qa_manifest_v1",
        "created_at": utc_now_iso(),
        "source_rag_ready_dir": str(rag_ready_dir),
        "output_dir": str(output_dir),
        "dataset_counts": dataset["counts"],
        "eval_set": {
            "mode": eval_set_mode,
            "questions_file": str(eval_questions_file) if eval_questions_file is not None else None,
        },
        "files": files,
        "validation": {
            "status": "PASS" if not missing else "FAIL",
            "missing_outputs": missing,
            "warnings": [
                "Manual review remains required.",
                "Citation validation does not prove scientific reliability.",
            ],
        },
        "scope": "QA grounded v1; local generation only, no training, no discovery",
    }


def run_qa_grounded_v1(
    *,
    rag_ready_dir: Path,
    retrieval_results_file: Path | None = None,
    source_policy: str | None = None,
    output_dir: Path,
    report_path: Path,
    model_names: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_predict: int = 512,
    timeout_seconds: int = 120,
    eval_set_mode: str = "v0_plus_static_v1",
    eval_questions_file: Path | str | None = None,
    static_v1_questions_enabled: bool = False,
    static_v1_questions_path: Path | str | None = None,
    llm_question_generation_enabled: bool = False,
) -> dict[str, Any]:
    """Run the full QA grounded v1 task and write requested artifacts."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if source_policy not in {None, "none", "explicit_arxiv"}:
        raise ValueError("invalid source policy")
    if eval_set_mode == "from_file" and eval_questions_file is None:
        raise ValueError("eval_set mode 'from_file' requires eval_questions_file/questions_file")
    use_eval_questions_file = eval_questions_file is not None
    dataset = load_rag_ready_v1_dataset(
        rag_ready_dir,
        include_eval_questions_v0=not use_eval_questions_file,
    )
    eval_set = build_eval_set_v1(
        [] if use_eval_questions_file else dataset["eval_questions_v0"],
        eval_questions_file=eval_questions_file,
        static_v1_questions_enabled=static_v1_questions_enabled,
        static_v1_questions_path=static_v1_questions_path,
        llm_question_generation_enabled=llm_question_generation_enabled,
    )
    retrieval_provenance = None
    if retrieval_results_file is not None:
        from kamiknows.rag.saved_retrieval import load_saved_contexts
        retrieved_contexts, retrieval_provenance = load_saved_contexts(
            retrieval_results_file, dataset=dataset, eval_set=eval_set,
            questions_file=eval_questions_file, top_k=top_k, source_policy=source_policy)
        source_policy = retrieval_provenance['source_policy']
    else:
        retrieved_contexts = retrieve_contexts_v1(
            chunks=dataset["chunks"], eval_set=eval_set, top_k=top_k,
            source_policy=source_policy or "none")
        source_policy = source_policy or "none"
    actual_retriever = (retrieved_contexts[0]["retriever"] if retrieved_contexts else
                        {"name": TfidfRetrieverV1.name, "version": TfidfRetrieverV1.version})
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = build_default_model_registry(
        model_names=model_names,
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
    )
    registry = annotate_registry_availability(registry=registry, base_url=base_url)
    write_json(output_dir / QA_OUTPUT_FILES["qa_model_registry"], registry)

    write_jsonl(output_dir / QA_OUTPUT_FILES["rag_eval_set"], eval_set)
    (output_dir / QA_OUTPUT_FILES["rag_eval_set_card"]).write_text(
        build_eval_set_card_markdown(eval_set),
        encoding="utf-8",
    )

    write_jsonl(output_dir / QA_OUTPUT_FILES["retrieved_contexts"], retrieved_contexts)

    known_chunk_ids = [str(chunk.get("chunk_id") or "") for chunk in dataset["chunks"]]
    answer_records_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_config in registry["models"]:
        records = generate_model_answer_records(
            model_config=model_config,
            retrieved_contexts=retrieved_contexts,
            known_chunk_ids=known_chunk_ids,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        answer_records_by_model[model_config["model_name"]] = records
        write_jsonl(
            output_dir / f"rag_answers_{model_config['model_slug']}_v1.jsonl",
            records,
        )

    audit_records = build_citation_audit_records(answer_records_by_model)
    write_jsonl(output_dir / QA_OUTPUT_FILES["citation_audit"], audit_records)
    citation_summary = build_citation_validation_summary(audit_records)
    write_json(output_dir / QA_OUTPUT_FILES["citation_validation_summary"], citation_summary)

    eval_summary = build_eval_summary_v1(
        eval_set=eval_set,
        retrieved_contexts=retrieved_contexts,
        registry=registry,
        answer_records_by_model=answer_records_by_model,
        citation_summary=citation_summary,
        top_k=top_k,
    )
    eval_summary["retriever"] = actual_retriever
    write_json(output_dir / QA_OUTPUT_FILES["rag_eval_summary"], eval_summary)

    (output_dir / QA_OUTPUT_FILES["rag_model_comparison"]).write_text(
        build_model_comparison_markdown(eval_summary=eval_summary),
        encoding="utf-8",
    )
    (output_dir / QA_OUTPUT_FILES["rag_error_analysis"]).write_text(
        build_error_analysis_markdown_v1(
            retrieved_contexts=retrieved_contexts,
            answer_records_by_model=answer_records_by_model,
        ),
        encoding="utf-8",
    )

    write_json(
        output_dir / QA_OUTPUT_FILES["qa_run_config"],
        build_run_config(
            rag_ready_dir=rag_ready_dir,
            output_dir=output_dir,
            top_k=top_k,
            base_url=base_url,
            registry=registry,
            eval_set_mode=eval_set_mode,
            eval_questions_file=eval_questions_file,
            static_v1_questions_enabled=static_v1_questions_enabled,
            static_v1_questions_path=static_v1_questions_path,
            llm_question_generation_enabled=llm_question_generation_enabled,
        ),
    )
    manifest = build_manifest(
        rag_ready_dir=rag_ready_dir,
        output_dir=output_dir,
        dataset=dataset,
        registry=registry,
        eval_set_mode=eval_set_mode,
        eval_questions_file=eval_questions_file,
    )
    if retrieval_provenance is not None:
        config_path = output_dir / QA_OUTPUT_FILES["qa_run_config"]
        config = read_json(config_path)
        config.update(retriever=actual_retriever, saved_retrieval=retrieval_provenance)
        write_json(config_path, config)
        manifest.update(retriever=actual_retriever, saved_retrieval=retrieval_provenance)
    config_path = output_dir / QA_OUTPUT_FILES["qa_run_config"]
    config = read_json(config_path)
    config["source_policy"] = source_policy
    write_json(config_path, config)
    manifest["source_policy"] = source_policy
    write_json(output_dir / QA_OUTPUT_FILES["qa_manifest"], manifest)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_task_report_markdown(
            rag_ready_dir=rag_ready_dir,
            output_dir=output_dir,
            dataset=dataset,
            eval_set=eval_set,
            eval_questions_file=eval_questions_file,
            registry=registry,
            eval_summary=eval_summary,
            citation_summary=citation_summary,
        ),
        encoding="utf-8",
    )

    return {
        "dataset": dataset,
        "registry": registry,
        "eval_set": eval_set,
        "retrieved_contexts": retrieved_contexts,
        "answer_records_by_model": answer_records_by_model,
        "citation_audit": audit_records,
        "citation_summary": citation_summary,
        "eval_summary": eval_summary,
        "manifest": manifest,
        "report_path": str(report_path),
    }


def _normalize_eval_question_v0(question: dict[str, Any]) -> dict[str, Any]:
    expected_arxiv_ids = list(question.get("expected_source_arxiv_ids", []) or [])
    question_type = str(question.get("question_type") or "")
    expected_behavior = "insufficient_evidence" if question_type in {"unanswerable", "no_answer"} else "answerable"
    return {
        "question_id": question.get("question_id", ""),
        "question": question.get("question", ""),
        "question_type": question_type,
        "expected_arxiv_ids": expected_arxiv_ids,
        "expected_source_arxiv_ids": expected_arxiv_ids,
        "expected_paper_ids": [
            str(arxiv_id).replace(".", "_") for arxiv_id in expected_arxiv_ids
        ],
        "expected_behavior": expected_behavior,
        "expected_section_keywords": list(question.get("expected_section_keywords", []) or []),
        "evaluation_criteria": question.get("evaluation_criteria", ""),
        "notes": question.get("notes", ""),
        "source": "eval_questions_v0",
    }


def _additional_eval_questions_v1() -> list[dict[str, Any]]:
    """Deprecated compatibility helper.

    Operational eval questions live in data/eval/rag_eval_set_v1_hardened.jsonl.
    This helper intentionally contains no question text and is not used by the
    QA runtime.
    """
    return []


def _comparison_sentence(per_model: list[dict[str, Any]]) -> str:
    available = [model for model in per_model if model.get("availability_status") == "available"]
    if not available:
        return (
            "No configured generator was available through local Ollama in this run, so qwen3:0.6b, "
            "qwen3:4b-instruct, and mistral:7b are compared only by availability/error records."
        )
    names = ", ".join(model["model_name"] for model in available)
    return (
        f"Available local generators in this run: {names}. See `rag_model_comparison_v1.md` "
        "for citation and abstention metrics."
    )


def _citation_validation_note(status: str) -> str:
    if status == "valid":
        return "All generated citation labels are valid for the retrieved context."
    if status == "insufficient_evidence_no_citation":
        return "Insufficient-evidence response has no citations, which is allowed."
    if status == "not_applicable_backend_error":
        return "Citation validation is not applicable because generation failed or was unavailable."
    if status == "invented_citation":
        return "At least one generated citation label was not present in the retrieved context."
    if status == "unknown_chunk_id":
        return "A cited label mapped to a chunk ID outside the known corpus."
    if status == "answered_without_valid_citation":
        return "The model answered without at least one valid retrieved-context citation."
    return "Unknown citation validation status."


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _error_categories() -> list[str]:
    return [
        "retrieval_miss",
        "answer_not_grounded",
        "invalid_citation",
        "invented_citation",
        "insufficient_evidence_expected",
        "insufficient_evidence_unexpected",
        "overclaim",
        "incomplete_answer",
        "backend_error",
        "ambiguous_question",
    ]


def _is_ambiguous_question(context: dict[str, Any]) -> bool:
    text = str(context.get("question") or "").lower()
    return any(marker in text for marker in ("first selected", "another selected", "selected paper"))


def _model_question(record: dict[str, Any]) -> str:
    return f"{record.get('model_name', '')}/{record.get('question_id', '')}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _unique_preserve_order(values: list[Any]) -> list[str]:
    seen = set()
    unique: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _dedupe(values: list[Any]) -> list[Any]:
    seen = set()
    deduped = []
    for value in values:
        marker = json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(value)
    return deduped
