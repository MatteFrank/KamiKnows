"""Embedding retrieval benchmark for Phase 2 model selection."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from kamiknows.rag.citation_validation import validate_citations
from kamiknows.rag.embedding_registry import (
    apply_embedding_prefix,
    get_embedding_registry,
    select_embedding_model_configs,
)
from kamiknows.rag.load_dataset import load_rag_ready_dataset, read_json
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag.retriever_v1 import tokenize_v1
from kamiknows.run_metadata import utc_now_iso

REQUIRED_OUTPUT_FILES = {
    "embedding_model_registry": "embedding_model_registry_v1.json",
    "embedding_benchmark_results": "embedding_benchmark_results_v1.jsonl",
    "embedding_eval_summary": "embedding_eval_summary_v1.json",
    "citation_validation_summary": "citation_validation_summary_v1.json",
    "embedding_model_comparison": "embedding_model_comparison_v1.md",
}


class EmbeddingEncoder:
    """Small encoder wrapper with deterministic local fallback."""

    def __init__(self, config: dict[str, Any], *, backend: str = "auto") -> None:
        if backend not in {"auto", "hash", "sentence-transformers"}:
            raise ValueError("backend must be auto, hash, or sentence-transformers")
        self.config = dict(config)
        self.requested_backend = backend
        self.backend = "hash"
        self.model: Any = None
        self.warning = ""
        self.dimension = int(self.config.get("embedding_dimension") or 384)
        if backend in {"auto", "sentence-transformers"}:
            self._try_load_sentence_transformer(required=backend == "sentence-transformers")

    def _try_load_sentence_transformer(self, *, required: bool) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError as exc:
            self.warning = "sentence_transformers is not installed; using deterministic hash fallback."
            if required:
                raise RuntimeError(self.warning) from exc
            return

        try:
            self.model = SentenceTransformer(
                self.config["model_name"],
                local_files_only=True,
            )
        except TypeError:
            self.warning = (
                "Installed sentence_transformers does not support local_files_only; "
                "using deterministic hash fallback to avoid network access."
            )
            if required:
                raise RuntimeError(self.warning)
            return
        except Exception as exc:
            self.warning = f"Could not load local model cache; using deterministic hash fallback: {exc}"
            if required:
                raise RuntimeError(self.warning) from exc
            return

        self.backend = "sentence-transformers"
        dim = getattr(self.model, "get_sentence_embedding_dimension", lambda: None)()
        if dim:
            self.dimension = int(dim)
        self.warning = ""

    def encode(self, texts: list[str], *, text_kind: str) -> list[list[float]]:
        """Encode texts as normalized vectors when configured."""
        prefixed = [apply_embedding_prefix(self.config, text, text_kind=text_kind) for text in texts]
        if self.backend == "sentence-transformers":
            vectors = self.model.encode(  # type: ignore[union-attr]
                prefixed,
                normalize_embeddings=bool(self.config.get("normalize_embeddings", True)),
                show_progress_bar=False,
            )
            return [_coerce_vector(vector) for vector in vectors]
        return [
            hash_embedding(
                text,
                dimension=self.dimension,
                model_name=str(self.config["model_name"]),
                normalize=bool(self.config.get("normalize_embeddings", True)),
            )
            for text in prefixed
        ]


def hash_embedding(
    text: str,
    *,
    dimension: int,
    model_name: str,
    normalize: bool = True,
) -> list[float]:
    """Create a deterministic lexical embedding for offline tests and fallback runs."""
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    vector = [0.0] * dimension
    counts = Counter(tokenize_v1(text))
    for token, count in counts.items():
        digest = hashlib.blake2b(
            f"{model_name}\0{token}".encode("utf-8"),
            digest_size=8,
        ).digest()
        raw = int.from_bytes(digest, "big")
        index = raw % dimension
        sign = 1.0 if (raw >> 63) == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    if normalize:
        return _normalize(vector)
    return vector


def build_embedding_text(chunk: dict[str, Any]) -> str:
    """Build passage text from chunk metadata and body text."""
    return " ".join(
        [
            str(chunk.get("title") or ""),
            str(chunk.get("section_heading") or ""),
            str(chunk.get("text") or ""),
        ]
    )


def retrieve_embedding_chunks(
    *,
    query_vector: list[float],
    chunk_vectors: list[list[float]],
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return top-k chunks by dot product similarity."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    scored: list[tuple[float, str, int, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        score = _dot(query_vector, chunk_vectors[index])
        scored.append((score, str(chunk.get("chunk_id") or ""), index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        build_retrieved_chunk_record(chunk=chunk, rank=rank, score=score)
        for rank, (score, _chunk_id, _index, chunk) in enumerate(scored[:top_k], start=1)
    ]


def build_retrieved_chunk_record(
    *,
    chunk: dict[str, Any],
    rank: int,
    score: float,
) -> dict[str, Any]:
    """Build one retrieved chunk with deterministic citation metadata."""
    text = " ".join(str(chunk.get("text") or "").split())
    return {
        "citation_label": f"[C{rank}]",
        "chunk_id": chunk.get("chunk_id", ""),
        "paper_id": chunk.get("paper_id", ""),
        "arxiv_id": chunk.get("arxiv_id", ""),
        "title": chunk.get("title", ""),
        "section": chunk.get("section_heading", ""),
        "section_heading": chunk.get("section_heading", ""),
        "source_type": chunk.get("source_type", ""),
        "retrieved_rank": rank,
        "rank": rank,
        "retrieval_score": round(float(score), 8),
        "score": round(float(score), 8),
        "text_preview": text[:280],
    }


def evaluate_embedding_retrieval(
    *,
    retrieved_chunks: list[dict[str, Any]],
    question: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate retrieval against expected chunk, paper, or arXiv IDs."""
    expected_chunk_ids = {
        str(value)
        for value in _as_list(question.get("expected_chunk_ids") or question.get("expected_chunk_id"))
        if value
    }
    expected_paper_ids = {
        str(value)
        for value in _as_list(question.get("expected_paper_ids") or question.get("expected_paper_id"))
        if value
    }
    expected_arxiv_ids = {
        str(value)
        for value in _as_list(question.get("expected_source_arxiv_ids") or question.get("expected_arxiv_ids"))
        if value
    }
    has_expected = bool(expected_chunk_ids or expected_paper_ids or expected_arxiv_ids)
    if not has_expected:
        return {
            "has_expected_source": False,
            "expected_source_hit": False,
            "expected_source_hit_rank": None,
            "reciprocal_rank": 0.0,
            "notes": "No expected source identifiers were specified.",
        }

    for chunk in retrieved_chunks:
        if (
            str(chunk.get("chunk_id") or "") in expected_chunk_ids
            or str(chunk.get("paper_id") or "") in expected_paper_ids
            or str(chunk.get("arxiv_id") or "") in expected_arxiv_ids
        ):
            rank = int(chunk.get("retrieved_rank") or chunk.get("rank") or 0)
            return {
                "has_expected_source": True,
                "expected_source_hit": True,
                "expected_source_hit_rank": rank,
                "reciprocal_rank": 1.0 / rank if rank else 0.0,
                "notes": "At least one expected source identifier was retrieved.",
            }
    return {
        "has_expected_source": True,
        "expected_source_hit": False,
        "expected_source_hit_rank": None,
        "reciprocal_rank": 0.0,
        "notes": "No retrieved chunk matched the expected source identifiers.",
    }


def compute_embedding_metrics(question_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute hit@k, miss count, mean rank, and MRR."""
    expected_records = [
        record for record in question_records if record["retrieval_eval"]["has_expected_source"]
    ]
    hits = [
        record for record in expected_records if record["retrieval_eval"]["expected_source_hit"]
    ]
    ranks = [
        record["retrieval_eval"]["expected_source_hit_rank"]
        for record in hits
        if record["retrieval_eval"]["expected_source_hit_rank"]
    ]
    reciprocal_ranks = [
        record["retrieval_eval"]["reciprocal_rank"] for record in expected_records
    ]
    denominator = len(expected_records)
    return {
        "expected_source_questions": denominator,
        "expected_source_hits": len(hits),
        "expected_source_hit_rate_at_k": len(hits) / denominator if denominator else 0.0,
        "retrieval_miss": denominator - len(hits),
        "mean_rank_expected_source": sum(ranks) / len(ranks) if ranks else None,
        "mrr": sum(reciprocal_ranks) / denominator if denominator else 0.0,
        "missing_expected_source_questions": [
            record["question_id"]
            for record in question_records
            if not record["retrieval_eval"]["has_expected_source"]
        ],
        "miss_question_ids": [
            record["question_id"]
            for record in expected_records
            if not record["retrieval_eval"]["expected_source_hit"]
        ],
    }


def benchmark_embedding_model(
    *,
    config: dict[str, Any],
    chunks: list[dict[str, Any]],
    eval_questions: list[dict[str, Any]],
    top_k: int,
    encoder_backend: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Benchmark one embedding model config."""
    encoder = EmbeddingEncoder(config, backend=encoder_backend)
    warnings = []
    if encoder.warning:
        warnings.append(encoder.warning)
    if encoder.backend == "hash":
        warnings.append(
            "Deterministic hash embeddings were used; this is an offline fallback, not a real model-quality result."
        )

    passage_texts = [build_embedding_text(chunk) for chunk in chunks]
    build_start = time.perf_counter()
    chunk_vectors = encoder.encode(passage_texts, text_kind="passage")
    index_build_time_sec = time.perf_counter() - build_start
    cache_path = save_embedding_cache(
        output_dir=output_dir,
        config=config,
        encoder=encoder,
        chunks=chunks,
        chunk_vectors=chunk_vectors,
    )

    question_records = []
    query_times = []
    known_chunk_ids = [str(chunk.get("chunk_id") or "") for chunk in chunks]
    citation_status_counts: Counter[str] = Counter()
    citation_invalid_records = []
    for question in eval_questions:
        question_text = str(question.get("question") or "")
        query_start = time.perf_counter()
        query_vector = encoder.encode([question_text], text_kind="query")[0]
        retrieved_chunks = retrieve_embedding_chunks(
            query_vector=query_vector,
            chunk_vectors=chunk_vectors,
            chunks=chunks,
            top_k=top_k,
        )
        query_times.append(time.perf_counter() - query_start)
        retrieval_eval = evaluate_embedding_retrieval(
            retrieved_chunks=retrieved_chunks,
            question=question,
        )
        citation_validation = validate_citations(
            {
                "citations": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "citation_label": chunk["citation_label"],
                    }
                    for chunk in retrieved_chunks
                ]
            },
            [chunk["chunk_id"] for chunk in retrieved_chunks],
            known_chunk_ids=known_chunk_ids,
        )
        citation_status_counts[citation_validation["status"]] += 1
        if not citation_validation["valid"]:
            citation_invalid_records.append(
                {
                    "question_id": question.get("question_id", ""),
                    "status": citation_validation["status"],
                    "notes": citation_validation["notes"],
                }
            )
        question_records.append(
            {
                "question_id": question.get("question_id", ""),
                "question": question_text,
                "question_type": question.get("question_type", ""),
                "expected_source_arxiv_ids": list(question.get("expected_source_arxiv_ids", []) or []),
                "retrieved_chunks": retrieved_chunks,
                "retrieval_eval": retrieval_eval,
                "citation_validation": citation_validation,
            }
        )

    metrics = compute_embedding_metrics(question_records)
    return {
        "record_type": "embedding_model_benchmark_v1",
        "model_key": config["model_key"],
        "model_name": config["model_name"],
        "provider_library": config["provider_library"],
        "encoder_backend": encoder.backend,
        "requested_encoder_backend": encoder_backend,
        "embedding_dimension": encoder.dimension,
        "requires_prefix": config["requires_prefix"],
        "query_prefix": config["query_prefix"],
        "passage_prefix": config["passage_prefix"],
        "normalize_embeddings": config["normalize_embeddings"],
        "max_length": config["max_length"],
        "top_k": top_k,
        "questions_total": len(eval_questions),
        "chunks_total": len(chunks),
        "expected_source_hit_rate_at_k": metrics["expected_source_hit_rate_at_k"],
        "retrieval_miss": metrics["retrieval_miss"],
        "mean_rank_expected_source": metrics["mean_rank_expected_source"],
        "mrr": metrics["mrr"],
        "average_query_time_sec": sum(query_times) / len(query_times) if query_times else 0.0,
        "index_build_time_sec": index_build_time_sec,
        "warnings": warnings,
        "embedding_cache_path": str(cache_path),
        "metrics": metrics,
        "question_records": question_records,
        "citation_summary": {
            "total_questions": len(question_records),
            "status_counts": dict(citation_status_counts),
            "invalid_records": citation_invalid_records,
        },
    }


def run_embedding_model_selection_v1(
    *,
    rag_ready_dir: Path,
    output_dir: Path,
    model_names_or_keys: list[str] | None = None,
    top_k: int = 5,
    encoder_backend: str = "auto",
    tfidf_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Run embedding model selection and write all required artifacts."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    dataset = load_rag_ready_dataset(rag_ready_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = get_embedding_registry()
    selected_configs = select_embedding_model_configs(model_names_or_keys)
    write_json(output_dir / "embedding_model_registry_v1.json", registry)

    benchmark_records = [
        benchmark_embedding_model(
            config=config,
            chunks=dataset["chunks"],
            eval_questions=dataset["eval_questions"],
            top_k=top_k,
            encoder_backend=encoder_backend,
            output_dir=output_dir,
        )
        for config in selected_configs
    ]
    write_jsonl(output_dir / "embedding_benchmark_results_v1.jsonl", benchmark_records)

    tfidf_summary = read_json(tfidf_summary_path) if tfidf_summary_path and tfidf_summary_path.exists() else None
    eval_summary = build_embedding_eval_summary(
        records=benchmark_records,
        dataset_counts=dataset["counts"],
        top_k=top_k,
        tfidf_summary=tfidf_summary,
        tfidf_summary_path=tfidf_summary_path,
    )
    write_json(output_dir / "embedding_eval_summary_v1.json", eval_summary)

    citation_summary = build_citation_validation_summary(benchmark_records)
    write_json(output_dir / "citation_validation_summary_v1.json", citation_summary)

    comparison_md = build_embedding_model_comparison_markdown(
        records=benchmark_records,
        eval_summary=eval_summary,
        citation_summary=citation_summary,
    )
    (output_dir / "embedding_model_comparison_v1.md").write_text(comparison_md, encoding="utf-8")

    return {
        "dataset": dataset,
        "registry": registry,
        "benchmark_records": benchmark_records,
        "eval_summary": eval_summary,
        "citation_summary": citation_summary,
    }


def build_embedding_eval_summary(
    *,
    records: list[dict[str, Any]],
    dataset_counts: dict[str, int],
    top_k: int,
    tfidf_summary: dict[str, Any] | None,
    tfidf_summary_path: Path | None,
) -> dict[str, Any]:
    """Build cross-model embedding summary and select a provisional baseline."""
    selected = select_provisional_baseline(records)
    return {
        "embedding_eval_summary_version": "embedding_eval_summary_v1",
        "created_at": utc_now_iso(),
        "dataset_counts": dataset_counts,
        "top_k": top_k,
        "models_total": len(records),
        "models_compared": [
            {
                "model_key": record["model_key"],
                "model_name": record["model_name"],
                "encoder_backend": record["encoder_backend"],
                "embedding_dimension": record["embedding_dimension"],
                "expected_source_hit_rate_at_k": record["expected_source_hit_rate_at_k"],
                "retrieval_miss": record["retrieval_miss"],
                "mean_rank_expected_source": record["mean_rank_expected_source"],
                "mrr": record["mrr"],
                "average_query_time_sec": record["average_query_time_sec"],
                "index_build_time_sec": record["index_build_time_sec"],
                "warnings": record["warnings"],
            }
            for record in records
        ],
        "comparison_to_tfidf_if_available": build_tfidf_comparison(
            selected=selected,
            tfidf_summary=tfidf_summary,
            tfidf_summary_path=tfidf_summary_path,
            top_k=top_k,
        ),
        "selected_provisional_embedding_model": {
            "model_key": selected["model_key"],
            "model_name": selected["model_name"],
            "encoder_backend": selected["encoder_backend"],
            "expected_source_hit_rate_at_k": selected["expected_source_hit_rate_at_k"],
            "mrr": selected["mrr"],
            "reason": (
                "Selected by highest expected-source hit rate, then MRR, then lower average query time."
            ),
            "status": "provisional_retrieval_baseline",
        },
        "manual_review": {
            "required": True,
            "status": "pending",
        },
        "scope": "embedding retrieval selection only; no answer generation, no free citations, no scientific correctness claim",
    }


def select_provisional_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the provisional model by retrieval metrics."""
    if not records:
        raise ValueError("records must not be empty")
    return max(
        records,
        key=lambda record: (
            record["expected_source_hit_rate_at_k"],
            record["mrr"],
            -record["average_query_time_sec"],
        ),
    )


def build_tfidf_comparison(
    *,
    selected: dict[str, Any],
    tfidf_summary: dict[str, Any] | None,
    tfidf_summary_path: Path | None,
    top_k: int,
) -> dict[str, Any]:
    """Compare selected embedding result to available TF-IDF summary."""
    if not tfidf_summary:
        return {
            "available": False,
            "notes": "No TF-IDF summary was available for comparison.",
        }
    if tfidf_summary.get("retriever_eval_summary_version") == "retriever_eval_summary_v1":
        hit_rates = tfidf_summary.get("hit_rate_by_top_k", {})
        tfidf_hit_rate = hit_rates.get(str(top_k))
        if isinstance(tfidf_hit_rate, int | float):
            return {
                "available": True,
                "summary_path": str(tfidf_summary_path) if tfidf_summary_path else None,
                "tfidf_retriever": "tfidf_local_v1",
                "tfidf_top_k": top_k,
                "tfidf_expected_source_hit_rate": tfidf_hit_rate,
                "selected_embedding_expected_source_hit_rate": selected["expected_source_hit_rate_at_k"],
                "delta": selected["expected_source_hit_rate_at_k"] - float(tfidf_hit_rate),
            }
    if tfidf_summary.get("rag_eval_summary_version") == "rag_eval_summary_v0":
        retrieval = tfidf_summary.get("retrieval", {})
        tfidf_hit_rate = retrieval.get("expected_source_hit_rate")
        if isinstance(tfidf_hit_rate, int | float):
            return {
                "available": True,
                "summary_path": str(tfidf_summary_path) if tfidf_summary_path else None,
                "tfidf_retriever": "tfidf_local_v0",
                "tfidf_top_k": retrieval.get("top_k"),
                "tfidf_expected_source_hit_rate": tfidf_hit_rate,
                "selected_embedding_expected_source_hit_rate": selected["expected_source_hit_rate_at_k"],
                "delta": selected["expected_source_hit_rate_at_k"] - float(tfidf_hit_rate),
            }
    return {
        "available": False,
        "summary_path": str(tfidf_summary_path) if tfidf_summary_path else None,
        "notes": "TF-IDF summary format did not include a comparable hit rate.",
    }


def build_citation_validation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize deterministic citation-label validation across benchmark records."""
    status_counts: Counter[str] = Counter()
    invalid_records = []
    total_question_records = 0
    for record in records:
        for question_record in record["question_records"]:
            total_question_records += 1
            status = question_record["citation_validation"]["status"]
            status_counts[status] += 1
            if not question_record["citation_validation"]["valid"]:
                invalid_records.append(
                    {
                        "model_key": record["model_key"],
                        "question_id": question_record["question_id"],
                        "status": status,
                        "notes": question_record["citation_validation"]["notes"],
                    }
                )
    return {
        "citation_validation_summary_version": "citation_validation_summary_v1",
        "total_question_records": total_question_records,
        "status_counts": dict(status_counts),
        "invalid_records": invalid_records,
        "all_citations_reference_retrieved_chunks": not invalid_records,
        "notes": "Citation labels are deterministic retrieval labels, not generated model citations.",
    }


def build_embedding_model_comparison_markdown(
    *,
    records: list[dict[str, Any]],
    eval_summary: dict[str, Any],
    citation_summary: dict[str, Any],
) -> str:
    """Build Markdown comparison table for embedding benchmark outputs."""
    selected = eval_summary["selected_provisional_embedding_model"]
    lines = [
        "# KamiKnows Embedding Model Comparison v1",
        "",
        "This benchmark compares retrieval behavior on the fixed FastCaloSimulation corpus and eval set.",
        "",
        "The selected embedding model is a provisional retrieval baseline, not a final scientific retrieval solution.",
        "",
        "## Metrics",
        "",
        "| Model | Backend | Dim | Hit@k | Misses | Mean Rank | MRR | Avg Query Sec | Warnings |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        warnings = "; ".join(record.get("warnings", [])) or "none"
        mean_rank = record["mean_rank_expected_source"]
        mean_rank_text = f"{mean_rank:.3f}" if mean_rank is not None else "n/a"
        lines.append(
            "| {model} | {backend} | {dim} | {hit:.3f} | {misses} | {mean_rank} | {mrr:.3f} | {query:.6f} | {warnings} |".format(
                model=record["model_name"],
                backend=record["encoder_backend"],
                dim=record["embedding_dimension"],
                hit=record["expected_source_hit_rate_at_k"],
                misses=record["retrieval_miss"],
                mean_rank=mean_rank_text,
                mrr=record["mrr"],
                query=record["average_query_time_sec"],
                warnings=warnings,
            )
        )
    lines.extend(
        [
            "",
            "## Selected Provisional Baseline",
            "",
            f"Selected: `{selected['model_name']}` using `{selected['encoder_backend']}` backend.",
            "",
            selected["reason"],
            "",
            "## Citation Compatibility",
            "",
            f"All citations reference retrieved chunks: {citation_summary['all_citations_reference_retrieved_chunks']}",
            f"Status counts: `{json.dumps(citation_summary['status_counts'], sort_keys=True)}`",
            "",
            "Manual review remains required. These metrics do not establish scientific correctness.",
            "",
        ]
    )
    return "\n".join(lines)


def save_embedding_cache(
    *,
    output_dir: Path,
    config: dict[str, Any],
    encoder: EmbeddingEncoder,
    chunks: list[dict[str, Any]],
    chunk_vectors: list[list[float]],
) -> Path:
    """Persist a local embedding cache for traceability."""
    cache_dir = output_dir / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{config['model_key']}_chunk_embeddings_v1.json"
    payload = {
        "cache_version": "embedding_cache_v1",
        "model_key": config["model_key"],
        "model_name": config["model_name"],
        "encoder_backend": encoder.backend,
        "embedding_dimension": encoder.dimension,
        "chunk_ids": [chunk.get("chunk_id", "") for chunk in chunks],
        "embeddings": [[round(value, 8) for value in vector] for vector in chunk_vectors],
    }
    write_json(path, payload)
    return path


def default_tfidf_summary_path(rag_ready_dir: Path) -> Path:
    """Return the default TF-IDF v1 summary path for the current output layout."""
    return Path(rag_ready_dir).parent / "rag_v1_fastcalosim_retrieval" / "retriever_eval_summary_v1.json"


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def _coerce_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
