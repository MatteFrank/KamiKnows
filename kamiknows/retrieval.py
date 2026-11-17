"""Step 3: retrieve the top-k chunks for each question.

The query is the question text only; annotations in the questions file
(expected sources, behaviour, ...) are used for evaluation, never for ranking.

Retrievers:
    tfidf   lexical cosine; title and section heading are counted twice
    dense   cosine between sentence-transformers embeddings of question and
            "title section text" of each chunk

Source policy ``explicit_arxiv``: if the question names arXiv ids, only chunks
of those papers are ranked and the best chunk of each named paper is always
kept. If a named paper is not in the dataset the question gets no passages.

Output: ``contexts.jsonl``, one record per question with its numbered passages
(C1, C2, ...) carrying full text, plus a manifest with hit@k and MRR.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from kamiknows.common import (
    create_output_dir,
    read_jsonl,
    read_manifest,
    write_jsonl,
    write_manifest,
)

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*|[0-9]+(?:\.[0-9]+)?")
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "selected",
    "that",
    "the",
    "their",
    "this",
    "to",
    "used",
    "what",
    "when",
    "where",
    "which",
    "with",
}
ARXIV_REFERENCE = re.compile(
    r"(?<![\w./])(?:arxiv\s*:\s*|https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/)?"
    r"(?P<id>(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})(?:v[1-9]\d*)?)(?:\.pdf)?(?![\w]|\.\d)",
    re.IGNORECASE,
)
VERSION = re.compile(r"v[1-9]\d*$", re.IGNORECASE)


# Indexes


def tokenize(text: str) -> list[str]:
    tokens = [token.lower().strip("+-_") for token in TOKEN_RE.findall(text or "")]
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


class TfidfIndex:
    name = "tfidf"

    def __init__(
        self, chunks: list[dict[str, Any]], *, title_weight: int = 2, section_weight: int = 2
    ) -> None:
        documents = [
            tokenize(
                " ".join([c["title"]] * title_weight + [c.get("retrieval_section_heading", c["section_heading"])] * section_weight + [c.get("retrieval_text", c["text"])])
            )
            for c in chunks
        ]
        frequency = Counter(term for tokens in documents for term in set(tokens))
        self.idf = {
            term: math.log((len(chunks) + 1) / (count + 1)) + 1.0 for term, count in frequency.items()
        }
        self.vectors = [self._vector(tokens) for tokens in documents]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vector = {term: n / total * self.idf[term] for term, n in counts.items() if term in self.idf}
        norm = math.sqrt(sum(x * x for x in vector.values()))
        return {term: x / norm for term, x in vector.items()} if norm else {}

    def scores(self, query: str) -> list[float]:
        query_vector = self._vector(tokenize(query))
        return [sum(x * vector.get(term, 0.0) for term, x in query_vector.items()) for vector in self.vectors]

    def describe(self) -> dict[str, Any]:
        return {"vocabulary": len(self.idf)}


class DenseIndex:
    name = "dense"

    def __init__(self, chunks: list[dict[str, Any]], encoder) -> None:
        self.encoder = encoder
        self.vectors = encoder.encode(
            [f"{c['title']} {c.get('retrieval_section_heading', c['section_heading'])} {c.get('retrieval_text', c['text'])}" for c in chunks], kind="passage"
        )

    def scores(self, query: str) -> list[float]:
        query_vector = self.encoder.encode([query], kind="query")[0]
        return [sum(q * x for q, x in zip(query_vector, vector)) for vector in self.vectors]

    def describe(self) -> dict[str, Any]:
        return self.encoder.describe()


def build_index(retriever: str, chunks: list[dict[str, Any]], *, embedding_model: str, device: str):
    if retriever == "tfidf":
        return TfidfIndex(chunks)
    if retriever == "dense":
        from kamiknows.embeddings import Encoder

        return DenseIndex(chunks, Encoder(embedding_model, device=device))
    raise ValueError(f"unknown retriever: {retriever}")


# Explicit sources


def extract_arxiv_ids(text: str) -> list[str]:
    """arXiv ids written in the question, e.g. 'arXiv:1705.02355v2' or '2106.05285'.

    An unversioned id covers every version, so a versioned duplicate is dropped.
    """
    ids = list(dict.fromkeys(match.group("id").lower() for match in ARXIV_REFERENCE.finditer(text)))
    unversioned = {i for i in ids if not VERSION.search(i)}
    return [i for i in ids if not VERSION.search(i) or VERSION.sub("", i) not in unversioned]


def matches_source(arxiv_id: str, requested: str) -> bool:
    """Exact match for a versioned request, any version for an unversioned one."""
    arxiv_id = arxiv_id.lower()
    return arxiv_id == requested if VERSION.search(requested) else VERSION.sub("", arxiv_id) == requested


def retrieve(
    index, chunks: list[dict[str, Any]], question: str, *, top_k: int, source_policy: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the passages for one question and the source-constraint diagnostics."""
    requested = extract_arxiv_ids(question) if source_policy == "explicit_arxiv" else []
    candidates = [
        i
        for i, chunk in enumerate(chunks)
        if not requested or any(matches_source(chunk["arxiv_id"], r) for r in requested)
    ]
    missing = [r for r in requested if not any(matches_source(chunks[i]["arxiv_id"], r) for i in candidates)]
    constraint = {
        "policy": source_policy,
        "requested_arxiv_ids": requested,
        "missing_arxiv_ids": missing,
        "status": "ok",
    }
    if missing:
        constraint["status"] = "requested_source_unavailable"
    elif len(requested) > top_k:
        constraint["status"] = "too_many_requested_sources"
    if constraint["status"] != "ok":
        return [], constraint

    scores = index.scores(question)
    ranked = sorted(candidates, key=lambda i: (-scores[i], chunks[i]["chunk_id"]))
    selected = {next(i for i in ranked if matches_source(chunks[i]["arxiv_id"], r)) for r in requested}
    for i in ranked:
        if len(selected) >= top_k:
            break
        selected.add(i)
    passages = []
    for rank, i in enumerate((i for i in ranked if i in selected), start=1):
        chunk = chunks[i]
        passages.append(
            {
                "label": f"C{rank}",
                "rank": rank,
                "score": round(scores[i], 6),
                **{
                    key: chunk[key]
                    for key in ("chunk_id", "paper_id", "arxiv_id", "title", "section_heading", "text")
                },
                **{
                    key: chunk[key]
                    for key in (
                        "evidence_type",
                        "plot_provenance",
                        "axis_metadata",
                        "plot_reference",
                        "plot_sources",
                        "evidence_level",
                    )
                    if key in chunk
                },
            }
        )
    # A relevant panel identifies its parent figure. Attach that figure's
    # bounded numerical overview once, rather than requiring each sibling panel
    # to independently win a top-k slot. Ranking still uses the question alone.
    figure_overviews = {}
    for chunk in chunks:
        if chunk.get("evidence_level") == "figure":
            for source in chunk.get("plot_sources", []):
                figure_overviews[(chunk["paper_id"], source["asset_id"], source["series_id"])] = chunk
    expanded = {p["chunk_id"] for p in passages if p.get("evidence_level") == "figure"}
    for passage in passages:
        source = passage.get("plot_provenance", {})
        overview = figure_overviews.get(
            (passage["paper_id"], source.get("asset_id"), source.get("series_id"))
        )
        if overview is not None and overview["chunk_id"] not in expanded:
            expanded.add(overview["chunk_id"])
            passage["text"] += "\n\nRelated numerical context from the same figure:\n" + overview["text"]
            passage["plot_sources"] = overview["plot_sources"]
            passage["figure_context_chunk_id"] = overview["chunk_id"]
    return passages, constraint


# Evaluation


def evaluate(passages: list[dict[str, Any]], question: dict[str, Any]) -> dict[str, Any]:
    """Hit if an expected paper is retrieved (all of them when required_all_expected_sources)."""
    expected = [source.lower() for source in question.get("expected_arxiv_ids") or []]
    if not expected:
        return {"has_expected": False}
    first_rank = {
        source: next((p["rank"] for p in passages if matches_source(p["arxiv_id"], source)), None)
        for source in expected
    }
    found = [rank for rank in first_rank.values() if rank is not None]
    require_all = bool(question.get("required_all_expected_sources"))
    hit = len(found) == len(expected) if require_all else bool(found)
    hit_rank = (max(found) if require_all else min(found)) if hit else None
    return {
        "has_expected": True,
        "require_all": require_all,
        "hit": hit,
        "hit_rank": hit_rank,
        "reciprocal_rank": 1 / hit_rank if hit_rank else 0.0,
        "missing_expected": [source for source, rank in first_rank.items() if rank is None],
    }


def retrieval_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [r for r in records if r["evaluation"]["has_expected"]]
    hits = [r for r in evaluated if r["evaluation"]["hit"]]
    return {
        "questions": len(records),
        "questions_with_expected_sources": len(evaluated),
        "hits": len(hits),
        "hit_rate": round(len(hits) / len(evaluated), 4) if evaluated else None,
        "mrr": round(mean(r["evaluation"]["reciprocal_rank"] for r in evaluated), 4) if evaluated else None,
        "missed_question_ids": [r["question_id"] for r in evaluated if not r["evaluation"]["hit"]],
        "source_constraint_status": dict(Counter(r["source_constraint"]["status"] for r in records)),
    }


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = read_jsonl(path)
    if not questions:
        raise ValueError(f"no questions in {path}")
    for number, question in enumerate(questions, start=1):
        if (
            not str(question.get("question_id") or "").strip()
            or not str(question.get("question") or "").strip()
        ):
            raise ValueError(f"{path}:{number} needs question_id and question")
    ids = [q["question_id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question_id in {path}")
    return questions


# Step entry point


def run_retrieval(
    *,
    dataset_dir: Path,
    questions_file: Path,
    output_dir: Path,
    retriever: str,
    embedding_model: str,
    device: str,
    top_k: int,
    source_policy: str,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    dataset = read_manifest(dataset_dir, "chunk")
    chunks = read_jsonl(dataset_dir / "chunks.jsonl")
    questions = load_questions(questions_file)
    output_dir = create_output_dir(output_dir)
    print(f"Indexing {len(chunks)} chunks with {retriever}...", flush=True)
    index = build_index(retriever, chunks, embedding_model=embedding_model, device=device)
    records = []
    for question in questions:
        passages, constraint = retrieve(
            index, chunks, question["question"], top_k=top_k, source_policy=source_policy
        )
        evaluation = evaluate(passages, question)
        if question.get("expected_plot_sources"):
            expected_plots = question["expected_plot_sources"]
            present = {
                (p.get("plot_provenance", {}).get("asset_id"), p.get("plot_provenance", {}).get("series_id"))
                for p in passages
            }
            present.update(
                (source["asset_id"], source["series_id"])
                for p in passages
                for source in p.get("plot_sources", [])
            )
            missing = [e for e in expected_plots if (e["asset_id"], e["series_id"]) not in present]
            evaluation["plot_evidence"] = {"all_expected_retrieved": not missing, "missing": missing}
        records.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "annotations": {k: v for k, v in question.items() if k not in {"question_id", "question"}},
                "source_constraint": constraint,
                "passages": passages,
                "evaluation": evaluation,
            }
        )
    write_jsonl(output_dir / "contexts.jsonl", records)
    summary = {**retrieval_metrics(records), "index": index.describe()}
    plot_evaluations = [
        r["evaluation"]["plot_evidence"] for r in records if "plot_evidence" in r["evaluation"]
    ]
    if plot_evaluations:
        summary["plot_evidence"] = {
            "questions": len(plot_evaluations),
            "all_expected_retrieved": sum(e["all_expected_retrieved"] for e in plot_evaluations),
        }
        summary["plot_evidence_by_style"] = {}
        for style in sorted(
            {
                r["annotations"].get("question_style", "unspecified")
                for r in records
                if "plot_evidence" in r["evaluation"]
            }
        ):
            group = [
                r
                for r in records
                if "plot_evidence" in r["evaluation"]
                and r["annotations"].get("question_style", "unspecified") == style
            ]
            summary["plot_evidence_by_style"][style] = {
                "questions": len(group),
                "all_expected_retrieved": sum(
                    r["evaluation"]["plot_evidence"]["all_expected_retrieved"] for r in group
                ),
            }
    params = {
        "retriever": retriever,
        "top_k": top_k,
        "source_policy": source_policy,
        "chunker": dataset["params"]["chunker"],
        "dataset_dir": dataset_dir,
    }
    if retriever == "dense":
        params["embedding_model"] = embedding_model
    write_manifest(
        output_dir,
        step="retrieve",
        params=params,
        inputs={"chunks": dataset_dir / "chunks.jsonl", "questions": questions_file},
        summary=summary,
    )
    return summary
