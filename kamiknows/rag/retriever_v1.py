"""Deterministic TF-IDF retriever v1 for retrieval-baseline evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

TOKEN_RE_V1 = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*|[0-9]+(?:\.[0-9]+)?")

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

QUESTION_TYPE_EXPANSIONS = {
    "method": ("method", "model", "approach", "architecture", "training"),
    "limitation": ("limitation", "validation", "caveat", "uncertainty", "future"),
    "dataset": ("dataset", "data", "sample", "benchmark", "events"),
    "metric": ("metric", "performance", "accuracy", "speed", "validation"),
    "result": ("result", "results", "performance", "comparison", "evaluation"),
}


def tokenize_v1(text: str) -> list[str]:
    """Tokenize text for v1 retrieval, removing high-frequency question glue."""
    tokens = [token.lower().strip("+-_") for token in TOKEN_RE_V1.findall(text or "")]
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


@dataclass(frozen=True, slots=True)
class RetrievedChunkV1:
    """One retrieved chunk with score and rank."""

    rank: int
    score: float
    chunk: dict[str, Any]

    def to_baseline_dict(self) -> dict[str, Any]:
        text = str(self.chunk.get("text") or "")
        preview = " ".join(text.split())[:280]
        return {
            "rank": self.rank,
            "score": round(self.score, 8),
            "chunk_id": self.chunk.get("chunk_id", ""),
            "arxiv_id": self.chunk.get("arxiv_id", ""),
            "paper_id": self.chunk.get("paper_id", ""),
            "title": self.chunk.get("title", ""),
            "section_heading": self.chunk.get("section_heading", ""),
            "text_preview": preview,
        }


class TfidfRetrieverV1:
    """Small local TF-IDF retriever with deterministic v1 scoring tweaks."""

    name = "tfidf_local_v1"
    version = "retriever_v1"

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        title_weight: int = 2,
        section_weight: int = 2,
        use_question_type_expansion: bool = True,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        if title_weight < 1 or section_weight < 1:
            raise ValueError("title_weight and section_weight must be >= 1")
        self.chunks = list(chunks)
        self.title_weight = title_weight
        self.section_weight = section_weight
        self.use_question_type_expansion = use_question_type_expansion
        self._doc_vectors: list[dict[str, float]] = []
        self._doc_norms: list[float] = []
        self._idf: dict[str, float] = {}
        self._fit()

    def _weighted_chunk_text(self, chunk: dict[str, Any]) -> str:
        title = str(chunk.get("title") or "")
        section = str(chunk.get("section_heading") or "")
        text = str(chunk.get("text") or "")
        return " ".join(
            [title] * self.title_weight
            + [section] * self.section_weight
            + [text]
        )

    def _fit(self) -> None:
        tokenized = [tokenize_v1(self._weighted_chunk_text(chunk)) for chunk in self.chunks]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        doc_count = len(self.chunks)
        self._idf = {
            term: math.log((doc_count + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }

        for tokens in tokenized:
            vector = self._tfidf_vector(tokens)
            self._doc_vectors.append(vector)
            self._doc_norms.append(_norm(vector))

    def build_query_terms(
        self,
        question: str,
        *,
        question_type: str = "",
        expected_section_keywords: list[str] | None = None,
    ) -> list[str]:
        """Build deterministic query terms for diagnostics and scoring."""
        terms = tokenize_v1(question)
        if self.use_question_type_expansion:
            terms.extend(QUESTION_TYPE_EXPANSIONS.get(question_type, ()))
        for keyword in expected_section_keywords or []:
            terms.extend(tokenize_v1(str(keyword)))
        return _dedupe_preserve_order(terms)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        question_type: str = "",
        expected_section_keywords: list[str] | None = None,
    ) -> list[RetrievedChunkV1]:
        """Return top-k chunks with cosine similarity scores."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_terms = self.build_query_terms(
            question,
            question_type=question_type,
            expected_section_keywords=expected_section_keywords,
        )
        query_vector = self._tfidf_vector(query_terms)
        query_norm = _norm(query_vector)

        scored: list[tuple[float, str, int, dict[str, Any]]] = []
        for index, chunk in enumerate(self.chunks):
            score = _cosine(query_vector, query_norm, self._doc_vectors[index], self._doc_norms[index])
            scored.append((score, str(chunk.get("chunk_id") or ""), index, chunk))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            RetrievedChunkV1(rank=rank, score=score, chunk=chunk)
            for rank, (score, _chunk_id, _index, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {
            term: (count / total) * self._idf.get(term, 0.0)
            for term, count in counts.items()
            if term in self._idf
        }


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(
    left: dict[str, float],
    left_norm: float,
    right: dict[str, float],
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    return dot / (left_norm * right_norm)
