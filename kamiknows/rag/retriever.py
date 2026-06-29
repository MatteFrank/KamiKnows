"""A tiny deterministic TF-IDF retriever for mini-RAG v0."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for the local retriever."""
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved chunk with score and rank."""

    rank: int
    score: float
    chunk: dict[str, Any]

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk.get("chunk_id", ""),
            "paper_id": self.chunk.get("paper_id", ""),
            "arxiv_id": self.chunk.get("arxiv_id", ""),
            "title": self.chunk.get("title", ""),
            "section_id": self.chunk.get("section_id", ""),
            "section_heading": self.chunk.get("section_heading", ""),
            "source_type": self.chunk.get("source_type", ""),
            "text": self.chunk.get("text", ""),
        }


class TfidfRetriever:
    """Small in-memory TF-IDF/cosine retriever over chunk records."""

    name = "tfidf_local_v0"
    version = "retriever_v0"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        self.chunks = list(chunks)
        self._doc_vectors: list[dict[str, float]] = []
        self._doc_norms: list[float] = []
        self._idf: dict[str, float] = {}
        self._fit()

    def _chunk_text(self, chunk: dict[str, Any]) -> str:
        return " ".join(
            [
                str(chunk.get("title") or ""),
                str(chunk.get("section_heading") or ""),
                str(chunk.get("text") or ""),
            ]
        )

    def _fit(self) -> None:
        tokenized = [tokenize(self._chunk_text(chunk)) for chunk in self.chunks]
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

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {
            term: (count / total) * self._idf.get(term, 0.0)
            for term, count in counts.items()
        }

    def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Return top-k chunks with cosine similarity scores."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_vector = self._tfidf_vector(tokenize(question))
        query_norm = _norm(query_vector)

        scored: list[tuple[float, str, int, dict[str, Any]]] = []
        for index, chunk in enumerate(self.chunks):
            score = _cosine(query_vector, query_norm, self._doc_vectors[index], self._doc_norms[index])
            scored.append((score, str(chunk.get("chunk_id") or ""), index, chunk))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            RetrievedChunk(rank=rank, score=score, chunk=chunk)
            for rank, (score, _chunk_id, _index, chunk) in enumerate(scored[:top_k], start=1)
        ]


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
