"""Simple word-based chunking for Fase 1F full-text outputs."""

from __future__ import annotations

import re
from typing import Any

WORD_RE = re.compile(r"\b[\w'-]+\b")


def count_words(text: str) -> int:
    """Count simple word-like tokens."""
    return len(WORD_RE.findall(text or ""))


def _word_windows(
    words: list[str],
    *,
    target_words: int,
    overlap_words: int,
    min_tail_words: int,
) -> list[list[str]]:
    if len(words) <= target_words:
        return [words] if words else []

    windows: list[list[str]] = []
    start = 0
    step = max(1, target_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + target_words)
        window = words[start:end]
        remaining = len(words) - end
        if remaining and remaining < min_tail_words:
            window = words[start:]
            windows.append(window)
            break
        windows.append(window)
        if end >= len(words):
            break
        start += step
    return windows


def generate_chunks(
    *,
    paper_id: str,
    arxiv_id: str,
    title: str,
    sections: list[dict[str, Any]],
    target_words: int = 400,
    overlap_words: int = 60,
    min_tail_words: int = 80,
) -> list[dict[str, Any]]:
    """Generate stable section chunks from parsed section records."""
    if target_words < 100:
        raise ValueError("target_words must be >= 100")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words")

    chunks: list[dict[str, Any]] = []
    for section in sections:
        text = str(section.get("text") or "").strip()
        if not text:
            continue
        words = text.split()
        for window in _word_windows(
            words,
            target_words=target_words,
            overlap_words=overlap_words,
            min_tail_words=min_tail_words,
        ):
            chunk_index = len(chunks) + 1
            chunk_text = " ".join(window).strip()
            chunks.append(
                {
                    "chunk_id": f"{paper_id}_chunk_{chunk_index:04d}",
                    "paper_id": paper_id,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "section_id": section.get("section_id"),
                    "section_heading": section.get("heading"),
                    "source_type": "section",
                    "text": chunk_text,
                    "word_count": count_words(chunk_text),
                }
            )
    return chunks
