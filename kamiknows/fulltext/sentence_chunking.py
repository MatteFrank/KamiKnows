"""Deterministic sentence groups over existing parsed sections, without downloads."""

import re
from typing import Any

from kamiknows.fulltext.chunking import count_words

ABBREVIATIONS = re.compile(r"(?:\b(?:e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Sec|Secs|Dr|Prof|vs)|\b[A-Z])\.$", re.I)


def split_sentences(text: str) -> list[str]:
    """Split normalized text with a conservative punctuation heuristic.

    This is not a linguistic parser. Keep common abbreviations and escaped
    LaTeX punctuation together; decimals have no whitespace after the dot.
    """
    text = " ".join(text.split())
    boundaries = []
    for match in re.finditer(r"[.!?][\"')\]]*\s+", text):
        prefix = text[:match.start() + 1]
        if prefix.endswith("\\.") or ABBREVIATIONS.search(prefix):
            continue
        boundaries.append(match.end())
    sentences = []
    start = 0
    for end in boundaries + [len(text)]:
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    return sentences


def generate_sentence_chunks(*, paper_id: str, arxiv_id: str, title: str,
                             sections: list[dict[str, Any]], target_words: int = 400,
                             overlap_words: int = 60) -> list[dict[str, Any]]:
    """Pack complete sentences; oversize sentences stay intact and are flagged.

    Size uses whitespace words, matching the original window construction.
    Overlap uses the largest suffix of whole sentences within the budget.
    Never retain the entire previous group, so every group makes progress.
    """
    if target_words < 100 or not 0 <= overlap_words < target_words:
        raise ValueError("target_words must be >=100 and 0 <= overlap_words < target_words")
    chunks = []
    for section in sections:
        sentences = split_sentences(str(section.get("text") or ""))
        lengths = [len(s.split()) for s in sentences]
        start = 0
        while start < len(sentences):
            end = start + 1
            size = lengths[start]
            while end < len(sentences) and size + lengths[end] <= target_words:
                size += lengths[end]
                end += 1
            text = " ".join(sentences[start:end])
            chunks.append({
                "chunk_id": f"{paper_id}::sentence_v1::{len(chunks)+1:04d}",
                "paper_id": paper_id, "arxiv_id": arxiv_id, "title": title,
                "section_id": section.get("section_id"),
                "section_heading": section.get("heading"), "source_type": "section",
                "text": text, "word_count": count_words(text),
                "sentence_span": {"start": start, "end_exclusive": end},
                "oversize_sentence": lengths[start] > target_words,
            })
            if end == len(sentences):
                break
            next_start, overlap = end, 0
            while next_start > start + 1 and overlap + lengths[next_start - 1] <= overlap_words:
                next_start -= 1
                overlap += lengths[next_start]
            start = next_start
    return chunks
