"""Tests for full-text chunk generation."""

from __future__ import annotations

from kamiknows.fulltext.chunking import generate_chunks


def test_generate_chunks_creates_stable_ids_and_overlap() -> None:
    words = [f"word{i}" for i in range(260)]
    sections = [
        {
            "section_id": "sec_001",
            "heading": "Long section",
            "text": " ".join(words),
            "word_count": len(words),
            "source_type": "latex_source",
        }
    ]

    chunks = generate_chunks(
        paper_id="1705_02355v2",
        arxiv_id="1705.02355v2",
        title="A paper",
        sections=sections,
        target_words=120,
        overlap_words=30,
        min_tail_words=40,
    )

    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == "1705_02355v2_chunk_0001"
    assert chunks[1]["chunk_id"] == "1705_02355v2_chunk_0002"
    assert chunks[0]["section_heading"] == "Long section"
    assert chunks[0]["source_type"] == "section"
    assert chunks[0]["word_count"] == 120
