"""Tests for full-text arXiv source helpers."""

from __future__ import annotations

from kamiknows.fulltext.arxiv_source import safe_paper_id


def test_safe_paper_id_normalizes_common_arxiv_forms() -> None:
    assert safe_paper_id("arXiv:1705.02355v2") == "1705_02355v2"
    assert safe_paper_id("https://arxiv.org/abs/1807.01954v2") == "1807_01954v2"
    assert safe_paper_id("hep-ph/9901001v1") == "hep_ph_9901001v1"
