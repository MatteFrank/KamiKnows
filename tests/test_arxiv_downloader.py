"""Tests for the minimal arXiv metadata ingestion module."""

from __future__ import annotations

import pytest

from kamiknows.ingestion.arxiv_downloader import (
    ArxivIngestionError,
    build_arxiv_query_url,
    normalize_arxiv_id,
    parse_arxiv_atom_feed,
    search_arxiv_metadata,
)

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <updated>2023-01-01T00:00:00Z</updated>
    <published>2023-01-01T00:00:00Z</published>
    <title>Fast calorimeter simulation for high energy physics</title>
    <summary>
      We present a lightweight method for fast calorimeter response simulation
      in high energy physics. The method uses parameterized shower shapes.
    </summary>
    <author><name>Ada Example</name></author>
    <author><name>Bruno Example</name></author>
    <category term="hep-ex" scheme="http://arxiv.org/schemas/atom" />
    <category term="physics.ins-det" scheme="http://arxiv.org/schemas/atom" />
  </entry>
</feed>
"""


def test_normalize_arxiv_id_accepts_common_forms() -> None:
    assert normalize_arxiv_id("2301.00001") == "2301.00001"
    assert normalize_arxiv_id("arXiv:2301.00001v2") == "2301.00001v2"
    assert normalize_arxiv_id("https://arxiv.org/abs/2301.00001v2") == "2301.00001v2"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2301.00001v2.pdf") == "2301.00001v2"


def test_build_arxiv_query_url_for_id() -> None:
    url = build_arxiv_query_url(arxiv_id="arXiv:2301.00001v1")

    assert url.startswith("https://export.arxiv.org/api/query?")
    assert "id_list=2301.00001v1" in url
    assert "max_results=1" in url


def test_build_arxiv_query_url_for_query() -> None:
    url = build_arxiv_query_url(query="cat:hep-ex AND calorimeter", max_results=2)

    assert "search_query=cat%3Ahep-ex+AND+calorimeter" in url
    assert "max_results=2" in url
    assert "sortBy=submittedDate" in url


def test_build_arxiv_query_url_requires_one_source() -> None:
    with pytest.raises(ValueError):
        build_arxiv_query_url()
    with pytest.raises(ValueError):
        build_arxiv_query_url(query="cat:hep-ex", arxiv_id="2301.00001")


def test_parse_arxiv_atom_feed_returns_minimal_metadata() -> None:
    records = parse_arxiv_atom_feed(SAMPLE_ARXIV_XML)

    assert len(records) == 1
    record = records[0]
    assert record == {
        "arxiv_id": "2301.00001v1",
        "title": "Fast calorimeter simulation for high energy physics",
        "authors": ["Ada Example", "Bruno Example"],
        "abstract": (
            "We present a lightweight method for fast calorimeter response "
            "simulation in high energy physics. The method uses parameterized "
            "shower shapes."
        ),
        "categories": ["hep-ex", "physics.ins-det"],
        "published": "2023-01-01T00:00:00Z",
        "url": "https://arxiv.org/abs/2301.00001v1",
    }


def test_parse_arxiv_atom_feed_rejects_empty_feed() -> None:
    empty_feed = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    with pytest.raises(ArxivIngestionError):
        parse_arxiv_atom_feed(empty_feed)


def test_search_arxiv_metadata_wraps_request_errors(monkeypatch) -> None:
    import requests

    def fake_get(*args, **kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr("kamiknows.ingestion.arxiv_downloader.requests.get", fake_get)

    try:
        search_arxiv_metadata("cat:hep-ex", max_results=1)
    except ArxivIngestionError as exc:
        assert "arXiv request failed" in str(exc)
    else:
        raise AssertionError("expected ArxivIngestionError")
