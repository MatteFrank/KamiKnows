"""Minimal arXiv metadata downloader for KamiKnows Fase 0.

This module intentionally retrieves only metadata, not PDFs, LaTeX sources,
chunks, embeddings, or model outputs. It is the smallest useful ingestion step:

arXiv query/id -> title, authors, abstract, categories, published date, URL

Network access is used only by the public fetch/search functions. XML parsing is
kept separate so it can be tested without calling arXiv.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode

import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivIngestionError(RuntimeError):
    """Raised when arXiv metadata cannot be retrieved or parsed."""


@dataclass(slots=True)
class ArxivPaperMetadata:
    """Minimal metadata used by early KamiKnows ingestion tests."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable dictionary."""
        return asdict(self)


def _collapse_whitespace(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize common arXiv ID forms for the API id_list parameter.

    Accepted inputs include:

    - "2301.00001"
    - "2301.00001v2"
    - "arXiv:2301.00001v2"
    - "https://arxiv.org/abs/2301.00001v2"
    """
    cleaned = arxiv_id.strip()
    if not cleaned:
        raise ValueError("arxiv_id must not be empty")

    cleaned = cleaned.removeprefix("arXiv:").removeprefix("arxiv:")
    if "/abs/" in cleaned:
        cleaned = cleaned.rsplit("/abs/", maxsplit=1)[-1]
    if "/pdf/" in cleaned:
        cleaned = cleaned.rsplit("/pdf/", maxsplit=1)[-1]
    cleaned = cleaned.removesuffix(".pdf")
    cleaned = cleaned.strip("/")

    if not cleaned:
        raise ValueError("arxiv_id must not be empty after normalization")
    return cleaned


def build_arxiv_query_url(
    *,
    query: str | None = None,
    arxiv_id: str | None = None,
    max_results: int = 1,
) -> str:
    """Build an arXiv API query URL for one ID or one search query."""
    if max_results < 1:
        raise ValueError("max_results must be >= 1")
    if bool(query) == bool(arxiv_id):
        raise ValueError("provide exactly one of query or arxiv_id")

    params: dict[str, str | int] = {
        "start": 0,
        "max_results": max_results,
    }
    if arxiv_id is not None:
        params["id_list"] = normalize_arxiv_id(arxiv_id)
    else:
        stripped_query = query.strip() if query is not None else ""
        if not stripped_query:
            raise ValueError("query must not be empty")
        params["search_query"] = stripped_query
        params["sortBy"] = "submittedDate"
        params["sortOrder"] = "descending"

    return f"{ARXIV_API_URL}?{urlencode(params)}"


def _entry_to_metadata(entry: ET.Element) -> ArxivPaperMetadata:
    entry_id = _collapse_whitespace(entry.findtext("atom:id", namespaces=ATOM_NS))
    if not entry_id:
        raise ArxivIngestionError("arXiv entry is missing atom:id")

    arxiv_id = normalize_arxiv_id(entry_id)
    title = _collapse_whitespace(entry.findtext("atom:title", namespaces=ATOM_NS))
    abstract = _collapse_whitespace(entry.findtext("atom:summary", namespaces=ATOM_NS))
    published = _collapse_whitespace(entry.findtext("atom:published", namespaces=ATOM_NS))

    authors = [
        _collapse_whitespace(author.findtext("atom:name", namespaces=ATOM_NS))
        for author in entry.findall("atom:author", namespaces=ATOM_NS)
    ]
    authors = [author for author in authors if author]

    categories = [
        category.attrib.get("term", "").strip()
        for category in entry.findall("atom:category", namespaces=ATOM_NS)
    ]
    categories = [category for category in categories if category]

    if not title:
        raise ArxivIngestionError(f"arXiv entry {arxiv_id} is missing title")
    if not abstract:
        raise ArxivIngestionError(f"arXiv entry {arxiv_id} is missing abstract")

    return ArxivPaperMetadata(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        categories=categories,
        published=published,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def parse_arxiv_atom_feed(xml_text: str) -> list[dict[str, Any]]:
    """Parse an arXiv Atom API response into minimal metadata dictionaries."""
    if not isinstance(xml_text, str) or not xml_text.strip():
        raise ArxivIngestionError("arXiv XML response is empty")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivIngestionError(f"invalid arXiv XML response: {exc}") from exc

    entries = root.findall("atom:entry", namespaces=ATOM_NS)
    if not entries:
        raise ArxivIngestionError("arXiv response did not contain any entries")

    return [_entry_to_metadata(entry).to_dict() for entry in entries]


def _get_arxiv_api_text(url: str, *, timeout_seconds: int) -> str:
    """Retrieve arXiv API text and wrap network errors with project errors."""
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ArxivIngestionError(f"arXiv request failed: {exc}") from exc
    return response.text


def fetch_arxiv_metadata_by_id(
    arxiv_id: str,
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Fetch minimal metadata for one arXiv ID."""
    url = build_arxiv_query_url(arxiv_id=arxiv_id, max_results=1)
    records = parse_arxiv_atom_feed(
        _get_arxiv_api_text(url, timeout_seconds=timeout_seconds)
    )
    return records[0]


def search_arxiv_metadata(
    query: str,
    *,
    max_results: int = 1,
    timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    """Fetch minimal metadata for a small arXiv search query."""
    url = build_arxiv_query_url(query=query, max_results=max_results)
    return parse_arxiv_atom_feed(
        _get_arxiv_api_text(url, timeout_seconds=timeout_seconds)
    )
