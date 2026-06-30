"""Citation validation helpers for structured RAG answers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


VALID_STATUSES = {
    "valid",
    "missing_citations",
    "unknown_citation",
    "cites_unretrieved_chunk",
}


def validate_citations(
    answer_record: dict[str, Any],
    retrieved_chunk_ids: Iterable[str],
    *,
    known_chunk_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate that answer citations reference retrieved chunks only.

    ``known_chunk_ids`` is optional. When supplied, citation IDs outside that
    corpus are classified as ``unknown_citation`` before unretrieved citations.
    """
    retrieved = {str(chunk_id) for chunk_id in retrieved_chunk_ids if chunk_id}
    known = {str(chunk_id) for chunk_id in known_chunk_ids if chunk_id} if known_chunk_ids is not None else None
    citations = answer_record.get("citations") or []
    if not citations:
        return {
            "status": "missing_citations",
            "valid": False,
            "cited_chunk_ids": [],
            "retrieved_chunk_ids": sorted(retrieved),
            "unknown_citation_chunk_ids": [],
            "unretrieved_citation_chunk_ids": [],
            "notes": "Answer contains no citations.",
        }

    cited_chunk_ids: list[str] = []
    malformed_citations = 0
    for citation in citations:
        if not isinstance(citation, dict):
            malformed_citations += 1
            continue
        chunk_id = str(citation.get("chunk_id") or "").strip()
        if not chunk_id:
            malformed_citations += 1
            continue
        cited_chunk_ids.append(chunk_id)

    unknown = []
    if known is not None:
        unknown = sorted({chunk_id for chunk_id in cited_chunk_ids if chunk_id not in known})
    unretrieved = sorted(
        {
            chunk_id
            for chunk_id in cited_chunk_ids
            if chunk_id not in retrieved and (known is None or chunk_id in known)
        }
    )

    if malformed_citations or unknown:
        status = "unknown_citation"
    elif unretrieved:
        status = "cites_unretrieved_chunk"
    else:
        status = "valid"

    return {
        "status": status,
        "valid": status == "valid",
        "cited_chunk_ids": cited_chunk_ids,
        "retrieved_chunk_ids": sorted(retrieved),
        "unknown_citation_chunk_ids": unknown,
        "unretrieved_citation_chunk_ids": unretrieved,
        "notes": _status_note(status, malformed_citations),
    }


def _status_note(status: str, malformed_citations: int) -> str:
    if status == "valid":
        return "All citations reference retrieved chunks."
    if status == "missing_citations":
        return "Answer contains no citations."
    if status == "unknown_citation":
        if malformed_citations:
            return "At least one citation is malformed or missing chunk_id."
        return "At least one citation references an unknown chunk_id."
    if status == "cites_unretrieved_chunk":
        return "At least one citation references a chunk outside the retrieved context."
    return "Unknown citation validation status."
