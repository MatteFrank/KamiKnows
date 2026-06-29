"""Minimal full-text parsing helpers for KamiKnows Fase 1F.

This package keeps full-text parsing separate from metadata ingestion,
abstract extraction, model backends, and future RAG/chunk retrieval work.
"""

from kamiknows.fulltext.arxiv_source import safe_paper_id

__all__ = ["safe_paper_id"]
