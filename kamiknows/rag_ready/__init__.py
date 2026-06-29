"""RAG-ready dataset v0 helpers for KamiKnows Fase 1G.

This package prepares validated, traceable dataset files for a future RAG
experiment. It does not create embeddings, a vector database, retrieval, or
generated answers.
"""

from kamiknows.rag_ready.build_dataset import build_rag_ready_dataset

__all__ = ["build_rag_ready_dataset"]
