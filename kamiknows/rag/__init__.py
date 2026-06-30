"""Mini-RAG v0 helpers for KamiKnows Fase 1H.

This package implements a small local retrieval plus optional grounded
generation demo over the RAG-ready dataset. It is not a robust scientific QA
system and does not create embeddings, a vector DB, fine-tuning data, or
discovery claims.
"""

from kamiknows.rag.retriever import TfidfRetriever
from kamiknows.rag.embedding_selection import EmbeddingEncoder
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1

__all__ = ["EmbeddingEncoder", "TfidfRetriever", "TfidfRetrieverV1"]
