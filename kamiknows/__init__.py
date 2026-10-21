"""KamiKnows: traceable RAG over scientific papers.

Pipeline steps, one module each:
    parsing     arXiv LaTeX source -> sections and equations
    chunking    sections -> chunks (word, sentence or semantic)
    retrieval   questions -> ranked passages (TF-IDF or dense)
    generation  passages -> cited answers from a local Ollama model
    review      answers -> manual review sheet
Shared helpers live in ``common`` (files, manifests) and ``embeddings``.
"""
