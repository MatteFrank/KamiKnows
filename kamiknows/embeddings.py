"""Sentence-transformers encoder shared by the semantic chunker and dense retrieval.

Models are loaded from the local Hugging Face cache only (no downloads at run
time). Texts longer than the model's maximum sequence length are not
truncated: their token ids are split into consecutive windows, each window is
encoded, and the normalized window vectors are averaged weighted by length.
"""

from __future__ import annotations

import math
from typing import Any

# Only BERT-style models: the long-text windows are built with [CLS] ... [SEP].
EMBEDDING_MODELS: dict[str, dict[str, str]] = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": "", "passage_prefix": "",
    },
    "e5-small-v2": {
        "name": "intfloat/e5-small-v2", "query_prefix": "query: ", "passage_prefix": "passage: ",
    },
}


class Encoder:
    def __init__(self, model_key: str, *, device: str = "cpu") -> None:
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f"unknown embedding model {model_key!r}; choose from {sorted(EMBEDDING_MODELS)}")
        from sentence_transformers import SentenceTransformer

        self.key = model_key
        self.config = EMBEDDING_MODELS[model_key]
        self.device = device
        try:
            self.model = SentenceTransformer(self.config["name"], device=device, local_files_only=True)
        except OSError as exc:
            raise RuntimeError(f"{self.config['name']} is not in the local Hugging Face cache: {exc}") from exc
        self.tokenizer = self.model.tokenizer
        self.window_tokens = self.model.max_seq_length - 2
        self.windowed_texts = 0
        probe = self.tokenizer("", add_special_tokens=True, truncation=False)["input_ids"]
        if probe != [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
            raise ValueError(f"{self.config['name']} does not use a [CLS] text [SEP] template")

    def describe(self) -> dict[str, Any]:
        return {"model": self.config["name"], "device": self.device,
                "max_seq_length": self.model.max_seq_length, "windowed_texts": self.windowed_texts}

    def encode(self, texts: list[str], *, kind: str) -> list[list[float]]:
        """Return one unit-norm vector per text; ``kind`` is 'query' or 'passage'."""
        prefix = self.config[f"{kind}_prefix"]
        texts = [prefix + text for text in texts]
        token_ids = [
            self.tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            for text in texts
        ]
        vectors: list[list[float] | None] = [None] * len(texts)
        short = [i for i, ids in enumerate(token_ids) if len(ids) <= self.window_tokens]
        if short:
            encoded = self.model.encode([texts[i] for i in short], normalize_embeddings=True,
                                        show_progress_bar=False)
            for i, vector in zip(short, encoded):
                vectors[i] = [float(x) for x in vector]
        for i, ids in enumerate(token_ids):
            if vectors[i] is None:
                vectors[i] = self._encode_windows(ids)
                self.windowed_texts += 1
        return vectors  # type: ignore[return-value]

    def _encode_windows(self, token_ids: list[int]) -> list[float]:
        import torch

        windows = [token_ids[i:i + self.window_tokens] for i in range(0, len(token_ids), self.window_tokens)]
        rows = []
        for window in windows:
            ids = [self.tokenizer.cls_token_id, *window, self.tokenizer.sep_token_id]
            row = {"input_ids": ids, "attention_mask": [1] * len(ids)}
            if "token_type_ids" in self.tokenizer.model_input_names:
                row["token_type_ids"] = [0] * len(ids)
            rows.append(row)
        features = self.tokenizer.pad(rows, padding=True, return_tensors="pt")
        features = {key: value.to(self.model.device) for key, value in features.items()}
        self.model.eval()
        with torch.no_grad():
            window_vectors = self.model(features)["sentence_embedding"].cpu().tolist()
        combined = [0.0] * len(window_vectors[0])
        for vector, window in zip(window_vectors, windows):
            norm = math.sqrt(sum(x * x for x in vector))
            for j, x in enumerate(vector):
                combined[j] += x / norm * len(window) / len(token_ids)
        norm = math.sqrt(sum(x * x for x in combined))
        return [x / norm for x in combined]
