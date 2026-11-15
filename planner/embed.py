"""Embeddings for near-duplicate removal and clustering.

Local: nomic-embed-text through Ollama. Online: fastembed (small model that runs inside the app).
Both return L2-normalised vectors, so a dot product is the cosine similarity.
"""
from __future__ import annotations

import numpy as np

from .config import FASTEMBED_MODEL, OLLAMA_EMBED_MODEL, PROVIDERS
from .llm import HTTPError, LLMError, post_json


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    return v / n


class OllamaEmbedder:
    name = f"Ollama · {OLLAMA_EMBED_MODEL}"
    # Similarity above which two keywords count as the same search (e.g. "braces cost" / "braces price").
    near_duplicate = 0.93
    # Distance (1 - similarity) under which keywords join one cluster.
    cluster_distance = 0.16

    def __init__(self, model: str = OLLAMA_EMBED_MODEL):
        self.model = model
        self._url = PROVIDERS["Local Ollama"].base_url.rstrip("/") + "/embeddings"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        try:
            for i in range(0, len(texts), 64):
                batch = [f"clustering: {t}" for t in texts[i:i + 64]]  # nomic task prefix
                r = post_json(self._url, {"model": self.model, "input": batch}, None, 180)
                out.extend(d["embedding"] for d in sorted(r["data"], key=lambda d: d.get("index", 0)))
        except HTTPError as err:
            if err.status == 404 or "not found" in err.message:
                raise LLMError(f"Embedding model '{self.model}' is not installed. Run: ollama pull {self.model}") from None
            raise LLMError("Could not reach Ollama for embeddings. Is Ollama running?") from None
        return _normalise(np.array(out, dtype=np.float32))


class FastEmbedder:
    name = f"fastembed · {FASTEMBED_MODEL}"
    near_duplicate = 0.94
    cluster_distance = 0.17

    def __init__(self, model: str = FASTEMBED_MODEL):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(list(self._model.embed(texts)), dtype=np.float32))
