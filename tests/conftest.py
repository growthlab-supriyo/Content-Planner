import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEmbedder:
    """Bag-of-words vectors: keywords sharing words are similar. Deterministic, no model needed."""
    name = "fake"
    near_duplicate = 0.95
    cluster_distance = 0.5
    dim = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.split():
                w = {"price": "cost", "prices": "cost", "costs": "cost"}.get(w, w)
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


class FakeLLM:
    """Returns scripted answers in order; records prompts."""
    label = "fake · test"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.calls = 0

    def json(self, system, user, temperature=0.7):
        self.calls += 1
        self.prompts.append(user)
        a = self.answers.pop(0) if self.answers else {"queries": []}
        if isinstance(a, Exception):
            raise a
        return a(user) if callable(a) else a


@pytest.fixture
def embedder():
    return FakeEmbedder()
