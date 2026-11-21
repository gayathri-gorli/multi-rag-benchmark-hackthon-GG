"""MOCK embeddings for UI/workflow testing only. Hashed bag of word unigrams+bigrams: it captures
word overlap, NOT meaning. Never use for evaluating retrieval quality or KPI numbers."""
import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

MOCK_LABEL = "MOCK EMBEDDINGS (lexical hashing, not semantic)"


class MockEmbeddings(Embeddings):
    dim = 256

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"\w+", text.lower())
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 100) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)
