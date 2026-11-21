import hashlib

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (offline, 64-d)."""

    def _v(self, t):
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)


def fake_llm(fn):
    """LLM stand-in: fn(prompt_text) -> reply text. Records prompts in .calls."""
    calls = []

    def run(x):
        text = x if isinstance(x, str) else "\n".join(str(getattr(m, "content", m[1] if isinstance(m, tuple) else m)) for m in x)
        calls.append(text)
        return AIMessage(content=fn(text))

    llm = RunnableLambda(run)
    llm.calls = calls
    return llm
