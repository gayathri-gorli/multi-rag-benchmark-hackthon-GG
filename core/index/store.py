"""Hybrid chunk store: Chroma (HNSW dense) + in-memory BM25 (sparse), per practice area.
The embedding model id is persisted; opening with a different model raises so we never
mix vector spaces."""
import json
import re
from pathlib import Path

from core.config import EMBEDDING_MAX_TOKENS
from core.providers import embedding_model_id
from core.tokenizer import count_tokens

_TOKEN_RE = re.compile(r"\w+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _slug(area: str) -> str:
    return re.sub(r"\W+", "_", area).strip("_").lower()


class EmbeddingMismatch(RuntimeError):
    pass


class ChunkStore:
    def __init__(self, area: str, embeddings, root: str | Path = "storage"):
        self.area = area
        self.embeddings = embeddings
        self.dir = Path(root) / _slug(area)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.chunks_path = self.dir / "chunks.json"
        self._check_embedding_model()
        self.chunks: dict[str, dict] = (
            {c["metadata"]["chunk_id"]: c for c in json.loads(self.chunks_path.read_text())}
            if self.chunks_path.exists() else {}
        )
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._vs = None

    def _check_embedding_model(self):
        current = embedding_model_id()
        if self.meta_path.exists():
            stored = json.loads(self.meta_path.read_text()).get("embedding_model")
            if stored != current:
                raise EmbeddingMismatch(f"Index built with {stored}, current is {current}. Re-index required.")
        else:
            self.meta_path.write_text(json.dumps({"embedding_model": current}))

    @property
    def vs(self):
        if self._vs is None:
            from langchain_chroma import Chroma

            self._vs = Chroma(collection_name=f"c_{_slug(self.area)}", embedding_function=self.embeddings,
                              persist_directory=str(self.dir / "chroma"),
                              collection_metadata={"hnsw:space": "cosine"})
        return self._vs

    def add_chunks(self, chunks: list[dict]) -> dict:
        new = [c for c in chunks if c["metadata"]["chunk_id"] not in self.chunks]
        over = sum(1 for c in new if count_tokens(c["text"]) > EMBEDDING_MAX_TOKENS)
        if new:
            # Chroma rejects None metadata values.
            clean = lambda m: {k: v for k, v in m.items() if v is not None}
            self.vs.add_texts([c["text"] for c in new], metadatas=[clean(c["metadata"]) for c in new],
                              ids=[c["metadata"]["chunk_id"] for c in new])
            for c in new:
                self.chunks[c["metadata"]["chunk_id"]] = c
            self.chunks_path.write_text(json.dumps(list(self.chunks.values())))
            self._bm25 = None
        return {"added": len(new), "skipped_duplicates": len(chunks) - len(new),
                "over_embedding_limit": over}

    def vector_search(self, query_or_vec, k: int) -> list[str]:
        if isinstance(query_or_vec, str):
            res = self.vs.similarity_search(query_or_vec, k=k)
        else:
            res = self.vs.similarity_search_by_vector(query_or_vec, k=k)
        return [d.metadata["chunk_id"] for d in res]

    def bm25_search(self, query: str, k: int) -> list[str]:
        if not self.chunks:
            return []
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25_ids = list(self.chunks)
            self._bm25 = BM25Okapi([_tok(self.chunks[i]["text"]) for i in self._bm25_ids])
        scores = self._bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self._bm25_ids[i] for i in order if scores[i] > 0]

    def get(self, chunk_id: str) -> dict:
        return self.chunks[chunk_id]
