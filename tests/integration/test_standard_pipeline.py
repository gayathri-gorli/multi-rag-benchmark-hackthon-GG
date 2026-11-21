import hashlib

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core.config import Hyperparams
from core.guardrails import PracticeArea
from core.index.store import ChunkStore, EmbeddingMismatch
from core.ingestion.chunker import chunk_records
from core.ingestion.loaders import load_file
from core.rag.standard import NODES, StandardRAG
from core.tokenizer import count_tokens


from tests.conftest import HashEmbeddings

DOC = (
    "Section 115BAA of the Income Tax Act provides a concessional corporate tax rate of 22 percent.\n\n"
    "Rule 36 of the CGST Rules restricts input tax credit to invoices reflected in GSTR-2B.\n\n"
    + "Filler regulatory text about unrelated matters. " * 300
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    f = tmp_path / "circular.txt"
    f.write_text(DOC)
    s = ChunkStore(PracticeArea.TAX.value, HashEmbeddings(), root=tmp_path / "storage")
    s.add_chunks(chunk_records(load_file(f, PracticeArea.TAX.value)))
    return s


def test_chunks_respect_600_token_limit_with_provenance(store):
    assert len(store.chunks) > 1
    for c in store.chunks.values():
        assert count_tokens(c["text"]) <= 600 + 5
        assert c["metadata"]["source"] == "circular.txt" and c["metadata"]["practice_area"]


def test_reingest_is_idempotent(store, tmp_path):
    n = len(store.chunks)
    res = store.add_chunks(list(store.chunks.values()))
    assert res["added"] == 0 and len(store.chunks) == n


def test_embedding_model_mismatch_refuses_to_open(store, tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "azure_openai")
    with pytest.raises(EmbeddingMismatch):
        ChunkStore(PracticeArea.TAX.value, HashEmbeddings(), root=tmp_path / "storage")


def test_standard_rag_end_to_end(store):
    llm = FakeListChatModel(responses=["Rate is 22 percent under Section 115BAA [S1]."])
    out = StandardRAG(store, llm, PracticeArea.TAX, Hyperparams(top_k=3)).run("Section 115BAA corporate tax rate")
    assert "115BAA" in out["answer"]
    assert out["citations"][0]["valid"] and out["citations"][0]["source"] == "circular.txt"
    assert out["guardrail_flags"] == []
    assert set(out["timings"]) >= set(NODES) and out["total_ms"] > 0
    assert any("115BAA" in c["text"] for c in out["chunks"])


def test_toggling_nodes_passes_through(store):
    llm = FakeListChatModel(responses=["See Rule 36 [S1]."])
    rag = StandardRAG(store, llm, PracticeArea.TAX, Hyperparams(top_k=2), enabled_nodes={"rerank": False})
    out = rag.run("Rule 36 input tax credit GSTR-2B")
    assert "rerank" in out["skipped"] and out["chunks"]  # synthesis falls back to fused order


def test_empty_corpus_returns_gap_disclosure_without_llm_call(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    s = ChunkStore(PracticeArea.GRC.value, HashEmbeddings(), root=tmp_path / "s")
    llm = FakeListChatModel(responses=["SHOULD NOT BE USED"])
    out = StandardRAG(s, llm, PracticeArea.GRC).run("KYC periodic updation")
    assert "Gap Disclosure" in out["answer"] and out["chunks"] == []
