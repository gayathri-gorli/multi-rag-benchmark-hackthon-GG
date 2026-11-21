import json

import pytest

from core.config import Hyperparams
from core.guardrails import PracticeArea
from core.index.graph import GraphStore
from core.index.store import ChunkStore
from core.ingestion.chunker import chunk_records
from core.ingestion.entity_extract import extract_chunks
from core.ingestion.loaders import load_file
from core.local_embed import LocalBgeSmallEmbeddings, find_model_dir
from core.mock_embed import MockEmbeddings
from core.rag.factory import build_engine
from core.rag.impact import compare_runs
from core.rag.standard import EDGES
from core.samples import sample_files
from tests.conftest import fake_llm

TAX = PracticeArea.TAX
Q = "If a company opts for the 22 percent concessional regime, can it still claim MAT credit?"


def _store(tmp_path, monkeypatch, embeddings, provider):
    monkeypatch.setenv("EMBEDDING_PROVIDER", provider)
    store = ChunkStore(TAX.value, embeddings, root=tmp_path / "s")
    chunks = []
    for f in sample_files(TAX):
        chunks += chunk_records(load_file(f, TAX.value))
    store.add_chunks(chunks)
    return store, chunks


def test_sample_documents_exist_and_are_labelled_synthetic():
    for area in PracticeArea:
        files = sample_files(area)
        assert len(files) == 2
        assert all("SYNTHETIC SAMPLE" in f.read_text() for f in files)


@pytest.mark.skipif(find_model_dir() is None, reason="bge-small not cached")
def test_local_bge_retrieves_the_right_sample_document(tmp_path, monkeypatch):
    store, _ = _store(tmp_path, monkeypatch, LocalBgeSmallEmbeddings(), "local")
    top = store.vector_search(Q, 1)[0]
    assert store.get(top)["metadata"]["source"] == "sample_direct_tax_circular.md"


def test_mock_mode_end_to_end_runs_without_semantic_claims(tmp_path, monkeypatch):
    store, _ = _store(tmp_path, monkeypatch, MockEmbeddings(), "mock")
    assert store.meta_path.read_text().count("mock:hash-256") == 1  # index is tagged as mock
    llm = fake_llm(lambda p: "MAT credit lapses under Section 115ZA [S1].")
    out = build_engine("Standard RAG", store=store, llm=llm, area=TAX, hp=Hyperparams()).run(Q)
    assert out["answer"] and out["chunks"]


@pytest.fixture
def kg_env(tmp_path, monkeypatch):
    store, chunks = _store(tmp_path, monkeypatch, MockEmbeddings(), "mock")
    kg = GraphStore(tmp_path / "g.json")
    cid = {c["metadata"]["source"]: c["metadata"]["chunk_id"] for c in chunks}

    def extractor(p):
        if "Section 115ZA" in p:
            return json.dumps([{"subject": "CBDT", "relation": "clarifies", "object": "Section 115ZA"},
                               {"subject": "Section 115ZA", "relation": "denies", "object": "MAT credit"}])
        return "[]"
    extract_chunks(chunks, kg, fake_llm(extractor))
    return store, kg


def run(store, kg, engine="Standard RAG", **kw):
    return build_engine(engine, store=store, llm=fake_llm(lambda p: "ok [S1]"), area=TAX,
                        hp=Hyperparams(top_k=3), graph=kg, **kw).run(Q, dry_run=True)


def test_dry_run_makes_no_synthesis_call(kg_env):
    store, kg = kg_env
    llm = fake_llm(lambda p: "should not be called")
    out = build_engine("Standard RAG", store=store, llm=llm, area=TAX, hp=Hyperparams()).run(Q, dry_run=True)
    assert llm.calls == [] and out["chunks"] and "dry run" in out["answer"]


def test_edge_hybrid_to_fusion_severed_leaves_nothing_for_synthesis(kg_env):
    store, kg = kg_env
    llm = fake_llm(lambda p: "must not be called")
    out = build_engine("Standard RAG", store=store, llm=llm, area=TAX, hp=Hyperparams(),
                       enabled_edges={"hybrid_search->rrf_fusion": False}).run(Q)
    assert out["severed"] == ["hybrid_search->rrf_fusion"] and out["chunks"] == []
    assert "Gap Disclosure" in out["answer"] and llm.calls == []
    # Every node still ran (only the data transition was cut).
    assert set(out["timings"]) >= {"rrf_fusion", "rerank", "synthesis"} and out["skipped"] == []


def test_edge_fusion_to_rerank_severed_changes_ordering_only(kg_env):
    store, kg = kg_env
    r = compare_runs(run(store, kg), run(store, kg, enabled_edges={"rrf_fusion->rerank": False}))
    assert r["severed_edges"][0][0] == "rrf_fusion->rerank" and "fused" in r["withheld_fields"]
    assert not r["empty"] and r["overlap"] > 0


def test_edge_rerank_to_synthesis_severed_drops_kg_evidence_for_graph_rag(kg_env):
    store, kg = kg_env
    base = run(store, kg, "Graph RAG")
    cut = run(store, kg, "Graph RAG", enabled_edges={"rerank->synthesis": False})
    assert "kg" in [c["metadata"]["chunk_id"] for c in base["chunks"]]
    assert "kg" not in [c["metadata"]["chunk_id"] for c in cut["chunks"]]
    assert "kg" in compare_runs(base, cut)["dropped"]


def test_edge_expansion_to_search_severed_removes_graph_seeds(kg_env):
    store, kg = kg_env
    base = run(store, kg, "Graph RAG")
    cut = run(store, kg, "Graph RAG", enabled_edges={"query_expansion->hybrid_search": False})
    assert base["graph_nodes"] and not cut.get("kg_paths") and "seeds" in cut["blocked"]
    assert len(cut["ranked_lists"]) < len(base["ranked_lists"])  # the graph list is gone


def test_all_edges_enabled_matches_baseline(kg_env):
    store, kg = kg_env
    r = compare_runs(run(store, kg), run(store, kg, enabled_edges={e: True for e in EDGES}))
    assert r["overlap"] == 1.0 and not r["order_changed"] and r["severed_edges"] == []
