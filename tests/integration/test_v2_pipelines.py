import json

import pytest

from core.config import Hyperparams
from core.graph_viz import communities, sphere_layout
from core.guardrails import PracticeArea
from core.index.graph import GraphStore
from core.index.node2vec import node2vec_embeddings
from core.index.store import ChunkStore
from core.ingestion.entity_extract import extract_chunks, parse_triples
from core.ingestion.normalize import aliases_in, node_key
from core.rag.factory import build_engine
from core.rag.hyde import HyDERAG
from tests.conftest import HashEmbeddings, fake_llm

AREA = PracticeArea.TAX
CHUNKS = [
    {"text": "The Income Tax Department issued Circular 12 clarifying Section 115BAA.",
     "metadata": {"source": "c1.txt", "page": 1, "chunk_id": "a1", "practice_area": AREA.value}},
    {"text": "Section 115BAA grants a 22 percent concessional rate but denies MAT credit.",
     "metadata": {"source": "c2.txt", "page": 1, "chunk_id": "a2", "practice_area": AREA.value}},
    {"text": "MAT credit is governed by Section 115JAA and may be carried forward 15 years.",
     "metadata": {"source": "c3.txt", "page": 2, "chunk_id": "a3", "practice_area": AREA.value}},
    {"text": "Unrelated: office canteen policy and parking allocation.",
     "metadata": {"source": "c4.txt", "page": 1, "chunk_id": "a4", "practice_area": AREA.value}},
]
TRIPLES = {
    "Circular 12": [{"subject": "Income Tax Dept", "relation": "issues", "object": "Circular 12"},
                    {"subject": "Circular 12", "relation": "clarifies", "object": "Section 115BAA"}],
    "22 percent": [{"subject": "Section 115BAA", "relation": "denies", "object": "MAT credit"}],
    "MAT credit is governed": [{"subject": "MAT credit", "relation": "governed by", "object": "Section 115JAA"}],
}


def extractor(prompt):
    for k, v in TRIPLES.items():
        if k in prompt:
            return "Here you go:\n" + json.dumps(v)
    return "[]"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openrouter")
    store = ChunkStore(AREA.value, HashEmbeddings(), root=tmp_path / "s")
    store.add_chunks(CHUNKS)
    kg = GraphStore(tmp_path / "s" / "graph.json")
    return store, kg


def test_alias_normalization_merges_synonyms():
    assert node_key("Income Tax Dept") == node_key("CBDT") == node_key("Tax Authority")
    assert aliases_in("What did the Income Tax Department say?") == ["CBDT"]


def test_parse_triples_tolerates_prose_and_garbage():
    assert len(parse_triples('noise [{"subject":"a","relation":"r","object":"b"}] tail')) == 1
    assert parse_triples("no json") == [] and parse_triples("[broken") == []


def test_extraction_builds_graph_caches_and_persists(env, tmp_path):
    store, kg = env
    llm = fake_llm(extractor)
    res = extract_chunks(CHUNKS, kg, llm)
    assert res["extracted"] == 4 and res["nodes"] >= 5 and len(llm.calls) == 4
    assert node_key("Income Tax Dept") in kg.G  # alias resolved to CBDT node
    extract_chunks(CHUNKS, kg, llm)  # cached: no new LLM calls
    assert len(llm.calls) == 4
    assert GraphStore(tmp_path / "s" / "graph.json").G.number_of_edges() == kg.G.number_of_edges()


def test_khop_expansion_respects_k_and_filters(env):
    _, kg = env
    extract_chunks(CHUNKS, kg, fake_llm(extractor))
    seed = [node_key("Section 115BAA")]
    d1, d2 = kg.expand(seed, 1), kg.expand(seed, 2)
    assert node_key("MAT credit") in d1 and node_key("Section 115JAA") not in d1
    assert node_key("Section 115JAA") in d2 and d2[node_key("Section 115JAA")] == 2
    # Edge filter (Tab 2): allow only the 'denies' edge -> cannot reach 115JAA.
    only = {(u, v, k) for u, v, k, r in kg.edge_list() if r == "denies"}
    assert node_key("Section 115JAA") not in kg.expand(seed, 2, allowed_edges=only)


def test_node2vec_shapes_and_determinism(env):
    _, kg = env
    extract_chunks(CHUNKS, kg, fake_llm(extractor))
    adj = {n: set(kg.G.successors(n)) | set(kg.G.predecessors(n)) for n in kg.G.nodes}
    a, b = node2vec_embeddings(adj, dim=4), node2vec_embeddings(adj, dim=4)
    assert a == b and set(a) == set(adj) and len(next(iter(a.values()))) <= 4


def test_sphere_layout_points_on_unit_sphere(env):
    _, kg = env
    extract_chunks(CHUNKS, kg, fake_llm(extractor))
    pos = sphere_layout(kg.G)
    assert set(pos) == set(kg.G.nodes)
    assert all(abs(x * x + y * y + z * z - 1) < 1e-9 for x, y, z in pos.values())
    assert set(communities(kg.G)) == set(kg.G.nodes)


def test_hyde_generates_n_passages_and_answers_original_question(env):
    store, _ = env

    def llm_fn(prompt):
        if "Write a concise passage" in prompt:
            return "Section 115BAA concessional rate 22 percent MAT credit"
        return "Rate is 22 percent per Section 115BAA [S1]."

    llm = fake_llm(llm_fn)
    rag = HyDERAG(store, llm, AREA, Hyperparams(hyde_n=3, top_k=2))
    out = rag.run("What is the corporate tax rate?")
    passages = [c for c in llm.calls if "Write a concise passage" in c]
    assert len(passages) == 3
    assert len({c.split("Write a concise passage")[1].split("that would")[0] for c in passages}) == 3  # distinct angles
    assert "Question: What is the corporate tax rate?" in llm.calls[-1]  # synthesis uses original query
    assert out["citations"] and not out["guardrail_flags"]
    assert any("115BAA" in c["text"] for c in out["chunks"])


def test_hyde_without_expansion_degrades_to_standard(env):
    store, _ = env
    llm = fake_llm(lambda p: "See Section 115BAA [S1].")
    HyDERAG(store, llm, AREA, enabled_nodes={"query_expansion": False}).run("Section 115BAA")
    assert len(llm.calls) == 1  # only synthesis; no hypothetical passages generated


def test_graph_rag_multihop_pulls_chunk_vector_search_misses(env):
    store, kg = env
    extract_chunks(CHUNKS, kg, fake_llm(extractor))
    llm = fake_llm(lambda p: "115BAA denies MAT credit, governed by Section 115JAA [S1][S2].")
    # Query names only the authority and circular; MAT credit is 2 hops away, Section 115JAA 3.
    q = "What does the Income Tax Department's Circular 12 say?"
    out = build_engine("Graph RAG", store=store, llm=llm, area=AREA, hp=Hyperparams(top_k=3, hop_k=2), graph=kg).run(q)
    assert set(out["seeds"]) == {node_key("CBDT"), node_key("Circular 12")}
    assert out["graph_nodes"][node_key("MAT credit")] == 2 and node_key("Section 115JAA") not in out["graph_nodes"]
    assert "a3" in out["ranked_lists"][-1]  # reached only by graph traversal (chunk shares no words with the query)
    assert "kg" in [c["metadata"]["chunk_id"] for c in out["chunks"]]  # KG evidence is a citable source
    assert any("--[denies]-->" in p for p in out["kg_paths"])
    assert "graph_traversal" in out["timings"]


def test_graph_rag_hop_radius_is_a_hyperparameter(env):
    store, kg = env
    extract_chunks(CHUNKS, kg, fake_llm(extractor))
    llm = fake_llm(lambda p: "ok [S1]")
    out = build_engine("Graph RAG", store=store, llm=llm, area=AREA, hp=Hyperparams(hop_k=1), graph=kg).run("Section 115BAA")
    assert node_key("Section 115JAA") not in out["graph_nodes"]


def test_graph_rag_requires_graph(env):
    store, _ = env
    with pytest.raises(ValueError):
        build_engine("Graph RAG", store=store, llm=fake_llm(lambda p: ""), area=AREA, hp=Hyperparams())


def test_rrf_toggle_off_falls_back_to_interleave(env):
    store, _ = env
    llm = fake_llm(lambda p: "x [S1]")
    out = build_engine("Standard RAG", store=store, llm=llm, area=AREA, hp=Hyperparams(top_k=2),
                       enabled_nodes={"rrf_fusion": False}).run("Section 115BAA MAT credit")
    assert "rrf_fusion" in out["skipped"] and len(out["chunks"]) == 2
