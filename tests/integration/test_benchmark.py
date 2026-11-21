import json

import pytest

from core.config import Hyperparams
from core.eval.benchmark import aggregate, aggregate_by_category, run_benchmark
from core.eval.golden_sets import GOLD_ALIASES, GOLDEN
from core.eval.kpis import entity_resolution_accuracy, faithfulness, latency_breakdown, retrieval_context_recall
from core.guardrails import PracticeArea
from core.index.graph import GraphStore
from core.index.store import ChunkStore
from core.ingestion.chunker import chunk_records
from core.ingestion.entity_extract import extract_chunks
from core.ingestion.loaders import load_file
from core.mock_embed import MockEmbeddings
from core.samples import sample_files
from core.tracking.tokens import TokenLedger
from tests.conftest import fake_llm

TAX = PracticeArea.TAX


def chunk(text, cid="c"):
    return {"text": text, "metadata": {"chunk_id": cid}}


# ---- KPI unit behaviour ---------------------------------------------------
def test_rcr_counts_facts_with_alternatives_and_ignores_kg_chunk():
    chunks = [chunk("The rate is 22 Percent under Section 115ZA."), chunk("CBDT issued it", "kg")]
    assert retrieval_context_recall(chunks, [["22 percent"], ["115za"]]) == 1.0
    assert retrieval_context_recall(chunks, [["22 percent"], ["cbdt"]]) == 0.5  # KG chunk text does not count
    assert retrieval_context_recall(chunks, [["x", "section 115za"]]) == 1.0
    assert retrieval_context_recall([], [["22 percent"]]) == 0.0 and retrieval_context_recall(chunks, []) is None


def test_faithfulness_ratio_and_edge_cases():
    judge = fake_llm(lambda p: 'Result: {"claims":[{"claim":"a","supported":true},{"claim":"b","supported":false},'
                               '{"claim":"c","supported":true},{"claim":"d","supported":true}]}')
    assert faithfulness("Answer text [S1]", [chunk("src")], judge)["score"] == 0.75
    assert faithfulness("## Gap Disclosure\nnothing", [chunk("src")], judge)["score"] is None
    assert faithfulness("x", [], judge)["score"] is None
    bad = fake_llm(lambda p: "not json at all")
    assert faithfulness("x [S1]", [chunk("src")], bad)["score"] is None


def test_latency_breakdown_sums_to_total():
    b = latency_breakdown({"total_ms": 1000.0, "timings": {"graph_traversal": 100.0}}, llm_ms=600.0)
    assert b["retrieval_ms"] == pytest.approx(300.0) and b["latency_ms"] == 1000.0
    assert latency_breakdown({"total_ms": 100.0, "timings": {}}, llm_ms=500.0)["retrieval_ms"] == 0.0


def test_golden_sets_cover_all_areas_and_categories():
    assert set(GOLDEN) == set(PracticeArea)
    cats = {q["category"] for qs in GOLDEN.values() for q in qs}
    assert cats == {"keyword", "multi-hop", "vocabulary-mismatch"}
    assert all(q["facts"] for qs in GOLDEN.values() for q in qs)


# ---- Benchmark runner -----------------------------------------------------
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import os
    os.environ["EMBEDDING_PROVIDER"] = "mock"
    root = tmp_path_factory.mktemp("bench")
    store = ChunkStore(TAX.value, MockEmbeddings(), root=root / "s")
    chunks = [c for f in sample_files(TAX) for c in chunk_records(load_file(f, TAX.value))]
    store.add_chunks(chunks)
    kg = GraphStore(root / "g.json")

    def extractor(p):
        if "Section 115ZA" in p:
            return json.dumps([{"subject": "Central Board of Direct Taxes", "relation": "clarifies", "object": "Section 115ZA"},
                               {"subject": "Section 115ZA", "relation": "denies", "object": "MAT credit"}])
        return "[]"
    extract_chunks(chunks, kg, fake_llm(extractor))
    return store, kg


def bench(env, **kw):
    store, kg = env
    args = dict(engines=["Standard RAG", "HyDE RAG", "Graph RAG"], area=TAX, questions=GOLDEN[TAX][:3], store=store,
                graph=kg, llm=fake_llm(lambda p: "Answer per Section 115ZA [S1]."), hp=Hyperparams(top_k=3),
                ledger=TokenLedger())
    return run_benchmark(**(args | kw))


def test_benchmark_covers_every_engine_and_question(env):
    rows = bench(env, retrieval_only=True)
    assert len(rows) == 9 and all(r["error"] is None for r in rows)
    assert all(0.0 <= r["rcr"] <= 1.0 for r in rows) and all(r["answer"] is None for r in rows)
    assert all(r["latency_ms"] > 0 and r["retrieval_ms"] >= 0 for r in rows)
    agg = aggregate(rows)
    assert set(agg.index) == {"Standard RAG", "HyDE RAG", "Graph RAG"} and (agg["questions"] == 3).all()


def test_full_run_with_judge_records_faithfulness(env):
    judge = fake_llm(lambda p: '{"claims":[{"claim":"x","supported":true},{"claim":"y","supported":false}]}')
    rows = bench(env, engines=["Standard RAG"], judge_llm=judge)
    assert all(r["faithfulness"] == 0.5 and r["answer"] for r in rows)
    assert aggregate(rows).loc["Standard RAG", "faithfulness"] == 0.5


def test_failing_question_does_not_abort_run(env):
    rows = bench(env, engines=["Standard RAG"], llm=fake_llm(lambda p: 1 / 0))
    assert len(rows) == 3 and all("ZeroDivisionError" in r["error"] for r in rows)
    assert aggregate(rows).empty


def test_graph_rag_without_graph_reports_error_row(env):
    store, _ = env
    rows = bench(env, engines=["Graph RAG"], graph=None, retrieval_only=True)
    assert all("Graph RAG requires a knowledge graph" in r["error"] for r in rows)


def test_hyde_makes_more_llm_calls_and_ledger_attributes_them_per_engine(env):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    ledger = TokenLedger()
    rows = bench(env, engines=["Standard RAG", "HyDE RAG"], questions=GOLDEN[TAX][:2], ledger=ledger,
                 llm=FakeListChatModel(responses=["Answer per Section 115ZA [S1]."]))
    assert all(r["error"] is None for r in rows)
    std, hyde = ledger.totals(engine="Standard RAG"), ledger.totals(engine="HyDE RAG")
    assert std["calls"] == 2 and hyde["calls"] == 2 * (1 + 3)  # N=3 hypothetical passages + synthesis per question
    assert {r.node for r in ledger.rows if r.engine == "HyDE RAG"} == {"query_expansion", "synthesis"}
    assert hyde["total_tokens"] > std["total_tokens"] and std["any_estimated"]  # fake model reports no usage -> estimated
    agg = aggregate(rows)
    assert agg.loc["HyDE RAG", "prompt_tokens"] > agg.loc["Standard RAG", "prompt_tokens"]


def test_category_breakdown_and_resolution_accuracy(env):
    store, kg = env
    rows = bench(env, retrieval_only=True)
    cat = aggregate_by_category(rows, "rcr")
    assert {"keyword", "multi-hop", "vocabulary-mismatch"} >= set(cat.index) and "Graph RAG" in cat.columns
    acc = entity_resolution_accuracy(kg, GOLD_ALIASES[TAX])
    assert 0.0 < acc < 1.0  # CBDT-family resolves; CBIC is not in this graph, so honestly not 100%
