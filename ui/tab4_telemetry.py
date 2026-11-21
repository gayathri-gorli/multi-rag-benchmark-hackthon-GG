import streamlit as st

from core.config import Hyperparams
from core.eval.benchmark import aggregate
from core.rag.factory import ENGINES


def _fmt(v, pct=True):
    return "n/a" if v is None or v != v else (f"{v:.0%}" if pct else f"{v:,.0f} ms")


def render():
    st.subheader("Hyperparameters & KPI Telemetry")
    hp: Hyperparams = st.session_state.hp
    c = st.columns(4)
    size = c[0].number_input("Chunk size (tokens)", 100, 2000, hp.chunk_size)
    ovl = c[0].number_input("Chunk overlap (tokens)", 0, 500, hp.chunk_overlap)
    n = c[1].slider("Multi-HyDE passage count N", 1, 6, hp.hyde_n)
    k = c[2].slider("Graph hop radius k", 1, 4, hp.hop_k)
    a = c[3].number_input("RRF constant α", 1, 200, hp.rrf_alpha)
    top_k = c[3].slider("Chunks to synthesis (top-k)", 1, 20, hp.top_k)
    if ovl >= size:
        st.error("Overlap must be smaller than chunk size.")
    else:
        st.session_state.hp = Hyperparams(chunk_size=size, chunk_overlap=ovl, hyde_n=n, hop_k=k,
                                          rrf_alpha=a, top_k=top_k)
        st.caption("Chunk settings apply to newly ingested documents; the rest apply to the next query or benchmark.")

    st.markdown("#### KPIs")
    bm = st.session_state.get("benchmark")
    agg = aggregate(bm["rows"]) if bm else None
    engine = st.selectbox("Engine", ENGINES, key="kpi_engine")
    have = agg is not None and not agg.empty and engine in agg.index
    t = st.session_state.last_timings
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Retrieval Context Recall", _fmt(agg.loc[engine, "rcr"]) if have else "n/a",
              help="Mean over the latest Tab 5 benchmark.")
    k2.metric("Faithfulness", _fmt(agg.loc[engine, "faithfulness"]) if have and "faithfulness" in agg else "n/a",
              help="LLM judge; only computed in Full mode with the judge enabled.")
    erra = bm.get("erra") if bm else None
    k3.metric("Entity-Relation Resolution Acc.", _fmt(erra) if engine == "Graph RAG" else "n/a (Graph RAG only)")
    if have:
        k4.metric("End-to-end latency (benchmark mean)", _fmt(agg.loc[engine, "latency_ms"], pct=False))
    else:
        k4.metric("End-to-end latency (last query)", f"{t['total']:,.0f} ms" if t else "n/a")
    if not bm:
        st.caption("Run the benchmark in Tab 5 to populate recall, faithfulness and resolution accuracy.")
    if t:
        st.markdown("Latest query: per-node latency (graph_traversal is a sub-step of hybrid_search)")
        st.bar_chart({k: v for k, v in t.items() if k != "total"}, horizontal=True)
