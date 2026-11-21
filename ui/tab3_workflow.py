import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.guardrails import PracticeArea
from core.rag.factory import ENGINES, build_engine
from core.rag.impact import compare_runs
from core.rag.standard import EDGE_CARRIES, EDGE_EFFECTS, EDGES, NODES, REQUIRED_NODES
from ui.common import area_from_label, get_graph, get_store

LABELS = {"query_expansion": "Query Expansion", "hybrid_search": "Hybrid Vector/Graph Search",
          "rrf_fusion": "RRF Fusion", "rerank": "Rerank", "synthesis": "Synthesis"}
NOTES = {
    "Standard RAG": {"query_expansion": "identity"},
    "HyDE RAG": {"query_expansion": "N synthetic passages"},
    "Graph RAG": {"query_expansion": "entity linking", "hybrid_search": "vector + BM25 + ↻ k-hop graph"},
}


def edge_label(e: str) -> str:
    a, b = e.split("->")
    return f"{LABELS[a]} → {LABELS[b]}"


def build_figure(engine: str, enabled: dict, edges: dict, last: dict, hp) -> go.Figure:
    timings = (last or {}).get("timings") or {}
    fig = go.Figure()
    xs = list(range(len(NODES)))
    for i, e in enumerate(EDGES):
        on = edges.get(e, True)
        fig.add_trace(go.Scatter(x=[i + 0.18, i + 0.82], y=[0, 0], mode="lines", hoverinfo="skip",
                                 line=dict(color="#666" if on else "#D62728", width=3, dash="solid" if on else "dot")))
        fig.add_annotation(x=i + 0.82, y=0, ax=i + 0.6, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowcolor="#666" if on else "#D62728")
        if not on:
            fig.add_annotation(x=i + 0.5, y=0.22, text="✂ severed", showarrow=False, font=dict(color="#D62728", size=12))
    colors, texts = [], []
    for n in NODES:
        on = enabled.get(n, True)
        colors.append("#4C78A8" if on and n in timings else "#9ecae9" if on else "#cccccc")
        note = NOTES[engine].get(n, "")
        if engine == "HyDE RAG" and n == "query_expansion":
            note = f"N={hp.hyde_n} passages"
        if engine == "Graph RAG" and n == "hybrid_search":
            note = f"vector + BM25 + ↻ k={hp.hop_k} hops"
        if n == "rrf_fusion":
            note = f"α={hp.rrf_alpha}"
        ms = f"<br>{timings[n]:.0f} ms" if n in timings and on else ""
        texts.append(f"<b>{LABELS[n]}</b>{ms}<br><i>{note if on else 'OFF (bypassed)'}</i>")
    fig.add_trace(go.Scatter(x=xs, y=[0] * len(xs), mode="markers+text", text=texts, textposition="bottom center",
                             marker=dict(size=46, color=colors, line=dict(color="#333", width=2)), hoverinfo="skip"))
    if engine == "Graph RAG":
        fig.add_annotation(x=1, y=0.55, text="↻ expand_hop (LangGraph cycle, up to k hops)", showarrow=True,
                           ax=0, ay=-10, arrowhead=2, font=dict(color="#E8590C"))
    fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False, range=[-0.6, len(NODES) - 0.4]),
                      yaxis=dict(visible=False, range=[-1.2, 1]))
    return fig


def _chunk_label(store, cid: str) -> str:
    if cid == "kg":
        return "Knowledge Graph paths"
    c = store.chunks.get(cid)
    return f"{c['metadata']['source']} #{c['metadata']['chunk_index']}" if c else cid


def _impact_preview(engine: str, enabled: dict, edges: dict):
    st.markdown("#### Impact preview (retrieval only, no synthesis call)")
    c1, c2 = st.columns([1, 2])
    area = area_from_label(c1.selectbox("Practice Area", [a.value for a in PracticeArea], key="wf_area"))
    query = c2.text_input("Test question", key="wf_query")
    hyde_cost = engine == "HyDE RAG" and enabled.get("query_expansion", True)
    if hyde_cost:
        st.caption("HyDE generates its hypothetical passages with the LLM even in preview "
                   "(a few small calls, run once for the baseline and once for the variant).")
    if not st.button("Compare against the full pipeline") or not query:
        return
    store, graph = get_store(area.value), get_graph(area.value)
    if not store.chunks:
        st.info("No documents indexed for this area yet. Load samples in Tab 1 first.")
        return
    if engine == "Graph RAG" and graph.G.number_of_nodes() == 0:
        st.info("Graph RAG needs a knowledge graph. Ingest with graph extraction enabled in Tab 1.")
        return
    from core.providers import get_llm

    common = dict(store=store, llm=get_llm(), area=area, hp=st.session_state.hp, graph=graph,
                  active_entities=st.session_state.active_entities.get(area.value),
                  active_edges=st.session_state.active_edges.get(area.value))
    with st.spinner("Running baseline and variant..."):
        base = build_engine(engine, **common).run(query, dry_run=True)
        var = build_engine(engine, enabled_nodes=enabled, enabled_edges=edges, **common).run(query, dry_run=True)
    r = compare_runs(base, var)
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks (full pipeline)", len(r["baseline_chunks"]))
    m2.metric("Chunks (your variant)", len(r["variant_chunks"]), delta=len(r["variant_chunks"]) - len(r["baseline_chunks"]))
    m3.metric("Context overlap", f"{r['overlap']:.0%}")
    if r["empty"]:
        st.error("Nothing reaches Synthesis, so the answer would be a Gap Disclosure.")
    if r["skipped_nodes"]:
        st.write("**Bypassed nodes:** " + ", ".join(LABELS[n] for n in r["skipped_nodes"]))
    for e, effect in r["severed_edges"]:
        st.write(f"**Severed {edge_label(e)}:** withheld `{', '.join(EDGE_CARRIES[e])}`. {effect}")
    if r["dropped"]:
        st.write("**Lost vs full pipeline:** " + "; ".join(_chunk_label(store, c) for c in r["dropped"]))
    if r["added"]:
        st.write("**Newly included:** " + "; ".join(_chunk_label(store, c) for c in r["added"]))
    if r["order_changed"]:
        st.write("Ranking order of shared chunks changed.")
    if not (r["dropped"] or r["added"] or r["order_changed"] or r["empty"]):
        st.success("Retrieved context is identical to the full pipeline for this question.")
    t = r["timings_ms"]
    st.dataframe(pd.DataFrame({"full pipeline (ms)": t["baseline"], "your variant (ms)": t["variant"]}).round(1))


def render():
    st.subheader("LangGraph Workflow Canvas")
    engine = st.radio("Pipeline", ENGINES, horizontal=True)
    nodes = st.session_state.enabled.setdefault(engine, {})
    edges = st.session_state.enabled_edges.setdefault(engine, {})
    st.caption("Nodes")
    for col, n in zip(st.columns(len(NODES)), NODES):
        required = n in REQUIRED_NODES
        nodes[n] = col.checkbox(LABELS[n], value=nodes.get(n, True), disabled=required, key=f"tog_{engine}_{n}",
                                help="Required node" if required else "Toggle off to bypass this node")
    st.caption("Transitions (edges)")
    for col, e in zip(st.columns(len(EDGES)), EDGES):
        edges[e] = col.checkbox(edge_label(e), value=edges.get(e, True), key=f"edge_{engine}_{e}",
                                help=f"Carries: {', '.join(EDGE_CARRIES[e])}. If disabled: {EDGE_EFFECTS[e]}")

    last = st.session_state.last_run.get(engine)
    st.plotly_chart(build_figure(engine, nodes, edges, last, st.session_state.hp), width="stretch")
    for e in EDGES:
        if not edges.get(e, True):
            st.warning(f"{edge_label(e)} disabled: {EDGE_EFFECTS[e]}")
    if last and (last.get("skipped") or last.get("severed")):
        st.caption("Last run - bypassed nodes: " + (", ".join(last.get("skipped") or []) or "none")
                   + " · severed edges: " + (", ".join(edge_label(e) for e in last.get("severed") or []) or "none"))
    if not last:
        st.caption("Run a query in Tab 1 with this engine to see per-node latency.")
    if engine == "Graph RAG" and last and last.get("kg_paths"):
        with st.expander("Entity-relation paths used in the last run"):
            st.write("Seeds: ", last.get("seeds"))
            st.code("\n".join(last["kg_paths"]))
    st.caption("Toggles apply to the next query. Disabling a node bypasses it; disabling an edge keeps both nodes "
               "but withholds the data that transition carries.")
    _impact_preview(engine, nodes, edges)
