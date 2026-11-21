import streamlit as st

from core.eval.benchmark import (SPEC_QUALITATIVE, SPEC_REFERENCE_LATENCY_MS, aggregate, aggregate_by_category,
                                 resolution_accuracy, run_benchmark)
from core.eval.golden_sets import GOLD_ALIASES, GOLDEN
from core.guardrails import PracticeArea
from core.providers import embedding_mode
from core.rag.factory import ENGINES
from ui.common import area_from_label, get_graph, get_store

COLS = {"rcr": "Retrieval Context Recall", "faithfulness": "Faithfulness", "latency_ms": "Latency (ms)",
        "retrieval_ms": "Retrieval (ms)", "graph_ms": "Graph traversal (ms)", "llm_ms": "LLM inference (ms)",
        "prompt_tokens": "Prompt tokens", "completion_tokens": "Completion tokens", "cost_usd": "Cost (USD)"}


def _llm_calls(engines, n_q, repeats, judge, retrieval_only) -> int:
    per_q = {"Standard RAG": 1, "HyDE RAG": 1 + st.session_state.hp.hyde_n, "Graph RAG": 1}
    if retrieval_only:
        return sum(n_q * repeats * (per_q[e] - 1) for e in engines)  # only HyDE expansion calls the LLM
    return sum(n_q * repeats * (per_q[e] + (1 if judge else 0)) for e in engines)


def render():
    st.subheader("Comparative Analysis & Live Performance Benchmark")
    mode = embedding_mode()
    if mode["mock"]:
        st.error("MOCK EMBEDDINGS active: these numbers are meaningless for comparing engines.")
    c1, c2 = st.columns([2, 3])
    area = area_from_label(c1.selectbox("Practice Area", [a.value for a in PracticeArea], key="bm_area"))
    engines = c2.multiselect("Engines", ENGINES, default=ENGINES)
    store, graph = get_store(area.value), get_graph(area.value)
    st.caption(f"{len(store.chunks)} chunks · graph {graph.G.number_of_nodes()} entities / {graph.G.number_of_edges()} relations"
               " · golden set is small and synthetic: it exercises the benchmark mechanics, not real-world accuracy.")

    questions = GOLDEN[area]
    picked = st.multiselect("Golden questions", [f"{q['id']} [{q['category']}] {q['question']}" for q in questions],
                            default=[f"{q['id']} [{q['category']}] {q['question']}" for q in questions])
    qs = [q for q in questions if any(p.startswith(q["id"] + " ") for p in picked)]
    m1, m2, m3 = st.columns(3)
    retrieval_only = m1.radio("Mode", ["Full (answers + tokens + cost)", "Retrieval only (no answer LLM calls)"]) \
        .startswith("Retrieval")
    judge = m2.checkbox("LLM faithfulness judge", value=not retrieval_only, disabled=retrieval_only)
    repeats = m3.number_input("Repeats per question", 1, 3, 1)
    calls = _llm_calls(engines, len(qs), repeats, judge, retrieval_only)
    st.caption(f"Estimated LLM calls: ~{calls}" + (" (HyDE passage generation only)" if retrieval_only else "")
               + " · graph engines need an ingested knowledge graph.")

    if st.button("Run benchmark", type="primary", disabled=not (engines and qs and store.chunks)):
        from core.providers import get_llm

        total = len(engines) * len(qs) * repeats
        bar, live = st.progress(0.0), st.empty()

        def on_row(row, rows):
            bar.progress(len(rows) / total, text=f"{len(rows)}/{total}: {row['engine']} · {row['qid']}")
            import pandas as pd
            live.dataframe(pd.DataFrame(rows)[["engine", "qid", "category", "rcr", "faithfulness", "latency_ms", "error"]]
                           .round(2), width="stretch")

        rows = run_benchmark(engines=engines, area=area, questions=qs, store=store, graph=graph, llm=get_llm(),
                             hp=st.session_state.hp, ledger=st.session_state.ledger,
                             judge_llm=get_llm() if judge and not retrieval_only else None,
                             retrieval_only=retrieval_only, repeats=int(repeats), on_row=on_row,
                             active_entities=st.session_state.active_entities.get(area.value))
        st.session_state.benchmark = {
            "rows": rows, "area": area.value, "mock": mode["mock"], "retrieval_only": retrieval_only,
            "erra": resolution_accuracy(graph, area, GOLD_ALIASES) if "Graph RAG" in engines else None}
        bar.empty()
        live.empty()
        st.rerun()  # Tab 4 renders before this tab; rerun so its KPI tiles pick up the new results

    bm = st.session_state.get("benchmark")
    if not bm:
        st.info("Run the benchmark to fill the comparison matrix below.")
    else:
        _results(bm)
    st.markdown("#### Reference: qualitative comparison from the problem statement")
    st.dataframe(SPEC_QUALITATIVE, hide_index=True, width="stretch")


def _results(bm: dict):
    import pandas as pd

    rows = bm["rows"]
    st.markdown(f"#### Results · {bm['area']}" + (" · retrieval only" if bm["retrieval_only"] else ""))
    if bm["mock"]:
        st.warning("These results were produced with MOCK embeddings.")
    errs = [r for r in rows if r.get("error")]
    if errs:
        st.warning(f"{len(errs)} run(s) failed: " + "; ".join(sorted({f"{r['engine']}: {r['error'][:90]}" for r in errs})))
    agg = aggregate(rows)
    if agg.empty:
        return
    show = agg.rename(columns=COLS)
    st.dataframe(show.round(3), width="stretch")
    k1, k2 = st.columns(2)
    k1.metric("Entity-Relation Resolution Accuracy (Graph RAG)",
              "n/a" if bm["erra"] is None else f"{bm['erra']:.0%}",
              help="Other engines do not resolve entities, so this KPI applies to Graph RAG only.")
    best = agg["rcr"].idxmax() if agg["rcr"].notna().any() else None
    k2.metric("Best Retrieval Context Recall", best or "n/a")

    st.markdown("**Latency: measured breakdown vs. the spec's reference**")
    lat = agg[["retrieval_ms", "graph_ms", "llm_ms"]].rename(columns={
        "retrieval_ms": "Retrieval", "graph_ms": "Graph traversal", "llm_ms": "LLM inference"})
    ref = pd.Series({e: SPEC_REFERENCE_LATENCY_MS[e] for e in agg.index}, name="Spec reference (ms)")
    st.bar_chart(lat)
    st.dataframe(pd.concat([agg["latency_ms"].rename("Measured mean (ms)"), ref], axis=1).round(0), width="stretch")
    st.caption("The spec's reference latencies assume a fast LLM; hosted Claude calls dominate the measured numbers.")

    st.markdown("**Retrieval Context Recall by question type**")
    st.dataframe(aggregate_by_category(rows, "rcr").round(2), width="stretch")
    if not bm["retrieval_only"]:
        st.markdown("**Faithfulness by question type**")
        st.dataframe(aggregate_by_category(rows, "faithfulness").round(2), width="stretch")
    with st.expander("Per-question detail"):
        st.dataframe(pd.DataFrame(rows).drop(columns=["answer", "sources"], errors="ignore").round(3), width="stretch")
    st.download_button("Download results (CSV)", pd.DataFrame(rows).to_csv(index=False), "benchmark_results.csv", "text/csv")
