import tempfile
import uuid
from pathlib import Path

import streamlit as st

from core.guardrails import PracticeArea
from core.providers import embedding_mode
from core.rag.factory import ENGINES
from ui.common import area_from_label, get_graph, get_store


def _ingest_paths(paths, area, extract_graph: bool):
    from core.ingestion.chunker import chunk_records
    from core.ingestion.loaders import load_file
    from core.providers import get_llm
    from core.tracking.tokens import make_langchain_callback

    store, graph = get_store(area.value), get_graph(area.value)
    run_id = uuid.uuid4().hex[:8]
    for p in paths:
        try:
            chunks = chunk_records(load_file(p, area.value), st.session_state.hp)
            res = store.add_chunks(chunks)
            msg = f"{p.name}: {res['added']} chunks added, {res['skipped_duplicates']} duplicates skipped."
            if res["over_embedding_limit"]:
                msg += (f" Warning: {res['over_embedding_limit']} chunks exceed the 512-token embedding limit; "
                        "their tails are truncated in the embedding (BM25 still sees the full text).")
            st.success(msg)
            if extract_graph:
                from core.ingestion.entity_extract import extract_chunks

                cb = make_langchain_callback(st.session_state.ledger, run_id, "Graph RAG", "entity_extraction")
                with st.spinner(f"Extracting entities and relations from {p.name} (LLM)..."):
                    g = extract_chunks(chunks, graph, get_llm(callbacks=[cb]))
                st.success(f"Graph: {g['nodes']} entities, {g['edges']} relations "
                           f"({g['extracted']} chunks extracted, {g['failed']} failed).")
        except Exception as e:
            st.error(f"{p.name}: {e}")


def _ingest(files, area, extract_graph: bool):
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for f in files:
            p = Path(d) / f.name  # keep the original name for provenance
            p.write_bytes(f.getvalue())
            paths.append(p)
        _ingest_paths(paths, area, extract_graph)


def render():
    st.subheader("Data Ingestion & Multi-RAG Chat Console")
    c1, c2 = st.columns(2)
    area_label = c1.selectbox("Practice Area", [a.value for a in PracticeArea])
    engine = c2.selectbox("RAG Engine", ENGINES)
    area = area_from_label(area_label)

    files = st.file_uploader("Drag and drop PDF / DOCX (also JSON, CSV, TXT)", accept_multiple_files=True,
                             type=["pdf", "docx", "json", "csv", "txt"])
    extract_graph = st.checkbox("Also extract the knowledge graph (LLM call per chunk; required for Graph RAG)", value=True)
    b1, b2 = st.columns(2)
    if files and b1.button("Ingest documents", type="primary"):
        _ingest(files, area, extract_graph)
    from core.samples import sample_files

    if b2.button(f"Load {len(sample_files(area))} SYNTHETIC sample documents for this area"):
        _ingest_paths(sample_files(area), area, extract_graph)
    g = get_graph(area.value)
    st.caption(f"Indexed chunks: {len(get_store(area.value).chunks)} · graph: "
               f"{g.G.number_of_nodes()} entities / {g.G.number_of_edges()} relations")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            if m.get("engine"):
                st.caption(m["engine"] + (" · MOCK EMBEDDINGS" if m.get("mock") else ""))
            st.markdown(m["content"])
            for flag in m.get("flags", []):
                st.warning(flag)
            if m.get("citations"):
                with st.expander("Sources"):
                    for c in m["citations"]:
                        st.markdown(f"**[{c['tag']}]** " + (f"{c['source']}" + (f", p.{c['page']}" if c.get("page") else "")
                                    if c["valid"] else "invalid citation tag (not in retrieved sources)"))
                        if c["valid"]:
                            st.caption(c["snippet"])

    q = st.chat_input("Ask a compliance question")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        run_id = uuid.uuid4().hex[:8]
        try:
            from core.providers import get_llm
            from core.rag.factory import build_engine
            from core.tracking.tokens import make_langchain_callback

            rag = build_engine(
                engine, store=get_store(area.value), llm=get_llm(), area=area, hp=st.session_state.hp,
                graph=get_graph(area.value) if engine == "Graph RAG" else None,
                enabled_nodes=st.session_state.enabled.get(engine, {}),
                enabled_edges=st.session_state.enabled_edges.get(engine, {}),
                callback_factory=lambda node: [make_langchain_callback(st.session_state.ledger, run_id, engine, node)],
                active_entities=st.session_state.active_entities.get(area.value),
                active_edges=st.session_state.active_edges.get(area.value))
            with st.spinner(f"{engine}: retrieving and synthesizing..."):
                out = rag.run(q)
            st.session_state.last_timings = out["timings"] | {"total": out["total_ms"]}
            st.session_state.last_run[engine] = {k: out.get(k) for k in ("timings", "skipped", "seeds", "kg_paths", "severed", "blocked")}
            st.session_state.chat.append({"role": "assistant", "engine": engine, "mock": embedding_mode()["mock"], "content": out["answer"],
                                          "citations": out["citations"], "flags": out["guardrail_flags"]})
        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()

    t = st.session_state.ledger.totals()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prompt tokens", f"{t['prompt_tokens']:,}")
    k2.metric("Completion tokens", f"{t['completion_tokens']:,}")
    k3.metric("Total tokens", f"{t['total_tokens']:,}")
    k4.metric("Est. cost (USD)", f"${t['cost_usd']:.4f}")
    if t["any_estimated"]:
        st.caption("Some counts are locally estimated (provider returned no usage).")
