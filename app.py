import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="PwC India Multi-RAG Benchmark", layout="wide")

from ui import tab1_chat, tab2_graph, tab3_workflow, tab4_telemetry, tab5_benchmark
from ui.common import init_state

init_state()
st.title("PwC India Enterprise Multi-RAG Comparative Benchmark")
from core.providers import embedding_mode

_mode = embedding_mode()
if _mode["mock"]:
    st.error(f"{_mode['label']} - retrieval quality and all benchmark numbers are NOT meaningful. "
             "UI and workflow testing only.")
else:
    st.caption(f"Embeddings: {_mode['label']}")

tabs = st.tabs(["1 · Ingestion & Chat", "2 · Knowledge Graph", "3 · Workflow Canvas",
                "4 · Hyperparameters & KPIs", "5 · Benchmark"])
with tabs[0]:
    tab1_chat.render()
with tabs[1]:
    tab2_graph.render()
with tabs[2]:
    tab3_workflow.render()
with tabs[3]:
    tab4_telemetry.render()
with tabs[4]:
    tab5_benchmark.render()
