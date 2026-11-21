from pathlib import Path

import streamlit as st

from core.config import Hyperparams
from core.guardrails import PracticeArea
from core.index.store import _slug
from core.rag.factory import ENGINES
from core.tracking.tokens import TokenLedger

STORAGE = Path("storage")


def init_state():
    st.session_state.setdefault("hp", Hyperparams())
    st.session_state.setdefault("ledger", TokenLedger())
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("last_timings", {})
    st.session_state.setdefault("last_run", {})        # engine -> {timings, skipped, seeds, kg_paths}
    st.session_state.setdefault("enabled", {e: {} for e in ENGINES})  # Tab 3 node toggles
    st.session_state.setdefault("enabled_edges", {e: {} for e in ENGINES})  # Tab 3 edge toggles
    st.session_state.setdefault("active_entities", {})  # area -> set(node keys)  (Tab 2 filter)
    st.session_state.setdefault("active_edges", {})     # area -> set((u, v, key)) (Tab 2 filter)


@st.cache_resource(show_spinner="Opening vector index...")
def get_store(area_value: str):
    from core.index.store import ChunkStore
    from core.providers import get_embeddings

    return ChunkStore(area_value, get_embeddings(), root=STORAGE)


@st.cache_resource
def get_graph(area_value: str):
    from core.index.graph import GraphStore

    return GraphStore(STORAGE / _slug(area_value) / "graph.json")


def area_from_label(label: str) -> PracticeArea:
    return next(a for a in PracticeArea if a.value == label)
