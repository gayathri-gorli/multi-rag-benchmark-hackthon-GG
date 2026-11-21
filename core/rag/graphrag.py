"""Graph RAG: vector + BM25 + k-hop knowledge-graph retrieval, fused with RRF.
The k-hop expansion is a cyclic LangGraph sub-graph (expand_hop loops until k hops or no new
entities) running inside the 'hybrid_search' node, so the visible DAG keeps the spec's five nodes."""
import time
from typing import TypedDict

from core.rag.standard import RAGState, StandardRAG


class TravState(TypedDict, total=False):
    frontier: list[str]
    dist: dict[str, int]
    hop: int
    k: int
    allowed_nodes: object
    allowed_edges: object


class GraphRAG(StandardRAG):
    engine = "Graph RAG"

    def __init__(self, store, llm, area, graph, hp=None, enabled_nodes=None, callback_factory=None,
                 active_entities=None, active_edges=None, enabled_edges=None):
        self.kg = graph
        # Tab 2 selections: None / empty means "no filter".
        self.active_entities = set(active_entities) if active_entities else None
        self.active_edges = set(active_edges) if active_edges else None
        super().__init__(store, llm, area, hp, enabled_nodes, callback_factory, enabled_edges)
        self.traversal = self._build_traversal()

    def _build_traversal(self):
        from langgraph.graph import END, START, StateGraph

        def hop(s: TravState) -> TravState:
            new = self.kg.step(s["frontier"], s["dist"], s["hop"], s.get("allowed_nodes"), s.get("allowed_edges"))
            return {"frontier": sorted(new), "dist": {**s["dist"], **{n: s["hop"] + 1 for n in new}},
                    "hop": s["hop"] + 1}

        g = StateGraph(TravState)
        g.add_node("expand_hop", hop)
        g.add_edge(START, "expand_hop")
        g.add_conditional_edges("expand_hop", lambda s: "again" if s["frontier"] and s["hop"] < s["k"] else "stop",
                                {"again": "expand_hop", "stop": END})
        return g.compile()

    # --- nodes -----------------------------------------------------------
    def _expand(self, s: RAGState) -> RAGState:
        """Query expansion = entity linking against the knowledge graph (alias-aware)."""
        seeds = self.kg.link_entities(s["query"])
        if self.active_entities:
            seeds = [x for x in seeds if x in self.active_entities] or sorted(self.active_entities)
        return {"queries": [s["query"]], "seeds": seeds}

    def _search(self, s: RAGState) -> RAGState:
        lists = self._vector_bm25(s.get("queries") or [s["query"]])
        t0 = time.perf_counter()
        seeds = [x for x in s.get("seeds", []) if x in self.kg.G]
        dist: dict[str, int] = {}
        if seeds:
            out = self.traversal.invoke({
                "frontier": seeds, "dist": {x: 0 for x in seeds}, "hop": 0, "k": self.hp.hop_k,
                "allowed_nodes": self.active_entities, "allowed_edges": self.active_edges})
            dist = out["dist"]
        graph_ids = [c for c in self.kg.chunks_for_nodes(dist) if c in self.store.chunks]
        if graph_ids:
            lists.append(graph_ids[: self.hp.top_k * 4])
        return {"ranked_lists": lists, "graph_nodes": dist, "kg_paths": self.kg.paths(dist),
                "sub_timings": {"graph_traversal": (time.perf_counter() - t0) * 1000}}

    def _rerank(self, s: RAGState) -> RAGState:
        out = super()._rerank(s)
        if s.get("kg_paths"):  # entity-relation evidence is citable like any other source
            out["chunks"] = out["chunks"] + [{
                "text": "\n".join(s["kg_paths"]),
                "metadata": {"source": "Knowledge Graph (entity-relation paths)", "page": None, "chunk_id": "kg"}}]
        return out
