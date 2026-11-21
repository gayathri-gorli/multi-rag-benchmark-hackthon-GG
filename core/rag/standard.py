"""Standard RAG as a LangGraph StateGraph with the spec's five nodes:
Query Expansion -> Hybrid Search -> RRF Fusion -> Rerank -> Synthesis.
HyDE and Graph RAG subclass this and override individual nodes, so all three share the
same DAG shape (Tab 3). Optional nodes can be toggled off (pass-through)."""
import time
from typing import Any, Callable, TypedDict

from core.citations import build_context, resolve_citations
from core.config import Hyperparams
from core.guardrails import PracticeArea, build_system_prompt, check_answer, redact_pii
from core.rag.fusion import rrf_fuse

NODES = ["query_expansion", "hybrid_search", "rrf_fusion", "rerank", "synthesis"]
REQUIRED_NODES = {"hybrid_search", "synthesis"}  # cannot be toggled off (no retrieval / no answer)
EDGES = [f"{a}->{b}" for a, b in zip(NODES, NODES[1:])]  # workflow transitions (Tab 3 edge toggles)
# State fields each transition hands to the downstream node. Disabling an edge withholds exactly these
# fields from that node and everything after it; the nodes themselves still run.
EDGE_CARRIES = {
    "query_expansion->hybrid_search": ["queries", "seeds"],
    "hybrid_search->rrf_fusion": ["ranked_lists", "kg_paths", "graph_nodes"],
    "rrf_fusion->rerank": ["fused"],
    "rerank->synthesis": ["chunks"],
}
EDGE_EFFECTS = {
    "query_expansion->hybrid_search": "Search sees only the raw query: no HyDE passages / no linked graph entities (Graph RAG loses its k-hop seeds).",
    "hybrid_search->rrf_fusion": "No ranked lists reach fusion or any later step: nothing is retrieved, so the answer is a gap disclosure.",
    "rrf_fusion->rerank": "Fused ranking is discarded; rerank falls back to round-robin over the raw ranked lists.",
    "rerank->synthesis": "Reranked chunks (and Graph RAG's KG evidence) are dropped; synthesis falls back to the top-k of the fused order.",
}
GAP_ANSWER = ("## Gap Disclosure\nNo relevant passages were found in the indexed corpus for this "
              "practice area, so no verifiable answer can be given.")


class RAGState(TypedDict, total=False):
    query: str
    queries: list[str]
    ranked_lists: list[list[str]]
    fused: list[str]
    chunks: list[dict]
    answer: str
    citations: list[dict]
    guardrail_flags: list[str]
    timings: dict[str, float]
    sub_timings: dict[str, float]
    skipped: list[str]
    seeds: list[str]
    kg_paths: list[str]
    graph_nodes: dict[str, int]
    blocked: list[str]   # state fields withheld by disabled edges
    severed: list[str]   # disabled edges that were hit in this run


def _timed(name, fn, enabled: dict[str, bool], edges_enabled: dict[str, bool]):
    incoming = next((e for e in EDGES if e.endswith(f"->{name}")), None)

    def wrapper(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        blocked = set(state.get("blocked", []))
        severed = list(state.get("severed", []))
        if incoming and not edges_enabled.get(incoming, True):
            blocked |= set(EDGE_CARRIES[incoming])
            severed.append(incoming)
        view = {k: v for k, v in state.items() if k not in blocked}  # what this node is allowed to see
        if enabled.get(name, True) or name in REQUIRED_NODES:
            out = fn(view)
        else:
            out = {"skipped": state.get("skipped", []) + [name]}
        timings = dict(state.get("timings", {}))
        timings[name] = (time.perf_counter() - t0) * 1000
        timings.update(out.pop("sub_timings", {}))
        return {**out, "timings": timings, "blocked": sorted(blocked), "severed": severed}
    return wrapper


class StandardRAG:
    engine = "Standard RAG"

    def __init__(self, store, llm, area: PracticeArea, hp: Hyperparams | None = None,
                 enabled_nodes: dict[str, bool] | None = None,
                 callback_factory: Callable[[str], list] | None = None,
                 enabled_edges: dict[str, bool] | None = None):
        self.store, self.llm, self.area = store, llm, area
        self.hp = hp or Hyperparams()
        self.enabled = enabled_nodes or {}
        self.edges_enabled = enabled_edges or {}
        self.dry_run = False  # retrieval-only preview (no synthesis LLM call)
        self.cb = callback_factory or (lambda node: [])  # node name -> LangChain callbacks (token ledger)
        self.graph = self._build()

    def _invoke(self, node: str, messages):
        return self.llm.invoke(messages, config={"callbacks": self.cb(node)}).content

    def _interleave(self, lists: list[list[str]]) -> list[str]:
        """Fallback ordering when RRF is toggled off: round-robin over the ranked lists."""
        out, seen = [], set()
        for rank in range(max((len(l) for l in lists), default=0)):
            for l in lists:
                if rank < len(l) and l[rank] not in seen:
                    seen.add(l[rank])
                    out.append(l[rank])
        return out

    # --- nodes -----------------------------------------------------------
    def _expand(self, s: RAGState) -> RAGState:
        return {"queries": [s["query"]]}

    def _vector_bm25(self, queries: list[str]) -> list[list[str]]:
        k = self.hp.top_k * 4
        lists = []
        for q in queries:
            lists.append(self.store.vector_search(q, k))
            lists.append(self.store.bm25_search(q, k))
        return [l for l in lists if l]

    def _search(self, s: RAGState) -> RAGState:
        return {"ranked_lists": self._vector_bm25(s.get("queries") or [s["query"]])}

    def _fuse(self, s: RAGState) -> RAGState:
        return {"fused": [d for d, _ in rrf_fuse(s.get("ranked_lists", []), self.hp.rrf_alpha)]}

    def _ordered_ids(self, s: RAGState) -> list[str]:
        return s.get("fused") or self._interleave(s.get("ranked_lists", []))

    def _rerank(self, s: RAGState) -> RAGState:
        # No reranker model is specified yet; keeps fused order and truncates to top_k.
        return {"chunks": [self.store.get(i) for i in self._ordered_ids(s)[: self.hp.top_k]]}

    def _synthesize(self, s: RAGState) -> RAGState:
        chunks = s.get("chunks") or [self.store.get(i) for i in self._ordered_ids(s)[: self.hp.top_k]]
        if not chunks:
            return {"chunks": [], "answer": GAP_ANSWER, "citations": [], "guardrail_flags": []}
        if self.dry_run:
            return {"chunks": chunks, "answer": "(dry run: synthesis skipped)", "citations": [], "guardrail_flags": []}
        msgs = [("system", build_system_prompt(self.area)),
                ("human", f"Sources:\n{build_context(chunks)}\n\nQuestion: {s['query']}")]
        answer = self._invoke("synthesis", msgs)
        return {"chunks": chunks, "answer": answer, "citations": resolve_citations(answer, chunks),
                "guardrail_flags": check_answer(answer, self.area)}

    # --- graph -----------------------------------------------------------
    def _build(self):
        from langgraph.graph import END, START, StateGraph

        fns = {"query_expansion": self._expand, "hybrid_search": self._search,
               "rrf_fusion": self._fuse, "rerank": self._rerank, "synthesis": self._synthesize}
        g = StateGraph(RAGState)
        for name in NODES:
            g.add_node(name, _timed(name, fns[name], self.enabled, self.edges_enabled))
        g.add_edge(START, NODES[0])
        for a, b in zip(NODES, NODES[1:]):
            g.add_edge(a, b)
        g.add_edge(NODES[-1], END)
        return g.compile()

    def run(self, query: str, dry_run: bool = False) -> dict[str, Any]:
        """dry_run=True stops short of the synthesis LLM call (retrieval-only impact preview)."""
        self.dry_run = dry_run
        t0 = time.perf_counter()
        out = self.graph.invoke({"query": redact_pii(query), "timings": {}, "skipped": [],
                                 "blocked": [], "severed": []})
        out["total_ms"] = (time.perf_counter() - t0) * 1000
        return out
