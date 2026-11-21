from core.rag.graphrag import GraphRAG
from core.rag.hyde import HyDERAG
from core.rag.standard import StandardRAG

ENGINES = ["Standard RAG", "HyDE RAG", "Graph RAG"]


def build_engine(engine: str, *, store, llm, area, hp, graph=None, enabled_nodes=None,
                 callback_factory=None, active_entities=None, active_edges=None, enabled_edges=None):
    kw = dict(hp=hp, enabled_nodes=enabled_nodes, callback_factory=callback_factory, enabled_edges=enabled_edges)
    if engine == "Standard RAG":
        return StandardRAG(store, llm, area, **kw)
    if engine == "HyDE RAG":
        return HyDERAG(store, llm, area, **kw)
    if engine == "Graph RAG":
        if graph is None:
            raise ValueError("Graph RAG requires a knowledge graph; ingest with graph extraction enabled.")
        return GraphRAG(store, llm, area, graph, active_entities=active_entities,
                        active_edges=active_edges, **kw)
    raise ValueError(f"Unknown engine: {engine}")
