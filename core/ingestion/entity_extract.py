"""Spec step 2 (LLM-guided entity-relation extraction) feeding the knowledge graph."""
import json
import re

from core.index.graph import GraphStore
from core.index.node2vec import node2vec_embeddings

EXTRACT_PROMPT = """Extract entity-relation triples from this regulatory/legal text for a knowledge graph.
Entities: authorities, statutes, sections/rules/clauses, obligations, penalties, thresholds, parties, instruments.
Use short, specific names (e.g. "Section 115BAA", "CBDT", "Input Tax Credit"). Relations are short verb phrases
(e.g. "issues", "amends", "applies to", "imposes penalty on", "requires").
Return ONLY a JSON array (max 12 items) of {{"subject": str, "relation": str, "object": str}}. No prose.

TEXT:
{text}"""


def parse_triples(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [t for t in data if isinstance(t, dict) and {"subject", "relation", "object"} <= set(t)]


def extract_chunks(chunks: list[dict], graph: GraphStore, llm, callbacks=None, embed: bool = True) -> dict:
    """Extract triples for chunks not yet in the cache, add to graph, refresh node embeddings, save."""
    done = failed = 0
    for c in chunks:
        cid = c["metadata"]["chunk_id"]
        if cid not in graph.cache:
            try:
                raw = llm.invoke(EXTRACT_PROMPT.format(text=c["text"]), config={"callbacks": callbacks or []}).content
                graph.cache[cid] = parse_triples(raw)
                done += 1
            except Exception:
                failed += 1  # not cached, so a later run retries it
                continue
        graph.add_triples(cid, graph.cache[cid])
    if embed and graph.G.number_of_nodes() > 1:
        adj = {n: set(graph.G.successors(n)) | set(graph.G.predecessors(n)) for n in graph.G.nodes}
        try:
            graph.node_vecs = node2vec_embeddings(adj)
        except ValueError:
            graph.node_vecs = {}
    graph.save()
    return {"extracted": done, "failed": failed, "nodes": graph.G.number_of_nodes(),
            "edges": graph.G.number_of_edges()}
