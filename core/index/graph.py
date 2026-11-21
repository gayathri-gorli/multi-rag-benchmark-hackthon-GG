"""Spec step 5(b)/6: graph adjacency index (Entity -> Relation -> Target) on NetworkX,
with provenance back to chunks, hop-by-hop expansion and JSON persistence.
A Neo4j adapter can implement the same methods later."""
import json
from pathlib import Path

import networkx as nx

from core.ingestion.normalize import aliases_in, canonical, key, node_key


class GraphStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.G = nx.MultiDiGraph()
        self.cache: dict[str, list[dict]] = {}   # chunk_id -> extracted triples (avoids re-paying the LLM)
        self.node_vecs: dict[str, list[float]] = {}
        if self.path.exists():
            self.load()

    # --- build -----------------------------------------------------------
    def add_triples(self, chunk_id: str, triples: list[dict]) -> int:
        added = 0
        for t in triples:
            try:
                s, r, o = (str(t["subject"]).strip(), str(t["relation"]).strip().lower(), str(t["object"]).strip())
            except (KeyError, TypeError):
                continue
            if not (s and r and o):
                continue
            sk, ok = node_key(s), node_key(o)
            if not sk or not ok or sk == ok:
                continue
            for k, name in ((sk, s), (ok, o)):
                if k not in self.G:
                    self.G.add_node(k, label=canonical(name), chunks=[])
                if chunk_id not in self.G.nodes[k]["chunks"]:
                    self.G.nodes[k]["chunks"].append(chunk_id)
            existing = self.G.get_edge_data(sk, ok) or {}
            if not any(d["relation"] == r and d["chunk_id"] == chunk_id for d in existing.values()):
                self.G.add_edge(sk, ok, relation=r, chunk_id=chunk_id)
                added += 1
        return added

    # --- query-time ------------------------------------------------------
    def label(self, k: str) -> str:
        return self.G.nodes[k]["label"]

    def link_entities(self, query: str) -> list[str]:
        """Entity linking: known aliases plus any graph node whose name appears in the query."""
        found = {node_key(c) for c in aliases_in(query)}
        padded = f" {key(query)} "
        found |= {k for k in self.G.nodes if len(k) >= 3 and f" {k} " in padded}
        return sorted(k for k in found if k in self.G)

    def step(self, frontier, dist: dict, hop: int, allowed_nodes=None, allowed_edges=None) -> set:
        """One hop outwards (edges treated as undirected). Returns newly reached nodes."""
        new = set()
        for u in frontier:
            neigh = [(v, (u, v, k)) for _, v, k in self.G.out_edges(u, keys=True)]
            neigh += [(v, (v, u, k)) for v, _, k in self.G.in_edges(u, keys=True)]
            for v, e in neigh:
                if allowed_edges is not None and e not in allowed_edges:
                    continue
                if allowed_nodes is not None and v not in allowed_nodes:
                    continue
                if v not in dist and v not in new:
                    new.add(v)
        return new

    def expand(self, seeds, k: int, allowed_nodes=None, allowed_edges=None) -> dict[str, int]:
        dist = {s: 0 for s in seeds if s in self.G}
        frontier = set(dist)
        for hop in range(k):
            new = self.step(frontier, dist, hop, allowed_nodes, allowed_edges)
            if not new:
                break
            dist.update({n: hop + 1 for n in new})
            frontier = new
        return dist

    def chunks_for_nodes(self, dist: dict[str, int]) -> list[str]:
        """Chunk ids ranked by sum of 1/(1+hop) over the entities they mention."""
        score: dict[str, float] = {}
        for n, h in dist.items():
            for cid in self.G.nodes[n]["chunks"]:
                score[cid] = score.get(cid, 0.0) + 1.0 / (1 + h)
        return [c for c, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]

    def paths(self, dist: dict[str, int], limit: int = 20) -> list[str]:
        edges = [(max(dist[u], dist[v]), u, v, d) for u, v, d in self.G.edges(data=True) if u in dist and v in dist]
        edges.sort(key=lambda e: (e[0], e[1], e[2]))
        return [f"{self.label(u)} --[{d['relation']}]--> {self.label(v)}" for _, u, v, d in edges[:limit]]

    def edge_list(self) -> list[tuple]:
        return [(u, v, k, d["relation"]) for u, v, k, d in self.G.edges(keys=True, data=True)]

    # --- persistence -----------------------------------------------------
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "nodes": [{"id": n, **d} for n, d in self.G.nodes(data=True)],
            "edges": [{"u": u, "v": v, **d} for u, v, d in self.G.edges(data=True)],
            "cache": self.cache, "node_vecs": self.node_vecs,
        }))

    def load(self):
        data = json.loads(self.path.read_text())
        for n in data["nodes"]:
            self.G.add_node(n["id"], label=n["label"], chunks=n["chunks"])
        for e in data["edges"]:
            self.G.add_edge(e["u"], e["v"], relation=e["relation"], chunk_id=e["chunk_id"])
        self.cache, self.node_vecs = data.get("cache", {}), data.get("node_vecs", {})
