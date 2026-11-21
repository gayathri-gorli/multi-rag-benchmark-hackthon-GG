"""Structural node embeddings in the Node2Vec family: p/q-biased second-order random walks,
then a PPMI co-occurrence matrix factorised with SVD. This is the closed-form equivalent of
skip-gram with negative sampling (Levy & Goldberg 2014), used here because gensim/torch are
not installed. It is NOT the reference node2vec implementation."""
import math
import random

import numpy as np


def node2vec_embeddings(adj: dict[str, set[str]], dim: int = 32, walk_length: int = 20, num_walks: int = 10,
                        window: int = 3, p: float = 1.0, q: float = 1.0, seed: int = 0,
                        max_nodes: int = 3000) -> dict[str, list[float]]:
    nodes = sorted(adj)
    n = len(nodes)
    if n < 2:
        return {k: [0.0] for k in nodes}
    if n > max_nodes:
        raise ValueError(f"Graph has {n} nodes; dense PPMI is capped at {max_nodes}.")
    idx = {k: i for i, k in enumerate(nodes)}
    nbrs = {k: sorted(v) for k, v in adj.items()}
    rng = random.Random(seed)
    co = np.zeros((n, n))
    for _ in range(num_walks):
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                cand = nbrs[cur]
                if not cand:
                    break
                if len(walk) == 1:
                    walk.append(rng.choice(cand))
                    continue
                prev = walk[-2]
                w = [(1 / p) if x == prev else (1.0 if x in adj[prev] else 1 / q) for x in cand]
                walk.append(rng.choices(cand, weights=w)[0])
            for i, a in enumerate(walk):
                for b in walk[i + 1: i + 1 + window]:
                    co[idx[a], idx[b]] += 1
                    co[idx[b], idx[a]] += 1
    total = co.sum()
    if total == 0:
        return {k: [0.0] * min(dim, n) for k in nodes}
    row = co.sum(1, keepdims=True)
    col = co.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co * total) / (row * col))
    ppmi = np.nan_to_num(np.maximum(pmi, 0), nan=0.0, posinf=0.0, neginf=0.0)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    d = min(dim, n - 1)
    emb = u[:, :d] * np.sqrt(s[:d])
    return {k: [round(float(x), 5) for x in emb[idx[k]]] for k in nodes}


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
