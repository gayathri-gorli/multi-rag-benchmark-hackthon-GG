"""Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (alpha + rank), rank starting at 1."""
from collections import defaultdict


def rrf_fuse(ranked_lists: list[list[str]], alpha: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        seen = set()
        for rank, doc_id in enumerate(lst, start=1):
            if doc_id in seen:  # ignore duplicates within one list
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (alpha + rank)
    # Stable tie-break on id so results are deterministic.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
