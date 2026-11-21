"""The four spec KPIs.
1 Retrieval Context Recall (deterministic phrase match)   2 Faithfulness (LLM judge, temperature 0)
3 Entity-Relation Resolution Accuracy (graph linking vs gold aliases)   4 End-to-end latency + breakdown."""
import json
import re

from core.ingestion.normalize import node_key

KG_CHUNK_ID = "kg"


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.\- ]", " ", t.lower()).split())


def retrieval_context_recall(chunks: list[dict], facts: list[list[str]]) -> float | None:
    """Share of ground-truth facts present in the retrieved document chunks.
    The synthetic knowledge-graph evidence chunk is excluded so engines are compared on documents only."""
    if not facts:
        return None
    text = _norm(" ".join(c["text"] for c in chunks if c["metadata"].get("chunk_id") != KG_CHUNK_ID))
    return sum(any(_norm(alt) in text for alt in fact) for fact in facts) / len(facts)


JUDGE_PROMPT = """You are a strict fact-checker. Break the ANSWER into its individual factual claims (ignore headings, \
formatting and advice phrased as recommendations). For each claim decide whether the SOURCES directly support it.
Return ONLY JSON: {{"claims": [{{"claim": str, "supported": true|false}}]}}

SOURCES:
{sources}

ANSWER:
{answer}"""


def faithfulness(answer: str, chunks: list[dict], llm, callbacks=None) -> dict:
    """Ratio of answer claims supported by the retrieved sources (0.0-1.0). None when the answer has no claims
    (e.g. a gap disclosure) so it does not inflate or deflate the average."""
    if not chunks or "gap disclosure" in answer.lower()[:40]:
        return {"score": None, "claims": []}
    sources = "\n\n".join(f"[S{i}] {c['text']}" for i, c in enumerate(chunks, 1))
    raw = llm.invoke(JUDGE_PROMPT.format(sources=sources, answer=answer), config={"callbacks": callbacks or []}).content
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        claims = [c for c in json.loads(m.group(0)).get("claims", []) if isinstance(c, dict) and "supported" in c]
    except (AttributeError, json.JSONDecodeError):
        return {"score": None, "claims": [], "error": "judge returned unparseable output"}
    if not claims:
        return {"score": None, "claims": []}
    return {"score": sum(bool(c["supported"]) for c in claims) / len(claims), "claims": claims}


def entity_resolution_accuracy(graph, gold: list[tuple[str, str]]) -> float | None:
    """For each (surface form, canonical entity): does linking a query that uses the surface form land on the
    canonical node that actually exists in the ingested knowledge graph?"""
    if not gold:
        return None
    hits = 0
    for surface, canonical in gold:
        target = node_key(canonical)
        if target in graph.G and target in graph.link_entities(f"What did the {surface} state?"):
            hits += 1
    return hits / len(gold)


def latency_breakdown(out: dict, llm_ms: float) -> dict:
    """total = retrieval + graph traversal + LLM inference (LLM time comes from the token ledger's per-call latency)."""
    total = out["total_ms"]
    graph = out.get("timings", {}).get("graph_traversal", 0.0)
    llm = min(llm_ms, total)
    return {"latency_ms": total, "graph_ms": graph, "llm_ms": llm, "retrieval_ms": max(total - llm - graph, 0.0)}
