"""Runs the golden questions through each engine with the same hyperparameters and records the 4 KPIs plus tokens/cost."""
import uuid

import pandas as pd

from core.eval.kpis import KG_CHUNK_ID, entity_resolution_accuracy, faithfulness, latency_breakdown, retrieval_context_recall
from core.rag.factory import build_engine

SPEC_REFERENCE_LATENCY_MS = {"Standard RAG": 450, "HyDE RAG": 1200, "Graph RAG": 850}
SPEC_QUALITATIVE = pd.DataFrame({
    "Metric": ["Simple keyword queries", "Multi-hop relational queries", "Domain vocabulary mismatch",
               "Average latency (ms)", "Hallucination risk", "Best enterprise use case"],
    "Standard RAG": ["High precision / low latency", "Poor (misses indirect relationships)",
                     "Poor (relies on exact vector distance)", "~450", "Moderate (out-of-context chunks)",
                     "Single document lookup (SOPs)"],
    "HyDE RAG": ["Moderate (HyDE overhead)", "Moderate (semantic expansion)", "Superior (synthesizes domain terms)",
                 "~1,200", "Moderate (synthetic passage drift)", "Vague regulatory inquiries"],
    "Graph RAG": ["Moderate (graph overhead)", "Superior (explores k-hop edges)",
                  "High (entity resolution maps synonyms)", "~850", "Lowest (strict entity-relation grounding)",
                  "Complex Tax / M&A due diligence"],
})
METRICS = ["rcr", "faithfulness", "latency_ms", "retrieval_ms", "graph_ms", "llm_ms",
           "prompt_tokens", "completion_tokens", "cost_usd"]


def run_benchmark(*, engines, area, questions, store, graph, llm, hp, ledger, judge_llm=None, retrieval_only=False,
                  repeats=1, active_entities=None, on_row=None) -> list[dict]:
    from core.tracking.tokens import make_langchain_callback

    rows = []
    for engine in engines:
        for rep in range(repeats):
            for qn in questions:
                run_id = uuid.uuid4().hex[:8]
                row = {"engine": engine, "repeat": rep + 1, "qid": qn["id"], "category": qn["category"],
                       "question": qn["question"], "error": None}
                try:
                    rag = build_engine(
                        engine, store=store, llm=llm, area=area, hp=hp, graph=graph, active_entities=active_entities,
                        callback_factory=lambda node, e=engine, r=run_id: [make_langchain_callback(ledger, r, e, node)])
                    out = rag.run(qn["question"], dry_run=retrieval_only)
                    row["rcr"] = retrieval_context_recall(out["chunks"], qn["facts"])
                    row["answer"] = None if retrieval_only else out["answer"]
                    row["sources"] = sorted({c["metadata"]["source"] for c in out["chunks"]})
                    row["guardrail_flags"] = len(out.get("guardrail_flags", []))
                    row["faithfulness"] = None
                    if judge_llm is not None and not retrieval_only:
                        cb = [make_langchain_callback(ledger, run_id, engine, "judge")]
                        row["faithfulness"] = faithfulness(out["answer"], out["chunks"], judge_llm, cb)["score"]
                    t = ledger.totals(run_id=run_id)
                    # Judge calls happen after the pipeline finished, so exclude them from pipeline LLM latency.
                    llm_ms = sum(r.latency_ms for r in ledger.rows if r.run_id == run_id and r.node != "judge")
                    row |= latency_breakdown(out, llm_ms)
                    row |= {"prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
                            "cost_usd": t["cost_usd"]}
                except Exception as e:  # one failing question must not abort the whole benchmark
                    row["error"] = f"{type(e).__name__}: {e}"
                rows.append(row)
                if on_row:
                    on_row(row, rows)
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if not r.get("error")])
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in METRICS if c in df]
    out = df.groupby("engine")[cols].mean(numeric_only=True)
    out["questions"] = df.groupby("engine")["qid"].count()
    out["errors"] = pd.Series({e: sum(1 for r in rows if r["engine"] == e and r.get("error")) for e in out.index})
    return out


def aggregate_by_category(rows: list[dict], metric: str = "rcr") -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if not r.get("error")])
    if df.empty or metric not in df:
        return pd.DataFrame()
    return df.pivot_table(index="category", columns="engine", values=metric, aggfunc="mean")


def resolution_accuracy(graph, area, gold_aliases) -> float | None:
    return entity_resolution_accuracy(graph, gold_aliases.get(area, []))
