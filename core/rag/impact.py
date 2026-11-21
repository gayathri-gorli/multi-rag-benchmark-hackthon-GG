"""Quantifies what a workflow variation (node/edge toggles) changed versus the full pipeline."""
from core.rag.standard import EDGE_EFFECTS


def _ids(out: dict) -> list[str]:
    return [c["metadata"]["chunk_id"] for c in out.get("chunks", [])]


def compare_runs(baseline: dict, variant: dict) -> dict:
    b, v = _ids(baseline), _ids(variant)
    union = set(b) | set(v)
    return {
        "baseline_chunks": b, "variant_chunks": v,
        "kept": [x for x in b if x in v], "dropped": [x for x in b if x not in v],
        "added": [x for x in v if x not in b],
        "overlap": (len(set(b) & set(v)) / len(union)) if union else 1.0,
        "order_changed": [x for x in b if x in v] != [x for x in v if x in b],
        "empty": not v,  # nothing reached synthesis -> gap disclosure
        "skipped_nodes": variant.get("skipped", []),
        "severed_edges": [(e, EDGE_EFFECTS[e]) for e in variant.get("severed", [])],
        "withheld_fields": variant.get("blocked", []),
        "timings_ms": {"baseline": baseline.get("timings", {}), "variant": variant.get("timings", {})},
    }
