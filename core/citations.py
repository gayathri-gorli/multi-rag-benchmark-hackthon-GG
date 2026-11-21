"""Source numbering for the prompt and mapping [S#] tags in answers back to provenance."""
import re


def build_context(chunks: list[dict]) -> str:
    """chunks: dicts with 'text' and 'metadata'. Returns numbered source block."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        loc = f"{m.get('source', 'unknown')}" + (f", p.{m['page']}" if m.get("page") is not None else "")
        parts.append(f"[S{i}] ({loc})\n{c['text']}")
    return "\n\n".join(parts)


def resolve_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Return provenance for every [S#] actually used in the answer (invalid tags flagged)."""
    used = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", answer)})
    out = []
    for n in used:
        if 1 <= n <= len(chunks):
            m = chunks[n - 1]["metadata"]
            out.append({"tag": f"S{n}", "source": m.get("source"), "page": m.get("page"),
                        "chunk_id": m.get("chunk_id"), "snippet": chunks[n - 1]["text"][:240],
                        "valid": True})
        else:
            out.append({"tag": f"S{n}", "valid": False})
    return out
