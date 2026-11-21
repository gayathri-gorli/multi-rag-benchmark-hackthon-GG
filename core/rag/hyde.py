"""Multi-HyDE: N synthetic passages from distinct regulatory angles replace/augment the query
for retrieval. Synthesis still answers the ORIGINAL question from real retrieved chunks."""
from core.rag.standard import RAGState, StandardRAG

ANGLES = [
    "the operative statutory or regulatory text (section / rule / clause wording)",
    "an official circular, notification or clarification issued by the regulator",
    "practical compliance guidance: procedure, thresholds, deadlines and penalties for non-compliance",
    "an authoritative commentary or tribunal/court interpretation",
    "a typical contractual or policy clause implementing the requirement",
    "a worked example or illustration an advisor would give a client",
]

HYDE_PROMPT = """You are helping retrieve documents for the practice area "{area}".
Write a concise passage (90-130 words) that reads like {angle}, and that would directly answer the question below.
Use precise domain terminology. Do not mention that this is hypothetical. Output only the passage.

Question: {query}"""


class HyDERAG(StandardRAG):
    engine = "HyDE RAG"

    def _expand(self, s: RAGState) -> RAGState:
        passages = []
        for i in range(self.hp.hyde_n):
            angle = ANGLES[i % len(ANGLES)]
            try:
                p = self._invoke("query_expansion", [("human", HYDE_PROMPT.format(
                    area=self.area.value, angle=angle, query=s["query"]))]).strip()
            except Exception:
                continue  # a failed hypothesis must not kill the query
            if p:
                passages.append(p)
        # Keep the original query as a safety net against synthetic passage drift.
        return {"queries": [s["query"]] + passages}
