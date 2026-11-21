"""Token ledger: per-call usage rows with provider-reported counts preferred and
an explicit 'estimated' flag when we had to fall back to local counting."""
import time
from dataclasses import dataclass, field, asdict
from threading import Lock

from core.tokenizer import count_tokens


@dataclass
class UsageRow:
    run_id: str
    engine: str
    node: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    estimated: bool = False
    ts: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenLedger:
    def __init__(self, price_per_1k_in: float = 0.003, price_per_1k_out: float = 0.015):
        self.rows: list[UsageRow] = []
        self._lock = Lock()
        self.price_in, self.price_out = price_per_1k_in, price_per_1k_out
        self.listeners = []  # callables(row) for live UI updates

    def record(self, row: UsageRow) -> None:
        with self._lock:
            self.rows.append(row)
        for fn in self.listeners:
            fn(row)

    def record_text(self, run_id, engine, node, model, prompt: str, completion: str, latency_ms: float):
        """Fallback when the provider returned no usage metadata."""
        self.record(UsageRow(run_id, engine, node, model, count_tokens(prompt),
                             count_tokens(completion), latency_ms, estimated=True))

    def totals(self, engine: str | None = None, run_id: str | None = None) -> dict:
        rows = [r for r in self.rows
                if (engine is None or r.engine == engine) and (run_id is None or r.run_id == run_id)]
        pin = sum(r.prompt_tokens for r in rows)
        pout = sum(r.completion_tokens for r in rows)
        return {
            "prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout,
            "cost_usd": pin / 1000 * self.price_in + pout / 1000 * self.price_out,
            "calls": len(rows), "any_estimated": any(r.estimated for r in rows),
        }

    def to_records(self) -> list[dict]:
        return [asdict(r) | {"total": r.total} for r in self.rows]


def make_langchain_callback(ledger: TokenLedger, run_id: str, engine: str, node: str):
    """LangChain callback that records provider-reported usage per LLM call."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _Handler(BaseCallbackHandler):
        def __init__(self):
            self._start = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
            self._start[run_id] = (time.perf_counter(), messages)

        def on_llm_start(self, serialized, prompts, *, run_id, **kw):
            self._start[run_id] = (time.perf_counter(), prompts)

        def on_llm_end(self, response, *, run_id, **kw):
            t0, inp = self._start.pop(run_id, (time.perf_counter(), None))
            ms = (time.perf_counter() - t0) * 1000
            usage = (response.llm_output or {}).get("token_usage") or {}
            if not usage:
                try:
                    meta = response.generations[0][0].message.usage_metadata or {}
                    usage = {"prompt_tokens": meta.get("input_tokens"),
                             "completion_tokens": meta.get("output_tokens")}
                except Exception:
                    usage = {}
            model = (response.llm_output or {}).get("model_name", "unknown")
            if usage.get("prompt_tokens") is not None:
                ledger.record(UsageRow(run_id_s, engine, node, model, usage["prompt_tokens"],
                                       usage.get("completion_tokens") or 0, ms))
            else:
                text = response.generations[0][0].text
                ledger.record_text(run_id_s, engine, node, model, str(inp), text, ms)

    run_id_s = run_id
    return _Handler()
