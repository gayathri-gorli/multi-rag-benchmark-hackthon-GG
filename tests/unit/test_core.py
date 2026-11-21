import pytest

from core.citations import build_context, resolve_citations
from core.config import Hyperparams
from core.guardrails import PracticeArea, build_system_prompt, check_answer, redact_pii
from core.rag.fusion import rrf_fuse
from core.tracking.tokens import TokenLedger, UsageRow


def test_hyperparam_defaults_match_spec():
    hp = Hyperparams()
    assert (hp.chunk_size, hp.chunk_overlap, hp.hyde_n, hp.hop_k, hp.rrf_alpha) == (600, 60, 3, 2, 60)


def test_rrf_math_and_determinism():
    out = dict(rrf_fuse([["a", "b"], ["b", "c"]], alpha=60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["a"] == pytest.approx(1 / 61)
    assert [d for d, _ in rrf_fuse([["a", "b"], ["b", "c"]])][0] == "b"


def test_rrf_ignores_duplicates_within_list():
    assert dict(rrf_fuse([["a", "a"]]))["a"] == pytest.approx(1 / 61)


def test_costar_prompt_contains_area_rules():
    p = build_system_prompt(PracticeArea.TAX)
    assert "Senior Principal Advisory AI Assistant" in p and "exact Section/Rule numbers" in p
    assert "MAC" in build_system_prompt(PracticeArea.DEALS)


def test_guardrail_flags():
    assert check_answer("See Section 115BAA [S1].", PracticeArea.TAX) == []
    assert any("Section/Rule" in f for f in check_answer("Tax is due [S1].", PracticeArea.TAX))
    assert any("citations" in f for f in check_answer("Tax is due.", PracticeArea.GRC))
    # A gap disclosure satisfies the citation rule (the Deals-specific rule still applies).
    flags = check_answer("## Gap Disclosure\nnone", PracticeArea.DEALS)
    assert not any("citations" in f for f in flags)


def test_pii_redaction():
    t = redact_pii("PAN ABCDE1234F, GSTIN 27ABCDE1234F1Z5, Aadhaar 1234 5678 9012")
    assert "ABCDE1234F" not in t and "[GSTIN]" in t and "[AADHAAR]" in t


def test_citation_resolution_flags_invalid_tags():
    chunks = [{"text": "x", "metadata": {"source": "a.pdf", "page": 2, "chunk_id": "c1"}}]
    cites = resolve_citations("Claim [S1] and [S5].", chunks)
    assert cites[0]["valid"] and cites[0]["source"] == "a.pdf" and not cites[1]["valid"]
    assert "[S1] (a.pdf, p.2)" in build_context(chunks)


def test_token_ledger_totals_and_estimate_flag():
    led = TokenLedger(price_per_1k_in=1, price_per_1k_out=2)
    led.record(UsageRow("r", "Standard RAG", "synthesis", "m", 1000, 500, 10))
    led.record(UsageRow("r", "HyDE RAG", "expansion", "m", 10, 5, 1, estimated=True))
    t = led.totals(engine="Standard RAG")
    assert t["total_tokens"] == 1500 and t["cost_usd"] == pytest.approx(2.0) and not t["any_estimated"]
    assert led.totals()["any_estimated"]
