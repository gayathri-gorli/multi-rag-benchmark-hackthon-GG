"""CO-STAR system prompt (verbatim from spec) and practice-area guardrails."""
import re
from enum import Enum


class PracticeArea(str, Enum):
    TAX = "Tax & Regulatory Services"
    GRC = "Risk Advisory & Governance (GRC)"
    DEALS = "Deal Advisory & Corporate Finance"


COSTAR_PROMPT = """\
C (Context): You are a Senior Principal Advisory AI Assistant serving PwC India domain consultants.
O (Objective): Deliver accurate, verifiable, and cited compliance analysis from indexed enterprise corpora.
S (Style): Professional, analytical, structured, and auditable corporate advisory tone.
T (Tone): Authoritative, objective, risk-aware, and precise.
A (Audience): Partners, Enterprise Clients, Tax Controllers, Chief Risk Officers, and Audit Committees.
R (Response): Structured markdown with inline statutory/clause citations and explicit gap disclosures.

Rules: answer ONLY from the numbered sources provided. Cite each claim inline as [S1], [S2].
If the sources do not support an answer, say so explicitly under a "Gap Disclosure" heading; never guess."""

AREA_FOCUS = {
    PracticeArea.TAX: "Direct Tax (CBDT notifications, Income Tax Act), Indirect Tax (CBIC circulars, 57th GST Council rulings, ISD rules, e-invoicing).",
    PracticeArea.GRC: "RBI Master Directions (KYC/PMLA), SEBI BRSR guidelines, ISO 31000 risk management frameworks, internal audit controls.",
    PracticeArea.DEALS: "M&A Share Purchase Agreements (SPA), Companies Act 2013 valuation standards, FEMA/FCRA foreign investment rules.",
}

AREA_RULES = {
    PracticeArea.TAX: "Must cite exact Section/Rule numbers. Distinguish between prospective amendments and retrospective clarifications.",
    PracticeArea.GRC: "Flag non-compliance penalties immediately. Strictly map controls to specific regulatory obligations.",
    PracticeArea.DEALS: "Highlight indemnities, material adverse change (MAC) clauses, and financial exposure thresholds.",
}


def build_system_prompt(area: PracticeArea) -> str:
    return (
        f"{COSTAR_PROMPT}\n\nPractice Area: {area.value}\n"
        f"Regulatory focus: {AREA_FOCUS[area]}\nEnforced guardrails: {AREA_RULES[area]}"
    )


_SECTION_RE = re.compile(r"\b(section|sec\.|rule|regulation|clause|article)\s*\d+[A-Za-z]*", re.I)
_PENALTY_RE = re.compile(r"penalt|fine|prosecution|punish|non-?compliance", re.I)
_DEAL_RE = re.compile(r"indemnit|material adverse|\bMAC\b|exposure|threshold|cap\b|basket", re.I)
_PII = [
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b"), "[GSTIN]"),
]


def redact_pii(text: str) -> str:
    # GSTIN first: it contains a PAN, which would otherwise be replaced partially.
    for pat, tag in (_PII[2], _PII[0], _PII[1]):
        text = pat.sub(tag, text)
    return text


def check_answer(answer: str, area: PracticeArea) -> list[str]:
    """Post-generation guardrail flags (empty list = passed)."""
    flags = []
    if "[S" not in answer and "gap disclosure" not in answer.lower():
        flags.append("No source citations and no gap disclosure.")
    if area is PracticeArea.TAX and not _SECTION_RE.search(answer):
        flags.append("Tax guardrail: no exact Section/Rule number cited.")
    if area is PracticeArea.GRC and not _PENALTY_RE.search(answer):
        flags.append("GRC guardrail: no non-compliance penalty flagged (confirm none apply).")
    if area is PracticeArea.DEALS and not _DEAL_RE.search(answer):
        flags.append("Deals guardrail: no indemnity/MAC/exposure threshold highlighted.")
    return flags
