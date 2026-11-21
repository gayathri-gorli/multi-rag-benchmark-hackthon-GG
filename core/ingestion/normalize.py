"""Spec step 3 (entity part): normalize aliases to canonical entity names,
e.g. 'CBDT', 'Income Tax Dept', 'Tax Authority' -> 'CBDT'."""
import re

ALIASES = {
    "cbdt": "CBDT", "central board of direct taxes": "CBDT", "income tax dept": "CBDT",
    "income tax department": "CBDT", "tax authority": "CBDT", "the tax authority": "CBDT",
    "cbic": "CBIC", "central board of indirect taxes and customs": "CBIC",
    "central board of indirect taxes": "CBIC",
    "rbi": "RBI", "reserve bank of india": "RBI", "reserve bank": "RBI",
    "sebi": "SEBI", "securities and exchange board of india": "SEBI",
    "mca": "MCA", "ministry of corporate affairs": "MCA",
    "gst council": "GST Council", "goods and services tax council": "GST Council",
    "fiu ind": "FIU-IND", "financial intelligence unit india": "FIU-IND",
    "pmla": "PMLA", "prevention of money laundering act": "PMLA",
    "kyc": "KYC", "know your customer": "KYC",
    "income tax act": "Income Tax Act, 1961", "income tax act 1961": "Income Tax Act, 1961",
    "it act": "Income Tax Act, 1961",
    "companies act": "Companies Act, 2013", "companies act 2013": "Companies Act, 2013",
    "fema": "FEMA", "foreign exchange management act": "FEMA",
    "fcra": "FCRA", "foreign contribution regulation act": "FCRA",
}


def key(text: str) -> str:
    """Lowercase, punctuation-stripped, whitespace-collapsed form used for matching."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def canonical(name: str) -> str:
    return ALIASES.get(key(name), " ".join(name.split()))


def node_key(name: str) -> str:
    return key(canonical(name))


def aliases_in(query: str) -> list[str]:
    """Canonical names of any known aliases that appear as whole phrases in the query."""
    padded = f" {key(query)} "
    return sorted({canon for alias, canon in ALIASES.items() if f" {alias} " in padded})
