"""Golden questions for the bundled SYNTHETIC sample documents (data/<area>/).
facts: ground-truth regulatory facts; each fact is a list of acceptable phrasings (any one counts).
This set is tiny and synthetic: it demonstrates the benchmark mechanics, it is not evidence of real-world quality."""
from core.guardrails import PracticeArea

KEYWORD, MULTIHOP, VOCAB = "keyword", "multi-hop", "vocabulary-mismatch"


def q(qid, category, question, *facts):
    return {"id": qid, "category": category, "question": question, "facts": [list(f) for f in facts]}


GOLDEN = {
    PracticeArea.TAX: [
        q("tax-1", KEYWORD, "What is the concessional tax rate under Section 115ZA?", ["22 percent"], ["115ZA"]),
        q("tax-2", MULTIHOP, "If a company opts for Section 115ZA, what happens to its MAT credit and which authority clarified this?",
          ["mat credit lapses", "unabsorbed mat credit lapses"], ["central board of direct taxes", "cbdt"]),
        q("tax-3", VOCAB, "Which form must a company file to choose the lower corporate tax regime, and what fee applies if it is missed?",
          ["form 10-za"], ["234zf"]),
        q("tax-4", KEYWORD, "Can a buyer claim ITC when the supplier has not filed GSTR-1?", ["gstr-1"], ["cannot claim itc"]),
        q("tax-5", MULTIHOP, "What interest and penalty apply if input tax credit is claimed in violation of Rule 36Z?",
          ["section 50"], ["section 122"]),
    ],
    PracticeArea.GRC: [
        q("grc-1", KEYWORD, "How often must high-risk customers undergo KYC updation?", ["every 2 years"]),
        q("grc-2", MULTIHOP, "Which internal control covers suspicious transaction reporting and which RBI paragraph and body does it map to?",
          ["c-18"], ["paragraph 24"], ["fiu-ind"]),
        q("grc-3", VOCAB, "What is the penalty for failing to report suspicious activity and how quickly must it be reported?",
          ["rs 1 crore"], ["7 working days"]),
        q("grc-4", MULTIHOP, "Who must a Severity 1 failure of a money laundering control be escalated to, and how fast?",
          ["chief risk officer"], ["24 hours"]),
    ],
    PracticeArea.DEALS: [
        q("deal-1", KEYWORD, "What is the Seller's liability cap and the claim basket under the SPA?", ["rs 40 crore"], ["rs 1 crore"]),
        q("deal-2", VOCAB, "How is a material adverse change defined in the agreement?", ["15 percent"], ["ebitda"]),
        q("deal-3", MULTIHOP, "Does the purchase price satisfy the FEMA pricing floor, and what was the registered valuer's fair value?",
          ["rs 195 crore"], ["rs 200 crore"]),
        q("deal-4", KEYWORD, "How long is the valuation report valid?", ["90 days"]),
    ],
}

# (surface form as it appears in text, canonical entity) - for Entity-Relation Resolution Accuracy.
GOLD_ALIASES = {
    PracticeArea.TAX: [("Income Tax Dept", "CBDT"), ("Tax Authority", "CBDT"),
                       ("Central Board of Direct Taxes", "CBDT"), ("CBDT", "CBDT"),
                       ("Central Board of Indirect Taxes and Customs", "CBIC")],
    PracticeArea.GRC: [("Reserve Bank of India", "RBI"), ("RBI", "RBI"),
                       ("Financial Intelligence Unit India", "FIU-IND"),
                       ("Prevention of Money Laundering Act", "PMLA"), ("PMLA", "PMLA")],
    PracticeArea.DEALS: [("Foreign Exchange Management Act", "FEMA"), ("FEMA", "FEMA"),
                         ("Companies Act 2013", "Companies Act, 2013"),
                         ("Foreign Contribution Regulation Act", "FCRA")],
}
