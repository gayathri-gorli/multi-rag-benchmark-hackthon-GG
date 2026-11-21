# PwC India Enterprise Multi-RAG Comparative Benchmark

A Streamlit application that compares **Standard RAG**, **Multi-HyDE RAG** and **Graph RAG** (LangChain + LangGraph)
on domain-grounded question answering across three practice areas: Tax & Regulatory Services, Risk Advisory &
Governance (GRC), and Deal Advisory & Corporate Finance.

## The five tabs
1. **Data Ingestion & Multi-RAG Chat** - upload PDF/DOCX (also JSON/CSV/TXT), pick practice area and engine, chat with inline `[S#]` source citations and live token/cost counters.
2. **Spherical Knowledge Graph** - entities and relations on a sphere (WebGL-free rotatable projection by default; optional true 3D). Selecting entities/relations restricts what Graph RAG may retrieve.
3. **LangGraph Workflow Canvas** - Query Expansion -> Hybrid Search -> RRF Fusion -> Rerank -> Synthesis. Toggle nodes and transitions (edges) and preview the effect on retrieved context.
4. **Hyperparameters & KPI Telemetry** - chunk size/overlap (600/60), Multi-HyDE N (3), graph hop radius k (2), RRF constant (60); the four KPIs.
5. **Comparative Benchmark** - runs a golden question set through all three engines; recall, faithfulness, entity-resolution accuracy, latency breakdown, tokens and cost; CSV export.

## How the three engines differ
| | Query expansion | Hybrid search | Fusion |
|---|---|---|---|
| Standard | identity | vector + BM25 | RRF |
| Multi-HyDE | N synthetic passages from distinct regulatory angles | vector + BM25 over query and passages | RRF |
| Graph RAG | alias-aware entity linking | vector + BM25 + cyclic k-hop knowledge-graph traversal (LangGraph loop) | RRF |

All answers use a CO-STAR system prompt plus per-practice-area guardrails, and cite sources or state a gap disclosure.

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # add OPENROUTER_API_KEY
.venv/bin/streamlit run app.py   # http://localhost:8501
.venv/bin/python -m pytest -q    # offline tests (fake LLM / embeddings)
```
In Tab 1 click **Load SYNTHETIC sample documents** to try it without your own corpus.

## Providers are configurable
`LLM_PROVIDER` = `openrouter` (Claude) | `azure_openai`; `EMBEDDING_PROVIDER` = `auto` | `local` | `mock` | `openrouter` | `hf` | `azure_openai`.
RAG code only depends on LangChain interfaces, so switching providers is a config change. The vector index records which
embedding model built it and refuses to open under a different one.

## Honest limitations
- **Embeddings:** the spec names `bge-large-en-v1.5` (1024-d). `auto` uses a locally cached `bge-small-en-v1.5` (384-d) through a small numpy runtime, or a clearly labelled **mock** mode. Use `openrouter`/`hf`/`azure_openai` for the spec model. The numpy runtime was sanity-checked, not compared against a reference implementation.
- **Node2Vec:** implemented as biased random walks + PPMI/SVD, not the reference library; embeddings are stored but not used for retrieval.
- **Rerank:** a pass-through; no reranker model is wired in.
- **Benchmark:** the bundled golden set and documents are small and **synthetic** (section numbers are fictional). It demonstrates the mechanics; it is not evidence of real-world accuracy. With so few chunks every engine retrieves everything, so recall does not discriminate between engines.
- **Latency:** dominated by hosted-LLM calls, far above the spec's reference figures.
- Cost: LLM entity extraction (Graph RAG ingestion) and HyDE passage generation consume tokens; the ledger reports provider-counted usage.
