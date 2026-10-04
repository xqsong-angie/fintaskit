# FinTasKit

A hands-on, production-grade RAG pipeline built around real financial filings (Wells Fargo, Tesla, AMD quarterly reports). Seven notebooks walk from PDF parsing through chunking, hybrid retrieval, LangGraph generation, Ragas evaluation, and FastAPI + LangSmith observability.

[![CI](https://github.com/zyziyun/fin-rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/zyziyun/fin-rag-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Keys
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-...   (LANGSMITH_* optional, only needed for notebook 06)

# 3. PDFs — download into data/uploads/  (URLs in Footnotes below)
#    wells_fargo.pdf, tesla.pdf, amd.pdf

# 4. Warm the cache (once, ~3.5 min, ~$0.025)
python scripts/precompute_cache.py \
    --inputs data/uploads/wells_fargo.pdf \
             data/uploads/tesla.pdf \
             data/uploads/amd.pdf

# 5. Open the first notebook
jupyter lab notebooks/00_quickstart.ipynb
```

If you have a pre-built `cache_bundle.zip`, unzip it at the repo root instead of step 4 — the cache is content-addressed, so a teammate's bundle works on your machine bit-for-bit.

---

## Lab progression

| # | Notebook | Topic |
|---|---|---|
| 00 | `quickstart` | 30 lines of LangChain LCEL — feel the chain before abstracting it |
| 01 | `parsing` | Multi-stage ingestion: PyMuPDF loader → struct parser → VLM captioner, with content-addressed caching |
| 02 | `chunking` | Three chunking strategies (fixed-size, recursive, parent-child) compared via a `CoverageDiagnostic` tool |
| 03 | `retrieval` | Vector + BM25 + RRF hybrid, with parent-child swap at retrieval time |
| 04 | `generation` | LangGraph state machine: short-query bypass, retrieve-generate, refusal-on-empty |
| 05 | `evaluation` | Ragas 4-metric eval + custom claim-level `HallucinationDetector` for compliance |
| 06 | `observability` | `@traceable` instrumentation + LangSmith dashboard + FastAPI `/ingest` and `/query` endpoints |

---

## Run it as an app

```bash
# Gradio UI (chat + citations + live config switching + claim-level verification)
python notebooks/07_ui_demo.py          # → http://127.0.0.1:7860

# or the HTTP service
uvicorn src.api.server:app --port 8000  # → http://127.0.0.1:8000/docs
```

Both sit on `src/api/app_factory.RAGStack`, which assembles chunker → retriever →
generator from a single `PipelineConfig`. Change `chunk_size`, `parent_size`,
retriever or model at runtime and the stack rebuilds only the affected part —
re-chunking reuses the parse and embedding caches, so a config sweep costs
~$0 in API calls. That is deliberate: **the config you benchmark is the config
the UI serves.** The same `PipelineConfig` drives `05_evaluation`, so an eval
result and the deployed behaviour cannot drift apart.

Multi-document scoping is explicit: `/query` accepts `document_ids` and the
retrieval path filters on `document_id` (`QueryPipeline._scoped_retrieve`).
Omitting it searches the whole collection — which is exactly how the
cross-document leakage demonstrated in `03_retrieval` Step 3 happens in
production.

---

## Architecture

```
                    [ User uploads PDF ]
                            ↓
┌──────────────────────────────────────────────┐
│  IngestionPipeline                           │
│   Loader → Parser → Captioner (VLM)          │
│   3-layer content-addressed cache            │
└──────────────────────────────────────────────┘
                            ↓
                       Document
                            ↓
                       Chunker (Strategy: 3 implementations)
                            ↓
                  Vector + BM25 indexes
                            ↓
                    HybridRetriever (RRF)  ← optional parent expansion
                            ↓
              QueryPipeline (LangGraph state machine)
                            ↓
                     Answer + citations
                            ↓
            Ragas + HallucinationDetector → metrics
                            ↓
                   LangSmith → dashboard
```

---

## Repo structure

```
fin-rag-lab/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── scripts/
│   └── precompute_cache.py     # one-shot, generates shareable cache_bundle
├── src/
│   ├── core/                   # domain models, abstract interfaces, cache, config
│   ├── loaders/                # Stage 1: PDF → page dicts
│   ├── parsers/                # Stage 2: structured Block detection
│   ├── captioners/             # Stage 3: VLM table/image captioning
│   ├── chunkers/               # 3 strategies, all `Runnable`
│   ├── retrievers/             # Vector / BM25 / Hybrid (RRF)
│   ├── generators/             # Citation-aware RAG generator
│   ├── evaluators/             # Ragas + claim-level hallucination + coverage
│   ├── pipelines/
│   │   ├── ingestion.py        # IngestionPipeline orchestrator
│   │   └── query.py            # LangGraph state machine
│   ├── observability/          # CostTracker, LangSmith helpers
│   └── api/
│       ├── app_factory.py      # RAGStack: config-driven assembly (single source of truth)
│       └── server.py           # FastAPI shell over RAGStack
├── notebooks/                  # 00–06 + 07_ui_demo.py, see Lab progression above
├── data/
│   ├── uploads/                # PDFs go here
│   └── golden_set/             # 30 financial QA across 5 categories
├── cache/                      # auto-managed, content-addressed
└── tests/                      # 66 unit + integration tests
```

---

## Design choices

### Strategy pattern + dependency injection everywhere
Every stage has a `BaseX` abstract class; concrete implementations inherit and override. You can swap PyMuPDF for LlamaParse, or OpenAI embeddings for Cohere, without touching the pipeline.

### Chunkers inherit from `Runnable`
First-class LangChain ecosystem citizens — you can do `chunker | embedder | indexer`, or `.batch()` / `.stream()`. But chunkers also keep their own semantic methods (`chunk()`, `chunk_with_parents()`) so the architecture isn't married to LangChain's evolving API.

### Content-addressed caching
Cache keys are SHA256 of inputs. Three guarantees:
1. **Correctness** — change input, change key. No stale data, ever.
2. **Sharable** — zip the cache, ship to a teammate, instant warm-up on their machine.
3. **Reproducible** — same inputs → same key on every machine, every time.

Chunk IDs derive from `sha256(source_bytes)[:12]` + a positional index, not
`uuid4`. Re-ingesting the same PDF therefore produces identical chunk IDs, which
makes indexing idempotent — the API rejects a duplicate by `source_hash` instead
of silently doubling the index.

### Scoping is a retrieval parameter, not an afterthought
A single shared collection is convenient and unsafe: ask a Wells Fargo question
and a Tesla chunk can win the ranking. `03_retrieval` reproduces the leak live
and closes it with a metadata filter. Because BM25 runs in-process and cannot do
server-side `where`, the portable implementation over-fetches (`filter_overfetch`)
and post-filters on `document_id` — the honest cost of one code path across both
retrievers.

### Parent chunks need page numbers as much as children do
`ParentChildChunker` reassembles parents from a flattened document, so the
`chunk → page` mapping is a `str.find` that has to survive two things: splitter
whitespace stripping, and the fact that consecutive parents **overlap** — child
N+1 starts *before* the end of child N. Anchoring the search on the chunk's tail
rather than its head is what makes every parent resolve to a page instead of
half of them coming back `None`. A citation with no page number is a citation
nobody can verify.

### VLM captioning eagerly at ingestion (not lazily at query)
Read-side latency is the customer-facing metric. We pay the VLM cost once at ingest (cached forever) so query latency stays p99 < 1s. Lazy VLM at query time would put a 2–3 second LLM call on the hot path.

### LangGraph for query (not just an LCEL chain)
Production RAG isn't linear:
- Short factual queries → top-3 chunks suffice (save cost)
- Long analytical queries → top-8 with parent expansion (more context)
- Empty retrieval → don't call the LLM (refuse → save $ + prevent hallucination)

That's a state machine with conditional edges. Every node is `@traceable`, so each branch shows up independently in LangSmith.

### Custom `HallucinationDetector` on top of Ragas
Ragas faithfulness gives one score per answer. Compliance-critical domains (finance, legal) need to point at *which* claim is unsupported. The detector decomposes an answer into atomic claims and verifies each against the joined context with NLI-style prompts, returning per-claim verdicts. Actionable in a UI: highlight the unsupported sentence in red.

### `compare_strategies` instead of hard-coding "use parent-child"
The right chunking strategy depends on your query distribution. We ship a tool, not a recommendation. `CoverageDiagnostic` quantifies "did retrieval get the data we needed?" in 30 lines. `compare_strategies(chunkers, doc, queries, retriever_factory)` lets a team make a measurement-driven decision in 5 minutes on a new corpus.

---

## What we observed on the real PDFs

| Document | Pages | Text blocks | Tables | Images | Time | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Wells Fargo Q4 2025 | 12 | 418 | **0** | 1 | 6s | $0.0004 |
| Tesla Q1 2026 | 31 | 530 | 11 | 10 | 95s | $0.0061 |
| AMD Q4 2025 | 34 | 482 | 21 | 61 | 112s | $0.0187 |
| **Total** | 77 | 1,430 | 32 | 72 | ~3.5 min | **$0.025** |

Two teaching moments:

1. **Wells Fargo: 0 tables detected.** PyMuPDF's `find_tables()` cannot recover tables that are laid out as positioned text without an underlying table structure — common for press-release-style PDFs. The financial data *is* there, but fragmented across 418 text blocks. This is exactly the failure mode `02_chunking.ipynb` opens with: a generic `CoverageDiagnostic` tool reveals fixed-size chunking can't retrieve the table content even when asked direct questions. We don't hard-code "WF misses tables" anywhere — the tool measures it.

2. **AMD: 36 cache hits within itself** (out of 104 lookups). AMD's slide deck reuses the same logo / page-header / footer images across pages. Content-addressed caching deduplicates them automatically — proof the cache key design works without writing a test.

---

## What this lab is NOT

- Not a teach-LangChain-from-scratch course (we use LangChain components but reorganize them around our own abstractions).
- Not exhaustive coverage of every chunking strategy (we cover 3; semantic chunking and contextual retrieval are discussed conceptually in `02_chunking`).
- Not a deployment course (`06` shows a FastAPI demo; real cloud deployment is its own project).

---

## Footnotes

**Source PDFs** (download manually):
- Wells Fargo Q4 2025: https://www.wellsfargo.com/assets/pdf/about/investor-relations/earnings/fourth-quarter-2025-earnings.pdf
- Tesla Q1 2026 Update: https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q1-2026-Update.pdf
- AMD Q4 2025 Earnings Slides: https://d1io3yog0oux5.cloudfront.net/_b0eb9fe85e9ee1621001cc760a9e1d73/amd/db/841/9223/presentation/AMD+Q4'25+Earnings+Slides+FINAL.pdf

**License**: MIT — see [LICENSE](LICENSE).
