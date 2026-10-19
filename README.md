---
title: FinTasKit Workbench
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
---
# FinTasKit: Multi-Modal RAG for Financial Reports

A hands-on, production-grade RAG pipeline built around real financial filings (Wells Fargo, Tesla, AMD quarterly reports), specialized for dense, chart-heavy financial reports (10-K, 10-Q, pitch decks). Powered by VLM captioning, hybrid retrieval, Cross-Encoder reranking, and evaluated using Ragas metrics.

This repository is an enhanced version of the financial PDF RAG system found at [zyziyun's fin-rag-lab](https://github.com/zyziyun/fin-rag-lab). It includes comprehensive performance evaluations of the system's core components and offers an optimized solution. Additionally, I have integrated the previously fragmented components into a unified, web-based platform, allowing users to upload one or more documents for analysis and Q&A.

[![CI](https://github.com/zyziyun/fin-rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/zyziyun/fin-rag-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

---
## 🌟 Key Features

- **VLM Ingestion**: Captions embedded tables, graphs, and visual charts directly during document ingestion.
- **Hybrid Retrieval**: Combines sparse BM25 (exact keyword matching for financial numbers) with dense vector search (semantic search via ChromaDB).
- **Two-Stage Reranking**: Uses `bge-reranker-base` to refine Top-10 retrieved chunks down to the Top-4 most relevant contexts.
- **Ragas Benchmarked**: Measured for Faithfulness (85%+), Context Precision (53%+), and Component-Level Page Recall (82%+).
  
---

## 🛠️ Installation & Setup


### 1. Install
```bash
git clone https://github.com/xqsong-angie/fintaskit
cd fintaskit

# Install dependencies
pip install -r requirements.txt
```

### 2. Keys

Create a .env file in the root directory:

```bash
cp .env.example .env
```
And edit .env: OPENAI_API_KEY=sk-...   (LANGSMITH_* optional, only needed for notebook 06)

### 3. PDFs — download into data/uploads/  (URLs in Footnotes below)
 wells_fargo.pdf, tesla.pdf, amd.pdf

### 4. Warm the cache (once, ~3.5 min, ~$0.025)

```bash
python scripts/precompute_cache.py \
    --inputs data/uploads/wells_fargo.pdf \
             data/uploads/tesla.pdf \
             data/uploads/amd.pdf

# 5. Open the first notebook
jupyter lab notebooks/00_quickstart.ipynb
```

If you have a pre-built `cache_bundle.zip`, unzip it at the repo root instead of step 4 — the cache is content-addressed, so a teammate's bundle works on your machine bit-for-bit.

---

## 🚀 Quickstart & Reproduction

Run End-to-End Evaluation (Ragas + Reranker)  
To reproduce our benchmark results on the 26 in-scope golden questions:

```
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

from src.evaluators.component_eval import load_golden, DOC_FILES
from src.evaluators.ragas_evaluator import RagasEvaluator
from src.api.app_factory import PipelineConfig, build_stack
from src.retrievers.reranked import RerankedHybridRetriever

# 1. Load Golden Set
golden_rows = load_golden('data/golden_set/golden.jsonl')
golden_dicts = [{'question': g.question, 'ground_truth': getattr(g, 'ground_truth', ''), 'category': g.category} for g in golden_rows if not g.is_ooc]

# 2. Build Pipeline
cfg = PipelineConfig(use_vlm=True, deep_k=10, quick_k=10, persist_dir='tmp_chroma_vlm_k10')
stack = build_stack(cfg)

for doc_key, pdf_path in DOC_FILES.items():
    stack.ingest(pdf_path)

# 3. Apply Reranker (Top-10 -> Top-4)
stack.retriever = RerankedHybridRetriever(base_retriever=stack.retriever, top_n=4)
stack.query_pipeline.retriever = stack.retriever

# 4. Evaluate
evaluator = RagasEvaluator()
df_results = evaluator.evaluate(lambda q: stack.query(q), golden_dicts)

print('\n=== RAGAS EVALUATION RESULTS ===')
print(df_results[['faithfulness', 'answer_relevancy', 'context_precision']].mean())
"
```
---

## Lab progression

Note: This section is inherited from the upstream repository and provided for educational purposes only. You can safely skip it if you only want to reproduce the benchmark results.

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

## Component evaluation

Ragas (notebook 05) answers "is the final answer good?". That is too late to
localise a regression, and it costs money per run. This harness answers "which
component is at fault?" by swapping one factor at a time (OFAT) over a labelled
golden set and scoring **retrieval against pages**, so the numbers are
deterministic and need no API key.

```bash
# 0. audit the golden set before trusting any metric
python scripts/make_label_pack.py qa --paraphrase
# fix golden set:
python scripts/make_label_pack.py pack

# 1. build a candidate-page paste pack per document (for hand labelling)
python scripts/make_label_pack.py pack --doc tesla

# 2. retrieval axes only — no LLM, no spend, safe for CI
python scripts/run_component_eval.py --axes loader,chunker,retriever --k 5

# 3. one axis at a time while debugging
python scripts/run_component_eval.py --axes chunker --k 5
python scripts/run_component_eval.py --axes retriever --k 5 --show-misses

# 4. everything, plus markdown for the PR
python scripts/run_component_eval.py --axes all --k 5 -o tmp_eval --markdown --persist-dir tmp_chroma_all

# 5. generator axis — costs money, keeps the answer-level metrics
python scripts/run_component_eval.py --axes generator \
    --generator-configs gpt-5-mini -o tmp_eval
```

Pass a **fresh `--persist-dir` per run**. The Chroma collection is not cleared
between configs, so reusing a directory that a previous `use_vlm=True/False` flip
populated leaves stale vectors behind and moves `page_recall` by ~0.02:

```bash
python scripts/run_component_eval.py --axes all --persist-dir tmp_chroma_run1
```

### What each axis isolates

| Axis | Swapped | Question it answers |
|---|---|---|
| `loader` | `use_vlm=True/False` | did captioning help or just add tokens? |
| `chunker` | `fixed_size` / `recursive` / `parent_child` | which split keeps the evidence retrievable? |
| `retriever` | `bm25` / `vector` / `hybrid` | does RRF beat either leg alone? |
| parent | `use_parent=False/True` | does small-to-big expansion pay for its lost precision? |
| `generator` | model | does it cite the right page, and refuse out-of-corpus? |

### Metrics, and one detail that matters

`page_recall` and `page_precision` are computed from the **full set of pages a
chunk covers**, resolved through `chunk.source_block_ids`
(`src/core/page_map.py`), not from the chunk's single `page_number`. A
`fixed_size` chunk straddling pages 21–22 counts for both; with one page per
chunk, page 22 is scored as a miss and recall is understated by construction.

Recall alone is also misleading here: large chunks trivially cover more gold
pages, so `parent:expand=True` posts the *highest* recall and the *lowest*
precision of any config (0.699 / 0.180). That trade-off is the finding, not a
bug — which is why both are always printed together.

The generator axis reports `citation_page_recall` / `citation_page_precision`
separately from retrieval, and **excludes** rows where the model emitted no
parseable citation (`citation_mode="fallback_all_sources"`, where
`RAGGenerator` attributes every retrieved chunk). Including those rows would
silently turn citation recall into a copy of retrieval recall; they are counted
in `citation_fallback_rows` instead.

---

## Run it as an app

```bash
# Gradio UI (chat + citations + live config switching + claim-level verification)
python notebooks/07_ui_demo.py          # → http://127.0.0.1:7860

# or the HTTP service
uvicorn src.api.server:app --port 8000  # → http://127.0.0.1:8000/docs
```

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


## Footnotes

**Source PDFs** (download manually):
- Wells Fargo Q4 2025: https://www.wellsfargo.com/assets/pdf/about/investor-relations/earnings/fourth-quarter-2025-earnings.pdf
- Tesla Q1 2026 Update: https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q1-2026-Update.pdf
- AMD Q4 2025 Earnings Slides: https://d1io3yog0oux5.cloudfront.net/_b0eb9fe85e9ee1621001cc760a9e1d73/amd/db/841/9223/presentation/AMD+Q4'25+Earnings+Slides+FINAL.pdf

**License**: MIT — see [LICENSE](LICENSE).
