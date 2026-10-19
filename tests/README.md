# Evaluation

## Headline result

Switching from a naive `RecursiveChunker(400/60)` to a `ParentChildChunker(parent=800, child=150)` with parent-expansion at retrieval time, evaluated on a 30-question financial QA set with Ragas:

| Metric | Baseline | Improved | Δ |
|---|---:|---:|---:|
| faithfulness | 0.765 | **0.881** | +0.116 |
| answer_relevancy | 0.386 | **0.618** | +0.232 |
| context_precision | 0.447 | **0.793** | +0.346 |
| context_recall | 0.467 | **0.667** | +0.200 |
| likely-hallucination cases (faith<0.5) | 3 / 30 | **0 / 30** | — |

By category (faithfulness):

| Category | n | Baseline | Improved | Δ |
|---|---:|---:|---:|---:|
| fact_finding | 10 | 0.735 | **0.960** | +0.225 |
| out_of_corpus | 4 | 0.484 | **0.720** | +0.236 |
| single_doc_multihop | 4 | 0.792 | 0.835 | +0.043 |
| semantic | 8 | 0.908 | 0.923 | +0.015 |
| cross_doc | 4 | 0.810 | 0.810 | 0.000 |

**Worked example — Q0**, "What was Wells Fargo's Q4 2025 net income?" (ground truth $5.4B):
- **Baseline** retrieved the right page but couldn't pin the number — produced a hedged "I found several candidate figures" answer (faith=0.60).
- **Improved** answered "$5,361 million ($5.4 billion)" directly (faith=1.00).

The full reproduction is `notebooks/05_evaluation.ipynb`. Numbers above came from a real run and are saved as `tmp_baseline_ragas.csv` / `tmp_improved_ragas.csv` (gitignored).

The `cross_doc` flat line is the next thing to chase — parent-child helps within a document, not across.

---
