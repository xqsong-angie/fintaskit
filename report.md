# FinTasKit: End-to-End Evaluation & Optimization Report

## 1. System Architecture Overview

This production-grade Financial RAG system process multi-modal financial documents (10-K, 10-Q earnings reports with charts and financial tables) through a two-stage retrieval and reranking pipeline.

[insert architecture photo here]


---

## 2. Quantitative Benchmark Results

Evaluations were conducted on a curated golden dataset (`data/golden_set/golden.jsonl`, N=26 in-scope financial questions across 3 earnings reports).

### Component-Level Retrieval Evaluation
- **Baseline Vector Retrieval ($k=3$)**: Page Recall = ~64%
- **VLM Captioning + Hybrid Retrieval ($k=10$)**: Page Recall = **82.05%** (+18.05% absolute gain, successfully resolving dense chart/PPT visual queries).

### Ragas End-to-End Evaluation & Reranker Ablation

| Pipeline Architecture | Faithfulness | Context Precision | Answer Relevancy | Context Recall | Token Cost Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid Baseline ($k=10$)** | 89.02% | 39.39% | 55.86% | N/A (Page Recall 82%) | Base |
| **Hybrid + Cross-Encoder ($k=10 \rightarrow \text{Top-4}$)** | **85.45%** | **53.74%** | 47.55% | N/A (Page Recall ~80%) | **~60% Token Reduction** |

#### Key Insights:
1. **Precision Boost**: Incorporating `bge-reranker-base` raised `context_precision` by **+36.4% relative** (from 39.39% to 53.74%), significantly reducing context noise fed to the LLM.
2. **Cost Efficiency**: Truncating Top-10 hits to Top-4 via Cross-Encoder cut prompt context tokens by ~60% with negligible loss in factual grounding (`faithfulness` remained above 85%).


## Details

mode=raw (raw text stats, without vlm captioning, total 77 pages, 1 page empty)
  wells_fargo  12 pages   30,475 chars

  tesla        31 pages   52,736 chars

  amd          34 pages   37,767 chars

30 golden rows, 76 pages indexed

mode=vlm的情况要补充，因为有图表没有覆盖到，run ```python scripts/make_label_pack.py qa --mode vlm``` 
打包模式输出相关数据 ```python scripts/make_label_pack.py pack``` 生成 tmp_label_pack.md 文件



example:



{"id": "q07", "question": "What was AMD's full-year 2025 revenue?", "category": "fact_finding", "expected_doc": "amd", "ground_truth": "AMD's full-year 2025 revenue was $34,639 million (about $34.6 billion), up 34% from $25,785 million in 2024.", "answer_numbers": ["34,639", "34.6"], "gold_pages": [13, 17], "paraphrase": null, "ambiguous": false}

### Q7  [fact_finding]
QUESTION: What was AMD's full-year 2025 revenue?
CURRENT ground_truth: AMD's full-year 2025 revenue was $34,639 million (about $34.6 billion), up 34% from $25,785 million in 2024.
CURRENT numbers: ['2025', '34,639', '34.6', '34', '25,785', '2024.']

--- amd page 13 (bm25 8.55) ---
13 | | Q4 2025 FINANCIAL RESULTS – February 3, 2026
FY 2025 Revenue
Record full year revenue of $34.6 billion increased 34% y/y
Revenue growth driven by growth in Data Center and Client and Gaming segments
$25.8B
$34.6B
2024 2025
--- amd page 1 (bm25 8.17) ---
AMD Financial Results
Fourth Quarter and Full Year 2025 February 3, 2026

{"id": "q07", "question": "What was AMD's full-year 2025 revenue?", "category": "fact_finding", "expected_doc": "amd", "ground_truth": "AMD's full-year 2025 revenue was a record $34.6 billion, up 34% year-over-year from $25.8 billion in 2024.", "answer_numbers": ["34.6", "34", "25.8"], "gold_pages": [13], "paraphrase": "What was the company's full-year 2025 revenue?", "ambiguous": false}

replace number format with the ones can be found in document,but you can also copy paste the chunk and problematic golden data row to LLMs (e.g., Gemini) to let them revise the data row for you. 

After all problematc data are revised, run python scripts/make_label_pack.py qa again

I ran two component evaluation:
1.python scripts/run_component_eval.py --axes loader,chunker,retriever --k 5
tmp_component_eval.summary.csv RAG 配置综合效果
tmp_component_eval.rows.csv 每一个问题（Q0 ~ Q29）在每一种配置下的细粒度得分

| config | page_recall | page_precision | any_hit | full_hit | doc_hit | doc_coverage | n_chunks | avg_pages_per_chunk | cov_pct_dense |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.7154 | 0.3059 | 0.8846 | 0.5769 | 0.8462 | 0.7917 | 577 | 1.1 | 0.2 |
| chunker:fixed_size | 0.8045 | 0.2192 | 0.9615 | 0.6923 | 0.8462 | 0.6667 | 111 | 1.74 | 0.2154 |
| chunker:recursive | 0.5987 | 0.2583 | 0.8077 | 0.4615 | 0.8462 | 0.7917 | 1317 | 1.0 | 0.2692 |
| retriever:bm25 | 0.6641 | 0.2504 | 0.8077 | 0.5 | 0.8462 | 0.7917 | 577 | 1.1 | 0.2077 |
| retriever:vector | 0.6192 | 0.3378 | 0.8077 | 0.4615 | 0.8462 | 0.7083 | 577 | 1.1 | 0.2077 |
| parent:expand=True | 0.7564 | 0.1822 | 0.8846 | 0.6538 | 0.8462 | 0.7917 | 577 | 1.1 | 0.2 | 

fixed_size 取得了最高的页面召回率 (page_recall = 0.8045)，但代价是 Precision 大幅下降 (0.2192)：

原因：因为固定大小分块（fixed_size）的 Chunk 比较大且固定，平均每个 Chunk 跨越了 1.74 个页面（avg_pages_per_chunk = 1.74），很容易“顺便”把相邻的页面也带进来，因此 full_hit（全命中率）高达 69.23%。但这也塞入了大量无关噪音。

recursive 分块召回率最低 (0.5987)：

原因：它切出了多达 1317 个碎片化小块（n_chunks = 1317），切得太细导致金融财报中的长表或完整陈述被截断，单块语义不完整，检索时很难精准召回全文上下文。

Hybrid 检索（Baseline）优于单腿检索（BM25 或 Vector）：

baseline (Hybrid: BM25 + Vector) 召回率是 0.7154，而单独跑 bm25 是 0.6641，单独跑 vector 是 0.6192。

结论：向量检索与关键词检索结合（RRF 融合）在这个财报数据集上明显优于单维检索。

parent:expand=True（父子块扩展）的权衡：

页面召回率提升到了 0.7564，但精确度降到了最低的 0.1822。这证明了小块检索匹配、大块上下文展开（Parent-Child Expansion）确实能有效找回缺失页面，但需要大模型在生成阶段具备筛选长上下文的能力。

挑出每行recall出问题的，得到
"
qid            category  page_recall  page_precision                               missed_pages
q06        fact_finding     0.000000             0.0                                    amd:p22
q15            semantic     0.333333             0.2                            amd:p24|amd:p27
q21 single_doc_multihop     0.000000             0.0                     amd:p7|amd:p10|amd:p11
q22           cross_doc     0.333333             0.2                           tesla:p4|amd:p10
q23           cross_doc     0.000000             0.0          tesla:p4|tesla:p28|amd:p7|amd:p10
q24           cross_doc     0.166667             0.2 tesla:p4|tesla:p5|tesla:p9|amd:p12|amd:p21
原因分析：
单图表与特殊文本布局（Parsing 阶段丢失）：VLM Captioning（--mode vlm）
多页长上下文与分块截断：尝试 Parent-Child Chunking（父子块扩展），用较小的子块（128/256 tokens）做精准语义匹配，但在检索命中后将整个 Parent Chunk：python scripts/run_component_eval.py --axes parent --k 5 --persist-dir tmp_chroma_parent -o eval_parent --markdown
扩大k值能否直接提升跨文档题：python scripts/run_component_eval.py --axes retriever --k 10 --persist-dir tmp_chroma_k10

2.python scripts/run_component_eval.py --axes all --k 5 -o tmp_eval --markdown
3.loader 轴：验证 VLM 能否解决图表找不到的问题：python scripts/run_component_eval.py --axes loader --persist-dir tmp_chroma_vlm

4.generator 轴（需要API Key）：评估 LLM 输出答案并标注页码引用的准确性：python scripts/run_component_eval.py --axes generator --generator-configs gpt-4o-mini -o tmp_eval --persist-dir tmp_chroma_gen

   Only *distinctive* terms count (page frequency < 25%). Generic

   finance words like 'revenue' are ignored — they're everywhere.

   A distinctive term that IS present means decide: still OOC?


 OUT-OF-CORPUS SANITY — does the answer actually exist in corpus?
Q26  What was Apple's Q4 2025 revenue?

   distinctive terms: ['apple']

   OK: no distinctive term found; best page amd:p33 (bm25 5.73) is an unrelated match



   Q29  What did Microsoft's CEO say about AI during the earnings call?

   distinctive terms: ['microsoft', 'call']

   REVIEW — these distinctive terms DO appear in corpus:

   'microsoft' at amd:p2

   'call' at wells_fargo:p9, tesla:p8, tesla:p30, amd:p22

   -> if the answer is derivable, this row is not OOC.

 
   

