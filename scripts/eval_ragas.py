import argparse
from pathlib import Path
import sys

# import root directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app_factory import PipelineConfig, build_stack
from src.evaluators.component_eval import DOC_FILES, load_golden
from src.evaluators.ragas_evaluator import RagasEvaluator
from src.retrievers.reranked import RerankedHybridRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Run End-to-End Ragas Evaluation"
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable Cross-Encoder Reranker",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_ragas_results.csv",
        help="Path to save evaluation CSV",
    )
    args = parser.parse_args()

    # 1. load Golden Set (filter In-Scope questions)
    golden_path = "data/golden_set/golden.jsonl"
    golden_rows = load_golden(golden_path)
    golden_dicts = [
        {
            "question": g.question,
            "ground_truth": getattr(g, "ground_truth", ""),
            "category": g.category,
        }
        for g in golden_rows
        if not g.is_ooc
    ]

    # 2. construct Stack (k=10 coarse filter)
    cfg = PipelineConfig(
        use_vlm=True,
        deep_k=10,
        quick_k=10,
        persist_dir="tmp_chroma_vlm_k10",
    )
    stack = build_stack(cfg)

    # Ingest 3 PDF Financial Report
    for doc_key, pdf_path in DOC_FILES.items():
        stack.ingest(pdf_path)

    # 3. Enable or disable Cross-Encoder Reranker
    if args.use_reranker:
        print("[Info] Applying BAAI/bge-reranker-base (Top-10 -> Top-4)...")
        reranked_retriever = RerankedHybridRetriever(
            base_retriever=stack.retriever, top_n=4
        )
        stack.retriever = reranked_retriever
        stack.query_pipeline.retriever = reranked_retriever

    # 4. run evaluation
    evaluator = RagasEvaluator()
    df_results = evaluator.evaluate(
        lambda q: {
            "answer": stack.query(q).get("answer", ""),
            "chunks": stack.query(q).get("chunks", []),
            "contexts": [c.text for c in stack.query(q).get("chunks", [])],
        },
        golden_dicts,
    )

    print("\n=== RAGAS EVALUATION RESULTS ===")
    print(
        df_results[
            ["faithfulness", "answer_relevancy", "context_precision"]
        ].mean()
    )

    df_results.to_csv(args.output, index=False)
    print(f"Saved detailed Ragas results to {args.output}")


if __name__ == "__main__":
    main()
