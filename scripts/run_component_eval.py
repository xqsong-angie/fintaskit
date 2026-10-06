"""
Run the component eval (loader / chunker / retriever, optionally generator).

Examples
--------
    # retrieval axes only -- no API key needed, no LLM calls, deterministic
    python scripts/run_component_eval.py --axes loader,chunker,retriever

    # one config, verbose per-question misses
    python scripts/run_component_eval.py --axes none --k 5 --show-misses

    # generator axis -- COSTS MONEY, 26 non-OOC questions per config
    python scripts/run_component_eval.py --axes generator --generator-configs gpt-5-mini

    # everything, as a table plus a CSV of every per-question row
    python scripts/run_component_eval.py --axes all -o tmp_component_eval

Writes <out>.summary.csv (one row per config) and <out>.rows.csv (one row per
question per config). Add --markdown for a paste-ready results table.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.api.app_factory import PipelineConfig, build_stack
from src.evaluators.component_eval import (
    DOC_FILES,
    ComponentEval,
    EvalConfig,
    load_golden,
    plan_ofat,
    score_citations,
    score_generation,
)
from src.observability import CostTracker

GOLDEN = ROOT / "data/golden_set/golden.jsonl"

CHUNKERS = ("fixed_size", "recursive", "parent_child")
RETRIEVERS = ("bm25", "vector", "hybrid")
GENERATOR_MODELS = ("gpt-5-mini", "gpt-5")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--axes", default="loader,chunker,retriever",
                    help="comma list of loader,chunker,retriever,generator,none,all")
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--k", type=int, default=5, help="retrieval depth")
    ap.add_argument("--baseline-chunker", default="parent_child", choices=CHUNKERS)
    ap.add_argument("--baseline-retriever", default="hybrid", choices=RETRIEVERS)
    ap.add_argument("--baseline-vlm", action="store_true", default=True)
    ap.add_argument("--no-vlm", dest="baseline_vlm", action="store_false")
    ap.add_argument("--baseline-parent", action="store_true",
                    help="baseline uses parent expansion")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--chunk-overlap", type=int, default=60)
    ap.add_argument("--parent-size", type=int, default=800)
    ap.add_argument("--child-size", type=int, default=150)
    ap.add_argument("--generator-configs", default="gpt-5-mini",
                    help="comma-separated models for the generator axis")
    ap.add_argument("--persist-dir", default=None,
                    help="chroma dir (default: a throwaway tmp dir)")
    ap.add_argument("-o", "--out", default="tmp_component_eval")
    ap.add_argument("--markdown", action="store_true",
                    help="also print a markdown table")
    ap.add_argument("--show-misses", action="store_true",
                    help="print per-question missed gold pages")
    return ap.parse_args(argv)


def build_corpus(args, tracker: CostTracker):
    persist = args.persist_dir or str(
        ROOT / "chroma_db_component_eval"
    )
    cfg = PipelineConfig(
        retriever=args.baseline_retriever,
        chunker=args.baseline_chunker,
        use_parent=args.baseline_parent,
        use_vlm=args.baseline_vlm,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parent_size=args.parent_size,
        child_size=args.child_size,
        persist_dir=persist,
        collection="component_eval",
    )
    stack = build_stack(cfg, tracker=tracker)
    for doc_key, path in DOC_FILES.items():
        res = stack.ingest(ROOT / path)
        print(f"  ingested {doc_key:11s} cache_hit={res.cache_hit} "
              f"chunks={res.n_chunks}", file=sys.stderr)
    return stack


def selected_configs(args) -> list[EvalConfig]:
    baseline = EvalConfig(
        label="baseline",
        chunker=args.baseline_chunker,
        retriever=args.baseline_retriever,
        use_vlm=args.baseline_vlm,
        use_parent=args.baseline_parent,
        k=args.k,
    )
    axes = {a.strip() for a in args.axes.split(",") if a.strip()}
    if "all" in axes:
        axes |= {"loader", "chunker", "retriever"}
    axes &= {"loader", "chunker", "retriever"}
    if not axes:
        return [baseline]
    return plan_ofat(baseline, axes)


def run_generator_axis(evalr: ComponentEval, cfg: EvalConfig,
                       models: list[str], args) -> list[dict[str, Any]]:
    """Generate an answer per non-OOC question for each model, then score."""
    from src.generators.rag_generator import RAGGenerator

    out = []
    for model in models:
        gen = RAGGenerator(model=model, cost_tracker=evalr.stack.tracker)
        answers: dict[str, dict[str, Any]] = {}
        for g in evalr.golden:
            hits = evalr.stack.retriever.retrieve(
                g.question, k=args.k, use_parent=cfg.use_parent
            )
            answers[g.qid] = gen.generate(g.question, hits)
        summary = score_generation(evalr.golden, answers)
        summary.update(score_citations(
            evalr.golden, answers,
            evalr.chunks_by_id(), evalr.blocks_by_id,
        ))
        summary.update({"config": f"generator:{model}", "model": model,
                        "chunker": cfg.chunker, "retriever": cfg.retriever})
        out.append(summary)
        print(f"\n  generator={model}  spend=${evalr.stack.tracker.total:.4f}")
        for k, v in summary.items():
            print(f"    {k:24s} {v}")
    return out


def print_table(summaries: list[dict[str, Any]]) -> None:
    cols = ["config", "page_recall", "page_precision", "any_hit", "full_hit",
            "doc_hit", "doc_coverage", "n_chunks", "avg_pages_per_chunk",
            "cov_pct_dense"]
    w = {c: max(len(c), 12) for c in cols}
    print("\n" + "  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for s in summaries:
        print("  ".join(str(s.get(c, "")).ljust(w[c]) for c in cols))


def print_markdown(summaries: list[dict[str, Any]]) -> None:
    cols = ["config", "page_recall", "page_precision", "any_hit", "full_hit",
            "doc_hit", "doc_coverage", "n_chunks", "avg_pages_per_chunk"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for s in summaries:
        print("| " + " | ".join(str(s.get(c, "")) for c in cols) + " |")


def main(argv=None) -> int:
    args = parse_args(argv)
    tracker = CostTracker()
    golden = load_golden(args.golden)
    scorable = [g for g in golden if not g.is_ooc]
    print(f"golden set: {len(golden)} rows "
          f"({len(scorable)} retrieval-scorable, "
          f"{len(golden) - len(scorable)} out-of-corpus)", file=sys.stderr)

    stack = build_corpus(args, tracker)
    evalr = ComponentEval(stack, golden, k=args.k)

    summaries, row_records, gen_summaries = [], [], []
    for cfg in selected_configs(args):
        result = evalr.run(cfg)
        summaries.append(result.summary())
        for r in result.rows:
            row_records.append({
                "config": cfg.label, **{
                    k: v for k, v in asdict(r).items()
                    if k not in ("hit_pages", "missed_pages", "retrieved_docs")
                },
                "hit_pages": "|".join(r.hit_pages),
                "missed_pages": "|".join(r.missed_pages),
                "retrieved_docs": "|".join(r.retrieved_docs),
            })
        print(f"  scored {cfg.label:24s} "
              f"page_recall={result.mean('page_recall'):.3f} "
              f"orphans={sum(r.unattributed_chunks for r in result.rows)}",
              file=sys.stderr)
        if args.show_misses:
            for r in result.rows:
                if r.missed_pages:
                    print(f"      {r.qid} [{r.category}] missed "
                          f"{','.join(r.missed_pages)} "
                          f"got={','.join(r.retrieved_docs)}")

    requested = {a.strip() for a in args.axes.split(",") if a.strip()}
    if "generator" in requested or "all" in requested:
        baseline = EvalConfig(
            label="baseline",
            chunker=args.baseline_chunker,
            retriever=args.baseline_retriever,
            use_parent=args.baseline_parent,
            k=args.k,
        )
        gen_summaries = run_generator_axis(
            evalr, baseline, args.generator_configs.split(","), args
        )

    print_table(summaries)
    if gen_summaries:
        print("\ngenerator axis:")
        for g in gen_summaries:
            print("  " + json.dumps(g, default=str))
    if args.markdown:
        print_markdown(summaries)

    out = Path(args.out)
    with out.with_suffix(".summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    with out.with_suffix(".rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_records[0].keys()))
        w.writeheader()
        w.writerows(row_records)
    if gen_summaries:
        with out.with_suffix(".generator.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gen_summaries[0].keys()))
            w.writeheader()
            w.writerows(gen_summaries)

    print(f"\nwrote {out.with_suffix('.summary.csv')}")
    print(f"wrote {out.with_suffix('.rows.csv')}")
    if gen_summaries:
        print(f"wrote {out.with_suffix('.generator.csv')}")
    print(f"total API spend: ${tracker.total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())