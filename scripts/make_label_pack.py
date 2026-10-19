"""
Golden-set labeling helper — golden.jsonl in, pasteable markdown out.

Two jobs, no LLM, zero API cost:

  qa    Auto-audit the existing set so you only hand-label the rows that are
        actually broken. Reports, per row: whether expected_doc resolves to a
        real PDF, whether ground_truth carries a number, whether that number
        is actually findable in the corpus, and (optionally) whether the
        question survives company-name removal.

  pack  Emit, per source document, the top-N candidate pages for each question
        that expects that document. Paste those page excerpts into any chat
        window and ask for the real numbers + gold_pages.

Retrieval over pages uses the project's own BM25Retriever — same code path the
eval will use, so candidate pages are representative rather than idealised.
Each document gets its own page-level index, so a distractor document can never
push the gold document's pages out of the pack.

Output size: `pack` defaults to 2 pages x 900 chars per question, which lands
around 9-18 KB per document section — small enough to paste one section per
message. Raise --max-pages/--chars if a question needs more context.

Usage:
    python scripts/make_label_pack.py qa
    python scripts/make_label_pack.py qa --paraphrase
    python scripts/make_label_pack.py pack                    # all 26 non-OOC rows
    python scripts/make_label_pack.py pack --doc tesla        # one document
    python scripts/make_label_pack.py pack --only fact_finding
    python scripts/make_label_pack.py pack --max-pages 3 --chars 1400
    python scripts/make_label_pack.py pack --mode vlm         # captions included
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.captioners.vlm_captioner import GPT4oCaptioner, NoOpCaptioner
from src.core.cache import CacheBundle
from src.core.models import DocumentChunk
from src.observability import CostTracker
from src.pipelines.ingestion import IngestionPipeline
from src.retrievers.bm25 import BM25Retriever

GOLDEN = Path("data/golden_set/golden.jsonl")
UPLOADS = Path("data/uploads")

# expected_doc -> (surface form, aliases used inside the PDFs)
COMPANIES = {
    "wells_fargo": ("Wells Fargo", ("wells fargo", "wells")),
    "tesla": ("Tesla", ("tesla",)),
    "amd": ("AMD", ("amd", "advanced micro devices")),
}

_STOPWORDS = {
    "what", "when", "which", "whose", "were", "was", "does", "did", "how",
    "why", "who", "that", "this", "with", "from", "for", "and", "their",
    "there", "they", "have", "has", "had", "been", "being", "into", "than",
    "then", "them", "these", "those", "some", "such", "only", "other", "will",
    "would", "could", "should", "about", "across", "during", "reported",
    "report", "reports", "comparing", "compare", "statement", "release",
    "the", "current", "each", "both", "between", "among", "above",
    "same", "most", "after", "before", "much", "many", "over", "under",
    "its", "it", "one", "per", "via", "due", "versus", "vs",
}


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
def load_pages(mode: str = "raw") -> dict[str, dict[int, str]]:
    """doc_key -> {page_number: page text}. Cache-warm; costs nothing."""
    tracker = CostTracker()
    captioner = NoOpCaptioner()
    if mode == "vlm":
        captioner = GPT4oCaptioner(cache=CacheBundle.from_root("cache").vlm,
                                   cost_tracker=tracker)
    pipe = IngestionPipeline(captioner=captioner, cost_tracker=tracker)

    pages: dict[str, dict[int, str]] = {}
    for key in COMPANIES:
        src = UPLOADS / f"{key}.pdf"
        if not src.exists():
            print(f"  ! missing {src}", file=sys.stderr)
            continue
        doc = pipe.ingest(src, verbose=False).document
        by_page: dict[int, list[str]] = defaultdict(list)
        for b in doc.blocks:
            if not b.text:
                continue
            page = b.page_number
            if page is None:
                continue
            body = b.text
            if mode == "vlm" and b.semantic_content:
                body = f"{b.text}\n[caption] {b.semantic_content}"
            by_page[page].append(body)
        pages[key] = {
            p: "\n".join(t).strip() for p, t in sorted(by_page.items())
        }
        print(f"  {key:11s} {doc.n_pages:3d} pages  "
              f"{sum(len(v) for v in pages[key].values()):7,d} chars", file=sys.stderr)
    return pages


def _page_chunks(pages: dict[str, dict[int, str]],
                 keep: str | None) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            document_id=doc_key,
            text=text,
            page_number=page,
            metadata={"doc": doc_key},
        )
        for doc_key, pmap in pages.items()
        for page, text in pmap.items()
        if text.strip() and (keep is None or doc_key == keep)
    ]


def build_indexes(pages: dict[str, dict[int, str]]) -> dict[str, BM25Retriever]:
    """One page-level BM25 index over everything, plus one per document.

    Per-document indexes matter: ranking globally and then filtering to a
    document silently drops the gold document whenever a distractor document
    outranks it, which is exactly the case you want to inspect.
    """
    indexes = {"__all__": BM25Retriever()}
    indexes["__all__"].index(_page_chunks(pages, None))
    for doc_key in pages:
        r = BM25Retriever()
        r.index(_page_chunks(pages, doc_key))
        indexes[doc_key] = r
    return indexes


def rank(index: BM25Retriever, query: str, pages: dict[str, dict[int, str]],
         k: int = 5):
    """-> [(doc_key, page, score, page_text)], best first."""
    return [
        (chunk.metadata["doc"], chunk.page_number, score,
         pages[chunk.metadata["doc"]].get(chunk.page_number, ""))
        for chunk, score in index.search_with_scores(query, k=k)
    ]


# --------------------------------------------------------------------------- #
# row analysis
# --------------------------------------------------------------------------- #
def load_rows() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[str]:
    return [m.group(0) for m in _NUM_RE.finditer(text)]


def numbers_present(numbers: list[str], corpus: str) -> tuple[list[str], list[str]]:
    """Split into (found, missing) against a whitespace/comma-normalised corpus."""
    flat = re.sub(r"[,\s]", "", corpus).lower()
    found, missing = [], []
    for n in numbers:
        probe = re.sub(r"[,\s]", "", n).lower()
        (found if probe and probe in flat else missing).append(n)
    return found, missing


def strip_entities(question: str) -> tuple[str, list[str]]:
    """Remove company names. -> (stripped question, names removed)."""
    out, removed = question, []
    for surface, aliases in COMPANIES.values():
        for alias in aliases:
            poss = re.compile(rf"\b{re.escape(alias)}'s\b", re.IGNORECASE)
            bare = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            if poss.search(out):
                removed.append(surface)
                out = poss.sub("the company's", out)
            elif bare.search(out):
                removed.append(surface)
                out = bare.sub("the company", out)
            if removed and removed[-1] == surface:
                break
    return re.sub(r"\s+", " ", out).strip(), removed


def content_terms(question: str) -> list[str]:
    """Distinctive lowercase word stems of a question, possessives stripped
    ("Apple's" -> "apple") so corpus lookups actually match."""
    out = []
    for raw in re.findall(r"[A-Za-z][A-Za-z\-']*", question.lower()):
        w = raw.rstrip("'s")
        if len(w) > 3 and w not in _STOPWORDS and w not in COMPANIES:
            out.append(w)
    return out


# --------------------------------------------------------------------------- #
# qa
# --------------------------------------------------------------------------- #
def run_qa(rows, pages, indexes, top_k: int,
            with_paraphrase: bool) -> None:
    corpora = {k: "\n".join(p.values()) for k, p in pages.items()}

    print("\n" + "=" * 78)
    print("1. ROW AUDIT — what to hand-label, in priority order")
    print("=" * 78)
    print(f"{'id':4s} {'category':20s} {'doc':11s} {'#':>2s}  status")
    need_label: list[str] = []
    for i, r in enumerate(rows):
        gt = r["ground_truth"]
        doc = r["expected_doc"]
        nums = numbers_in(gt)
        flags = []

        if doc == "none":
            flags.append("OOC (skip Ragas)")
        if not nums:
            flags.append("NO NUMBER")
        if doc in corpora and nums:
            found, missing = numbers_present(nums, corpora[doc])
            if missing:
                flags.append(f"NUM NOT IN CORPUS: {missing}")
            elif found:
                flags.append("nums ok")
        if doc not in corpora and doc not in ("any", "none"):
            flags.append(f"DOC KEY UNRESOLVED: {doc!r}")
        if "gold_pages" not in r:
            flags.append("no gold_pages")

        hard = any(f.startswith(("NO NUMBER", "NUM NOT IN")) for f in flags)
        if hard:
            need_label.append(f"Q{i}")
        print(f"Q{i:<3d} {r['category']:20s} {doc:11s} {len(nums):2d}  "
              f"{'; '.join(flags)}")

    print(f"\n  -> hand-label these {len(need_label)} rows first: "
          f"{', '.join(need_label)}")
    print("  -> then add gold_pages to every non-OOC row.")

    print("\n" + "=" * 78)
    print("2. OUT-OF-CORPUS SANITY — does the answer actually exist in corpus?")
    print("=" * 78)
    print("   Only *distinctive* terms count (page frequency < 25%). Generic")
    print("   finance words like 'revenue' are ignored — they're everywhere.")
    print("   A distinctive term that IS present means decide: still OOC?")
    print()
    n_pages_total = sum(len(p) for p in pages.values())

    def where(term: str) -> list[str]:
        hits = []
        for doc_key, pmap in pages.items():
            for page, text in pmap.items():
                if term in text.lower():
                    hits.append(f"{doc_key}:p{page}")
        return hits[:4]

    for i, r in enumerate(rows):
        if r["expected_doc"] != "none":
            continue
        terms = content_terms(r["question"])
        distinctive = [
            t for t in terms
            if sum(t in text.lower() for pmap in pages.values() for text in pmap.values())
            < 0.25 * n_pages_total
        ]
        present = {t: where(t) for t in distinctive}
        present = {t: w for t, w in present.items() if w}
        top = rank(indexes["__all__"], r["question"], pages, k=1)[0]
        print(f"Q{i:<3d} {r['question']}")
        print(f"     distinctive terms: {distinctive or '(none — cleanly OOC)'}")
        if present:
            print("     REVIEW — these distinctive terms DO appear in corpus:")
            for t, w in present.items():
                print(f"       {t!r} at {', '.join(w)}")
            print("       -> if the answer is derivable, this row is not OOC.")
        else:
            print(f"     OK: no distinctive term found; best page {top[0]}:p{top[1]} "
                  f"(bm25 {top[2]:.2f}) is an unrelated match")
        print()

    if not with_paraphrase:
        return

    print("=" * 78)
    print("3. PARAPHRASE FEASIBILITY — strip the company name, then re-rank")
    print("=" * 78)
    print("   gold_rank = rank of the intended doc after stripping. gold_rank 1")
    print("   -> the paraphrase is still uniquely answerable (safe to add).")
    print("   gold_rank > 1 -> ambiguous: set \"ambiguous\": true and score it")
    print("   with doc_diversity@k instead of answer correctness.")
    print(f"\n{'id':4s} {'rank':>4s} {'div':>4s}  proposal")
    ambiguous, safe = [], []
    for i, r in enumerate(rows):
        if r["expected_doc"] == "none":
            continue
        stripped, removed = strip_entities(r["question"])
        if not removed:
            print(f"Q{i:<3d}     -    -  (no entity to strip) {r['question'][:44]}")
            continue
        if stripped.lower().count("the company") > 1:
            print(f"Q{i:<3d}     -    -  SKIP multi-entity: {r['question'][:44]}")
            continue
        hits = rank(indexes["__all__"], stripped, pages, k=top_k)
        order = list(dict.fromkeys(h[0] for h in hits))
        div = len(order)
        gold = r["expected_doc"]
        gold_rank = order.index(gold) + 1 if gold in order else 0
        mark = "ok" if gold_rank == 1 else f"AMBIG@r{gold_rank}"
        (safe if gold_rank == 1 else ambiguous).append(f"Q{i}")
        print(f"Q{i:<3d} {gold_rank:4d} {div:4d}  [{mark:9s}] {stripped}")
        print(f"          docs@{top_k} = {order}")
    print(f"\n  -> safe to strip ({len(safe)}): {', '.join(safe) or 'none'}")
    print(f"  -> ambiguous ({len(ambiguous)}): {', '.join(ambiguous) or 'none'}")


# --------------------------------------------------------------------------- #
# pack
# --------------------------------------------------------------------------- #
def run_pack(rows, pages, indexes, top_k: int, chars: int,
             only: str | None, doc_filter: str | None,
             out_path: str) -> None:
    wanted = {c.strip() for c in only.split(",")} if only else None

    buckets: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["expected_doc"] == "none":
            continue
        if wanted and r["category"] not in wanted:
            continue
        if doc_filter and r["expected_doc"] != doc_filter:
            continue
        buckets[r["expected_doc"]].append((i, r))
    targets = [r for b in buckets.values() for _, r in b]

    lines = [
        "# Golden-set labeling pack",
        "",
        ("Auto-generated by `scripts/make_label_pack.py pack`. Page text is "
         "verbatim from `data/uploads/*.pdf` (raw parse, no LLM)."),
        "",
        (
            "> For each question below, return JSON with `answer_numbers` "
            "(exact figures from the text, keep thousands separators and "
            "units), `gold_pages` (the page numbers above that support the "
            "answer), and `ground_truth` (one sentence containing the figure "
            "and the period)."
        ),
        "",
    ]
    n_pages_emitted = 0

    def emit(i: int, r: dict, hits) -> None:
        nonlocal n_pages_emitted
        lines.extend([
            f"\n{'=' * 78}",
            f"### Q{i}  [{r['category']}]",
            f"QUESTION: {r['question']}",
            f"CURRENT ground_truth: {r['ground_truth']}",
            f"CURRENT numbers: {numbers_in(r['ground_truth']) or '(none)'}",
            "",
        ])
        if not hits:
            lines.append("(no candidate pages ranked)")
            return
        for hdoc, page, score, text in hits:
            body = text[:chars]
            if len(text) > chars:
                body += f"\n  [...{len(text) - chars:,} chars elided...]"
            lines.extend([
                f"--- {hdoc} page {page} (bm25 {score:.2f}) ---",
                body,
            ])
            n_pages_emitted += 1

    def emit_section(title: str, rows_for_section, *, restrict_to: str | None,
                     global_rank: bool) -> None:
        # restrict_to set -> rank inside that document only (never drops pages)
        lines.extend([f"\n\n{'#' * 78}", f"# {title}",
                      f"# {len(rows_for_section)} questions", "#" * 78, ""])
        for i, r in rows_for_section:
            idx = indexes["__all__"] if global_rank else indexes[restrict_to]
            emit(i, r, rank(idx, r["question"], pages, k=top_k))

    for doc_key in sorted(k for k in buckets if k in COMPANIES):
        emit_section(f"{COMPANIES[doc_key][0]}  ({doc_key}.pdf)",
                     buckets[doc_key], restrict_to=doc_key, global_rank=False)

    if buckets.get("any"):
        emit_section("CROSS-DOCUMENT  (expected_doc=any — answer spans filings)",
                     buckets["any"], restrict_to=None, global_rank=True)

    Path(out_path).write_text("\n".join(lines) + "\n")
    size = Path(out_path).stat().st_size
    print(f"\nwrote {out_path}  "
          f"({len(targets)} questions, {n_pages_emitted} page excerpts, "
          f"{size / 1024:.0f} KB)")
    print("sections: " + ", ".join(
        sorted(k for k in buckets if k in COMPANIES) + (["any"] if buckets.get("any") else [])
    ))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["qa", "pack"])
    ap.add_argument("--mode", choices=["raw", "vlm"], default="raw",
                    help="raw = verbatim PDF text (default); vlm = + captions")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-pages", type=int, default=2,
                    help="candidate pages per question in `pack`")
    ap.add_argument("--chars", type=int, default=900,
                    help="chars of page text per excerpt in `pack`")
    ap.add_argument("--only", default=None,
                    help="comma-separated categories, e.g. fact_finding")
    ap.add_argument("--doc", default=None, choices=sorted(COMPANIES),
                    help="restrict `pack` to one source document")
    ap.add_argument("--paraphrase", action="store_true",
                    help="also run the entity-stripping / doc_diversity check")
    ap.add_argument("-o", "--out", default="tmp_label_pack.md")
    args = ap.parse_args()

    print(f"loading corpus (mode={args.mode})...", file=sys.stderr)
    pages = load_pages(args.mode)
    indexes = build_indexes(pages)
    rows = load_rows()
    print(f"{len(rows)} golden rows, "
          f"{sum(len(p) for p in pages.values())} pages indexed", file=sys.stderr)

    if args.command == "qa":
        run_qa(rows, pages, indexes, args.top_k, args.paraphrase)
    else:
        run_pack(rows, pages, indexes, args.max_pages, args.chars,
                 args.only, args.doc, args.out)


if __name__ == "__main__":
    main()