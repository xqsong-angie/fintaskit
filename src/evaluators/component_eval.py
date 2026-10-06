"""
Component-level evaluation for the four RAG stages.

  loader    -> use_vlm on/off (what actually reaches the embedding)
  chunker   -> fixed_size | recursive | parent_child (x parent expansion)
  retriever -> bm25 | vector | hybrid
  generator -> numeric_hit_rate / refusal behaviour (needs an API key)

Everything except the generator axis is deterministic and CPU-only, so it can
gate a PR. The generator axis costs money and is non-deterministic (gpt-5 pins
temperature=1), so it is opt-in and reported separately.

Scoring notes
-------------
Page numbers are only meaningful *within* a document: page 4 of the AMD deck is
not page 4 of the Tesla deck. Gold pages are therefore keyed by document and a
retrieved set is always resolved as `{doc_key: {pages}}` via
`src.core.page_map`, never as a flat set of integers. Flattening them inflates
every score.

`gold_pages` in golden.jsonl is either a list (single-document rows) or a dict
(doc -> pages, for cross_doc rows). Both normalise to dict[str, set[int]]; a row
whose expected_doc is "any" uses ANY_DOC as the key, meaning "any one document
may satisfy these pages".

Chunking is compared one-factor-at-a-time from a baseline config rather than as
a full factorial: with 3 chunkers x 2 parent modes x 3 retrievers x 2 caption
modes a full grid is 36 runs, and most cells answer the same question.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.models import Document, DocumentChunk
from src.core.page_map import build_block_index, chunk_pages

ANY_DOC = "*"

DOC_FILES = {
    "wells_fargo": "data/uploads/wells_fargo.pdf",
    "amd": "data/uploads/amd.pdf",
    "tesla": "data/uploads/tesla.pdf",
}


# --------------------------------------------------------------------------- #
# golden set
# --------------------------------------------------------------------------- #
@dataclass
class GoldenRow:
    qid: str
    question: str
    category: str
    expected_doc: str
    expected_docs: list[str]
    gold_pages: dict[str, set[int]]
    answer_numbers: list[str]
    rubric: list[str]
    expected_behavior: str | None

    @property
    def is_ooc(self) -> bool:
        return self.expected_behavior == "refuse" or self.expected_doc == "none"

    @property
    def total_gold_pages(self) -> int:
        return sum(len(v) for v in self.gold_pages.values())

    @classmethod
    def from_dict(cls, d: dict[str, Any], index: int) -> GoldenRow:
        raw_pages = d.get("gold_pages") or []
        expected_doc = d.get("expected_doc", "any")
        if isinstance(raw_pages, dict):
            gold = {doc: set(pages) for doc, pages in raw_pages.items()}
        else:
            key = ANY_DOC if expected_doc in ("any", "none") else expected_doc
            gold = {key: set(raw_pages)} if raw_pages else {}
        expected_docs = d.get("expected_docs") or (
            [] if expected_doc in ("any", "none") else [expected_doc]
        )
        return cls(
            qid=d.get("id") or f"q{index:02d}",
            question=d["question"],
            category=d.get("category", "unknown"),
            expected_doc=expected_doc,
            expected_docs=list(expected_docs),
            gold_pages=gold,
            answer_numbers=list(d.get("answer_numbers") or []),
            rubric=list(d.get("rubric") or []),
            expected_behavior=d.get("expected_behavior"),
        )


def load_golden(path: str | Path) -> list[GoldenRow]:
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if line.strip():
            rows.append(GoldenRow.from_dict(json.loads(line), i))
    return rows


# --------------------------------------------------------------------------- #
# per-question scoring
# --------------------------------------------------------------------------- #
@dataclass
class RowScore:
    qid: str
    category: str
    page_recall: float
    page_precision: float
    any_hit: float
    full_hit: float
    doc_hit: float
    doc_coverage: float
    n_retrieved_pages: int
    n_gold_pages: int
    hit_pages: list[str] = field(default_factory=list)
    missed_pages: list[str] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)
    unattributed_chunks: int = 0


def resolve_pages(
    chunks: Iterable[DocumentChunk],
    blocks_by_id: dict[str, Any],
    doc_of_chunk: dict[str, str],
) -> tuple[dict[str, set[int]], int]:
    """(doc_key -> pages, n_chunks_with_no_page_info)."""
    out: dict[str, set[int]] = {}
    orphans = 0
    for c in chunks:
        pages = chunk_pages(c, blocks_by_id)
        if not pages:
            orphans += 1
            continue
        doc = doc_of_chunk.get(c.document_id, "?")
        out.setdefault(doc, set()).update(pages)
    return out, orphans


def score_row(
    row: GoldenRow,
    retrieved: dict[str, set[int]],
    n_orphans: int = 0,
) -> RowScore:
    hit, total = 0, 0
    hit_lbl: list[str] = []
    miss_lbl: list[str] = []

    for doc, gold in row.gold_pages.items():
        total += len(gold)
        if doc == ANY_DOC:
            best = max(
                (len(gold & pages) for pages in retrieved.values()), default=0
            )
            hit += best
            for p in sorted(gold):
                (hit_lbl if best and any(p in pages for pages in retrieved.values())
                 else miss_lbl).append(f"any:p{p}")
        else:
            got = retrieved.get(doc, set())
            hit += len(gold & got)
            hit_lbl += [f"{doc}:p{p}" for p in sorted(gold & got)]
            miss_lbl += [f"{doc}:p{p}" for p in sorted(gold - got)]

    if row.expected_docs:
        present = [d for d in row.expected_docs if retrieved.get(d)]
        doc_coverage = len(present) / len(row.expected_docs)
        doc_hit = 1.0 if len(row.expected_docs) == 1 and doc_coverage == 1.0 else 0.0
    else:
        doc_coverage, doc_hit = 0.0, 0.0

    n_retrieved = sum(len(p) for p in retrieved.values())
    return RowScore(
        qid=row.qid,
        category=row.category,
        page_recall=(hit / total) if total else 0.0,
        # Recall alone rewards big chunks: fixed_size covers 1.7 pages per chunk
        # against parent_child's 1.1, so it wins page_recall while retrieving far
        # more irrelevant material. Precision is the other half of that trade-off.
        page_precision=(hit / n_retrieved) if n_retrieved else 0.0,
        any_hit=1.0 if hit > 0 else 0.0,
        full_hit=1.0 if total and hit == total else 0.0,
        doc_hit=doc_hit,
        doc_coverage=doc_coverage,
        n_retrieved_pages=n_retrieved,
        n_gold_pages=total,
        hit_pages=hit_lbl,
        missed_pages=miss_lbl,
        retrieved_docs=sorted(retrieved),
        unattributed_chunks=n_orphans,
    )


# --------------------------------------------------------------------------- #
# config + aggregate result
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    label: str
    chunker: str = "parent_child"
    retriever: str = "hybrid"
    use_vlm: bool = True
    use_parent: bool = False
    k: int = 5

    def as_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "chunker": self.chunker,
            "retriever": self.retriever,
            "use_vlm": self.use_vlm,
            "use_parent": self.use_parent,
        }


@dataclass
class ConfigResult:
    config: EvalConfig
    rows: list[RowScore]
    n_chunks: int
    n_parents: int
    indexed_chars: int
    avg_pages_per_chunk: float
    coverage: dict[str, float] = field(default_factory=dict)

    @property
    def scored(self) -> list[RowScore]:
        return [r for r in self.rows if r.n_gold_pages > 0]

    def mean(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.scored]
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict[str, Any]:
        cov = [r.doc_coverage for r in self.rows if r.category == "cross_doc"]
        return {
            "config": self.config.label,
            "chunker": self.config.chunker,
            "retriever": self.config.retriever,
            "use_vlm": self.config.use_vlm,
            "use_parent": self.config.use_parent,
            "k": self.config.k,
            "n_chunks": self.n_chunks,
            "n_parents": self.n_parents,
            "avg_pages_per_chunk": round(self.avg_pages_per_chunk, 2),
            "page_recall": round(self.mean("page_recall"), 4),
            f"page_recall@{self.config.k}": round(self.mean("page_recall"), 4),
            "page_precision": round(self.mean("page_precision"), 4),
            "any_hit": round(self.mean("any_hit"), 4),
            "full_hit": round(self.mean("full_hit"), 4),
            "doc_hit": round(self.mean("doc_hit"), 4),
            "doc_coverage": round(sum(cov) / len(cov), 4) if cov else 0.0,
            "unattributed_chunks": sum(r.unattributed_chunks for r in self.rows),
            **{f"cov_{k}": round(v, 4) for k, v in self.coverage.items()},
        }


def plan_ofat(
    baseline: EvalConfig,
    axes: set[str] | None = None,
) -> list[EvalConfig]:
    """One-factor-at-a-time around the baseline: cheap, and every row answers a
    question. A full grid is 3 chunkers x 2 parent modes x 3 retrievers x 2
    caption modes = 36 runs, and most cells answer the same question.

    `axes` filters which factors get explored (subset of loader/chunker/
    retriever); the baseline is always included.
    """
    axes = axes or {"loader", "chunker", "retriever"}
    out = [baseline]
    if "loader" in axes:
        for vlm in (True, False):
            if vlm != baseline.use_vlm:
                out.append(EvalConfig(**{**baseline.__dict__, "label":
                                        f"loader:use_vlm={vlm}", "use_vlm": vlm}))
    if "chunker" in axes:
        for ch in ("fixed_size", "recursive", "parent_child"):
            if ch != baseline.chunker:
                out.append(EvalConfig(**{**baseline.__dict__, "label":
                                        f"chunker:{ch}", "chunker": ch,
                                        "use_parent": False}))
    if "retriever" in axes:
        for r in ("bm25", "vector", "hybrid"):
            if r != baseline.retriever:
                out.append(EvalConfig(**{**baseline.__dict__, "label":
                                        f"retriever:{r}", "retriever": r}))
    if "chunker" in axes and baseline.chunker == "parent_child":
        for up in (True, False):
            if up != baseline.use_parent:
                out.append(EvalConfig(**{**baseline.__dict__, "label":
                                        f"parent:expand={up}", "use_parent": up}))
    return out


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
class ComponentEval:
    """Owns one ingested corpus and re-scores it under different configs.

    Building the corpus once and calling `.reconfigure()` per config is what
    makes the grid affordable: DocumentCache returns the parsed Document in
    ~10 ms and the embedding cache is content-addressed, so switching chunker or
    retriever costs nothing and only `use_vlm` re-parses.
    """

    def __init__(self, stack, golden: list[GoldenRow], k: int = 5):
        self.stack = stack
        self.golden = golden
        self.k = k
        self.blocks_by_id: dict[str, Any] = {}
        self.doc_of_id: dict[str, str] = {}
        self._refresh()

    def _refresh(self) -> None:
        """Rebuild block_id -> block and document_id -> doc_key from the stack's
        *current* Documents.

        Must be re-run after every re-ingest: flipping `use_vlm` makes the
        DocumentCache produce genuinely new Documents with new document_ids and
        block_ids, so a stale index silently maps every chunk to "?" and all
        page metrics collapse to zero instead of raising.
        """
        self.blocks_by_id = {}
        self.doc_of_id = {}
        for doc_hash, document_id in self.stack._by_hash.items():
            doc: Document = self.stack._docs[doc_hash]
            key = Path(doc.source_path or "").stem or doc_hash[:8]
            self.doc_of_id[document_id] = key
            self.blocks_by_id.update(build_block_index(doc))

    def chunks_by_id(self) -> dict[str, DocumentChunk]:
        """chunk_id -> chunk, for resolving the generator's cited chunk ids.

        Rebuilt per call because reconfigure() swaps the whole chunk set.
        """
        return {c.chunk_id: c for c in self.stack._chunks}

    def score_retrieval(self) -> list[RowScore]:
        self._refresh()
        rows: list[RowScore] = []
        for g in self.golden:
            if g.is_ooc:
                continue  # nothing to retrieve; scored by the generator axis
            hits = self.stack.retriever.retrieve(
                g.question, k=self.k, use_parent=self.stack.cfg.use_parent
            )
            retrieved, orphans = resolve_pages(hits, self.blocks_by_id, self.doc_of_id)
            rows.append(score_row(g, retrieved, orphans))
        return rows

    def coverage_proxies(self) -> dict[str, float]:
        """Numeric-density proxies from evaluators.coverage — the VLM signal."""
        from src.evaluators.coverage import CoverageDiagnostic

        diags = CoverageDiagnostic(self.stack.retriever, k=self.k).diagnose(
            [g.question for g in self.golden if not g.is_ooc]
        )
        if not diags:
            return {}
        n = len(diags)
        return {
            "pct_dense": sum(d.pct_dense for d in diags) / n,
            "numeric_density": sum(d.avg_numeric_density for d in diags) / n,
            "unique_pages": sum(d.n_unique_pages for d in diags) / n,
        }

    def run(self, config: EvalConfig) -> ConfigResult:
        stack = self.stack.reconfigure(**config.as_pipeline_kwargs())
        saved, self.stack = self.stack, stack
        try:
            self._refresh()
            rows = self.score_retrieval()
            coverage = self.coverage_proxies()
            pages = [len(chunk_pages(c, self.blocks_by_id)) for c in stack._chunks]
        finally:
            self.stack = saved
        return ConfigResult(
            config=config,
            rows=rows,
            n_chunks=len(stack._chunks),
            n_parents=len(stack._parents),
            indexed_chars=sum(len(c.text) for c in stack._chunks),
            avg_pages_per_chunk=(sum(pages) / len(pages)) if pages else 0.0,
            coverage=coverage,
        )


# --------------------------------------------------------------------------- #
# generator axis
# --------------------------------------------------------------------------- #
_REFUSAL_MARKERS = (
    "could not find", "not in the corpus", "no relevant information",
    "does not contain", "not available in the provided", "i don't have",
)


def _norm_number(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit() or ch == ".")


def numeric_hits(answer: str, numbers: list[str]) -> tuple[int, int]:
    """(#matched, #expected) with comma/$ noise stripped."""
    flat = _norm_number(answer)
    hit = sum(1 for n in numbers if _norm_number(n) and _norm_number(n) in flat)
    return hit, len(numbers)


def looks_like_refusal(answer: str, generator_refused: bool) -> bool:
    if generator_refused:
        return True
    low = answer.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def score_generation(
    golden: list[GoldenRow],
    answers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """answers: qid -> the generator's result dict."""
    num_any = num_all = num_n = 0
    ooc_n = ooc_refused = 0
    in_scope_refused = 0
    in_scope_n = 0
    for g in golden:
        res = answers.get(g.qid)
        if not res:
            continue
        answer = res.get("answer", "")
        if g.answer_numbers:
            hit, total = numeric_hits(answer, g.answer_numbers)
            num_n += 1
            num_any += 1 if hit else 0
            num_all += 1 if hit == total else 0
        if g.is_ooc:
            ooc_n += 1
            ooc_refused += 1 if looks_like_refusal(answer, res.get("refused", False)) else 0
        else:
            in_scope_n += 1
            in_scope_refused += 1 if looks_like_refusal(answer, res.get("refused", False)) else 0
    return {
        "questions_answered": len(answers),
        "numeric_rows": num_n,
        "numeric_hit_any": round(num_any / num_n, 4) if num_n else None,
        "numeric_hit_all": round(num_all / num_n, 4) if num_n else None,
        "ooc_rows": ooc_n,
        "ooc_refusal_rate": round(ooc_refused / ooc_n, 4) if ooc_n else None,
        "in_scope_rows": in_scope_n,
        "false_refusal_rate": round(in_scope_refused / in_scope_n, 4) if in_scope_n else None,
    }


def score_citations(
    golden: list[GoldenRow],
    answers: dict[str, dict[str, Any]],
    chunks_by_id: dict[str, DocumentChunk],
    blocks_by_id: dict[str, Any],
) -> dict[str, Any]:
    """Do the *cited* chunks point at the gold pages?

    This is the metric that separates "the retriever found it" from "the answer
    says it came from there", and for a financial RAG the second one is the one
    a reviewer actually checks.

    One trap worth naming: when the model emits no parseable citation, RAGGenerator
    falls back to attributing *every* retrieved chunk, which would otherwise make
    citation recall a silent copy of retrieval recall. Those rows are counted
    separately and excluded from the headline numbers.
    """
    scored = [g for g in golden if g.gold_pages and not g.is_ooc]
    n = hit_n = full_n = any_n = 0
    fallback_n = 0
    rec_sum = prec_sum = 0.0
    per_row: list[dict[str, Any]] = []
    for g in scored:
        res = answers.get(g.qid)
        if not res:
            continue
        mode = res.get("citation_mode", "")
        gold: set[int] = set()
        for doc_key, pages in g.gold_pages.items():
            if not g.expected_docs or doc_key in g.expected_docs:
                gold |= set(pages)
        cited: list[DocumentChunk] = [
            chunks_by_id[cid] for cid in res.get("citations", [])
            if cid in chunks_by_id
        ]
        cited_pages: set[int] = set()
        for c in cited:
            cited_pages |= chunk_pages(c, blocks_by_id)
        if mode == "fallback_all_sources":
            fallback_n += 1
            per_row.append({"qid": g.qid, "mode": mode, "scored": False,
                            "cited_pages": sorted(cited_pages)})
            continue
        n += 1
        overlap = gold & cited_pages
        rec = len(overlap) / len(gold) if gold else 0.0
        prec = len(overlap) / len(cited_pages) if cited_pages else 0.0
        rec_sum += rec
        prec_sum += prec
        hit_n += 1 if overlap else 0
        any_n += 1 if any(cited_pages & gold) else 0
        full_n += 1 if gold and gold <= cited_pages else 0
        per_row.append({"qid": g.qid, "mode": mode, "scored": True,
                        "gold_pages": sorted(gold),
                        "cited_pages": sorted(cited_pages),
                        "recall": round(rec, 4), "precision": round(prec, 4)})
    return {
        "citation_rows": n,
        "citation_fallback_rows": fallback_n,
        "citation_page_recall": round(rec_sum / n, 4) if n else None,
        "citation_page_precision": round(prec_sum / n, 4) if n else None,
        "citation_any_hit": round(any_n / n, 4) if n else None,
        "citation_full_hit": round(full_n / n, 4) if n else None,
    }