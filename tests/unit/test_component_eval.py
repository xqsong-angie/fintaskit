"""Tests for component-eval scoring logic (no PDF, no network, no API key)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluators.component_eval import (
    ANY_DOC,
    EvalConfig,
    GoldenRow,
    load_golden,
    numeric_hits,
    plan_ofat,
    score_citations,
    score_generation,
    score_row,
)


def _row(**kw) -> GoldenRow:
    base = {
        "qid": "q00", "question": "q?", "category": "fact_finding",
        "expected_doc": "amd", "expected_docs": ["amd"],
        "gold_pages": {"amd": {3}}, "answer_numbers": [], "rubric": [],
        "expected_behavior": None,
    }
    base.update(kw)
    return GoldenRow(**base)


# --------------------------------------------------------------------- parsing
def test_gold_pages_list_is_keyed_by_expected_doc():
    r = GoldenRow.from_dict(
        {"question": "q", "expected_doc": "tesla", "gold_pages": [4, 5]}, 0
    )
    assert r.gold_pages == {"tesla": {4, 5}}


def test_gold_pages_dict_is_kept_per_document():
    r = GoldenRow.from_dict(
        {
            "question": "q", "expected_doc": "any",
            "expected_docs": ["amd", "tesla"],
            "gold_pages": {"amd": [10], "tesla": [4, 5]},
        },
        0,
    )
    assert r.gold_pages == {"amd": {10}, "tesla": {4, 5}}
    assert r.expected_docs == ["amd", "tesla"]


def test_out_of_corpus_row_is_detected_and_has_no_gold_pages():
    r = GoldenRow.from_dict(
        {"question": "q", "expected_doc": "none", "gold_pages": [],
         "expected_behavior": "refuse"},
        0,
    )
    assert r.is_ooc
    assert r.total_gold_pages == 0
    assert r.expected_docs == []


def test_real_golden_set_parses_every_row():
    rows = load_golden(Path(__file__).resolve().parents[2]
                       / "data/golden_set/golden.jsonl")
    assert len(rows) == 30
    assert all(r.question for r in rows)
    # every non-OOC row must have at least one gold page to score against
    assert all(r.total_gold_pages > 0 for r in rows if not r.is_ooc)
    assert all(r.is_ooc or r.gold_pages for r in rows)


# --------------------------------------------------------------------- scoring
def test_page_recall_counts_matched_gold_pages():
    r = _row(gold_pages={"amd": {3, 4}})
    s = score_row(r, {"amd": {3}})
    assert s.page_recall == 0.5
    assert s.any_hit == 1.0
    assert s.full_hit == 0.0
    assert s.missed_pages == ["amd:p4"]


def test_page_numbers_are_scoped_per_document():
    """Page 4 of the AMD deck is not page 4 of the Tesla deck. Flattening the
    retrieved pages into one set would count this as a hit."""
    r = _row(gold_pages={"amd": {4}})
    assert score_row(r, {"tesla": {4}}).page_recall == 0.0
    assert score_row(r, {"amd": {4}}).page_recall == 1.0


def test_any_doc_satisfies_an_any_scoped_gold_page():
    r = _row(expected_doc="any", expected_docs=[],
             gold_pages={ANY_DOC: {2}})
    assert score_row(r, {"tesla": {2}}).page_recall == 1.0
    assert score_row(r, {"tesla": {9}}).page_recall == 0.0


def test_cross_doc_doc_coverage_is_a_fraction():
    r = _row(expected_doc="any", expected_docs=["amd", "tesla", "wells_fargo"],
             gold_pages={ANY_DOC: {1}})
    s = score_row(r, {"amd": {1}, "tesla": {1}})
    assert s.doc_coverage == 2 / 3


def test_precision_penalises_chunks_that_carry_extra_pages():
    r = _row(gold_pages={"amd": {3}})
    tight = score_row(r, {"amd": {3}})
    loose = score_row(r, {"amd": {3, 4, 5, 6}})
    assert tight.page_recall == loose.page_recall == 1.0
    assert tight.page_precision == 1.0
    assert loose.page_precision == 0.25


def test_orphan_chunks_are_carried_into_the_score():
    s = score_row(_row(), {"amd": {3}}, n_orphans=2)
    assert s.unattributed_chunks == 2


# ---------------------------------------------------------------- numeric hits
def test_numeric_hits_tolerates_separators_and_currency():
    hit, total = numeric_hits("Revenue was $34,639 million.", ["34,639", "34.6"])
    assert (hit, total) == (1, 2)
    assert numeric_hits("about $34.6 billion", ["34,639", "34.6"]) == (1, 2)


def test_numeric_hits_all_when_every_number_present():
    assert numeric_hits("34,639 vs 25,785", ["34,639", "25,785"])[0] == 2


# ------------------------------------------------------------ generator scoring
def test_generator_axis_scores_numbers_and_refusals():
    golden = [
        _row(qid="q0", answer_numbers=["34,639"]),
        _row(qid="q1", expected_doc="none", expected_behavior="refuse",
             expected_docs=[], gold_pages={}),
    ]
    answers = {
        "q0": {"answer": "Revenue was $34,639 million.", "refused": False},
        "q1": {"answer": "I could not find that in the provided sources.",
               "refused": True},
    }
    s = score_generation(golden, answers)
    assert s["numeric_hit_all"] == 1.0
    assert s["ooc_refusal_rate"] == 1.0
    assert s["false_refusal_rate"] == 0.0


def test_false_refusal_is_counted_on_in_scope_rows():
    golden = [_row(qid="q0")]
    answers = {"q0": {"answer": "I could not find any relevant information.",
                      "refused": True}}
    assert score_generation(golden, answers)["false_refusal_rate"] == 1.0


# ----------------------------------------------------------------------- plans
def test_plan_ofat_respects_the_requested_axes():
    base = EvalConfig(label="baseline")
    assert [c.label for c in plan_ofat(base, {"retriever"})] == [
        "baseline", "retriever:bm25", "retriever:vector",
    ]
    assert all("retriever:" not in c.label
               for c in plan_ofat(base, {"chunker"}))


def test_plan_ofat_includes_the_loader_arm_both_ways():
    base = EvalConfig(label="baseline", use_vlm=True)
    labels = [c.label for c in plan_ofat(base, {"loader"})]
    assert "loader:use_vlm=False" in labels


def test_plan_ofat_never_returns_duplicates():
    base = EvalConfig(label="baseline", chunker="parent_child",
                      retriever="hybrid", use_vlm=True, use_parent=False)
    for axes in ({"loader"}, {"chunker"}, {"retriever"},
                 {"loader", "chunker", "retriever"}):
        cfgs = plan_ofat(base, axes)
        assert len({c.label for c in cfgs}) == len(cfgs)

# ------------------------------------------------------------------ citations
def _cited_setup():
    """One gold page (amd p3) and three candidate chunks."""
    from src.core.models import DocumentBlock, DocumentChunk

    blocks = {f"b{i}": DocumentBlock(block_id=f"b{i}", block_type="paragraph",
                                     text="t", page_number=p)
              for i, p in enumerate([3, 3, 9])}
    chunks = {
        f"c{i}": DocumentChunk(chunk_id=f"c{i}", document_id="d", text="t",
                               page_number=p, source_block_ids=[f"b{i}"])
        for i, p in enumerate([3, 3, 9])
    }
    return blocks, chunks


def test_citation_page_recall_and_precision():
    golden = [_row(qid="q0", gold_pages={"amd": {3, 4}})]
    blocks, chunks = _cited_setup()
    answers = {"q0": {"citations": ["c0", "c2"], "citation_mode": "parsed"}}
    s = score_citations(golden, answers, chunks, blocks)
    assert s["citation_rows"] == 1
    # c0 -> p3, c2 -> p9: one of two gold pages, one of two cited pages
    assert s["citation_page_recall"] == 0.5
    assert s["citation_page_precision"] == 0.5
    assert s["citation_any_hit"] == 1.0
    assert s["citation_full_hit"] == 0.0


def test_citation_full_hit_when_gold_pages_are_covered():
    golden = [_row(qid="q0", gold_pages={"amd": {3}})]
    blocks, chunks = _cited_setup()
    answers = {"q0": {"citations": ["c0", "c1"], "citation_mode": "structured"}}
    assert score_citations(golden, answers, chunks, blocks)["citation_full_hit"] == 1.0


def test_fallback_all_sources_rows_are_excluded_not_counted_as_hits():
    """RAGGenerator attributes every retrieved chunk when the model cites
    nothing, so including those rows would quietly turn citation recall into a
    copy of retrieval recall."""
    golden = [_row(qid="q0", gold_pages={"amd": {3}})]
    blocks, chunks = _cited_setup()
    answers = {"q0": {"citations": ["c0", "c1", "c2"],
                      "citation_mode": "fallback_all_sources"}}
    s = score_citations(golden, answers, chunks, blocks)
    assert s["citation_rows"] == 0
    assert s["citation_fallback_rows"] == 1
    assert s["citation_page_recall"] is None, "must not be scored as a hit"


def test_unknown_citation_ids_are_ignored():
    golden = [_row(qid="q0", gold_pages={"amd": {3}})]
    blocks, chunks = _cited_setup()
    answers = {"q0": {"citations": ["nope", "c0"], "citation_mode": "parsed"}}
    assert score_citations(golden, answers, chunks, blocks)["citation_full_hit"] == 1.0


def test_ooc_rows_are_not_citation_scored():
    golden = [_row(qid="q0", expected_doc="none", expected_docs=[],
                   gold_pages={}, expected_behavior="refuse")]
    blocks, chunks = _cited_setup()
    answers = {"q0": {"citations": [], "citation_mode": "none"}}
    assert score_citations(golden, answers, chunks, blocks)["citation_rows"] == 0
