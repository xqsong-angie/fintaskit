"""
Tests for RAGStack config wiring.

Two knobs used to be decorative and are now load-bearing, so they get tests
that would have caught the old behaviour:

  - `PipelineConfig.use_parent` was validated but never forwarded to the
    retriever, so parent expansion was always on and the small-to-big ablation
    could not be run.
  - `PipelineConfig.use_vlm` did not exist, so the loader axis (caption tables
    and images vs index them raw) could not be swept at all.

No API calls: VectorRetriever and IngestionPipeline are both substituted.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from src.api import app_factory
from src.api.app_factory import PipelineConfig, RAGStack
from src.core.cache import CacheBundle
from src.core.models import Document, DocumentBlock, DocumentChunk
from src.observability import CostTracker
from src.retrievers.hybrid import HybridRetriever


# =============================================================
# Doubles
# =============================================================
class FakeVectorRetriever:
    """Stands in for the Chroma-backed retriever so no embeddings are needed."""

    def __init__(self, **kwargs):
        self.chunks: list[DocumentChunk] = []

    def index(self, chunks):
        self.chunks = list(chunks)

    def reset(self):
        self.chunks = []

    def retrieve(self, query, k=5, use_parent=False):
        return self.chunks[:k]

    def search_with_scores(self, query, k=10):
        return [(c, 1.0) for c in self.chunks[:k]]


class FakeIngestion:
    """Returns a Document whose block text depends on the caption mode.

    Mirrors what the real pipeline does: with a VLM the table block carries
    semantic_content, without one it stays raw markdown. Same source_hash in
    both modes, exactly like the real thing (it is a property of the PDF).
    """

    TABLE_MD = "| Metric | Q1 2026 |\n| --- | --- |\n| Revenue | 19.3 |"

    def __init__(self, use_vlm: bool):
        self.use_vlm = use_vlm
        self.calls = 0

    def ingest(self, source, **kwargs):
        self.calls += 1
        doc = Document(
            title="Fake Quarterly Update",
            source_type="pdf",
            source_path=str(source),
            source_hash="0" * 64,
            n_pages=2,
            blocks=[
                DocumentBlock(
                    block_type="paragraph",
                    text="Tesla reported record deliveries. " * 20,
                    page_number=1,
                    heading_path=["Financial Summary"],
                ),
                DocumentBlock(
                    block_type="table",
                    text=self.TABLE_MD,
                    semantic_content=(
                        "Table: Tesla Q1 2026 revenue of 19.3 billion dollars."
                        if self.use_vlm else None
                    ),
                    page_number=2,
                    heading_path=["Financial Summary"],
                ),
            ],
        )
        return SimpleNamespace(
            document=doc, total_cost_usd=0.0, parse_cache_hit=self.calls > 1,
        )


@pytest.fixture
def vector_stub(monkeypatch):
    monkeypatch.setattr(app_factory, "VectorRetriever", FakeVectorRetriever)


@pytest.fixture
def cache(tmp_path):
    return CacheBundle.from_root(tmp_path / "cache", enabled=False)


def _stack(cache, **overrides) -> RAGStack:
    cfg = PipelineConfig(**{"chunker": "recursive", **overrides})
    stack = RAGStack(cfg, cache, CostTracker(), ingestion=FakeIngestion(cfg.use_vlm))
    # Pre-seed the opposite caption mode so reconfigure(use_vlm=...) selects a
    # double instead of building a real IngestionPipeline (which would try to
    # read a real PDF off disk).
    stack._ingest_pool[not cfg.use_vlm] = FakeIngestion(not cfg.use_vlm)
    return stack


def _corpus_text(stack: RAGStack) -> str:
    return "\n".join(c.text for c in stack._chunks)


# =============================================================
# PipelineConfig
# =============================================================
def test_use_parent_requires_parent_child_chunker():
    with pytest.raises(ValueError):
        PipelineConfig(chunker="recursive", use_parent=True).validate()


def test_use_vlm_changes_the_index_fingerprint():
    """Captions change chunk text, so flipping the loader must rebuild the index."""
    on = PipelineConfig(chunker="parent_child", use_vlm=True).fingerprint()
    off = PipelineConfig(chunker="parent_child", use_vlm=False).fingerprint()
    assert on != off


def test_retriever_does_not_change_the_index_fingerprint():
    """Retriever is a query-side knob — changing it must not re-chunk anything."""
    a = PipelineConfig(retriever="vector").fingerprint()
    b = PipelineConfig(retriever="bm25").fingerprint()
    assert a == b


# =============================================================
# RAGStack: use_parent reaches the query pipeline
# =============================================================
def test_use_parent_is_forwarded_to_the_query_pipeline(vector_stub, cache):
    off = _stack(cache, chunker="parent_child", use_parent=False)
    on = _stack(cache, chunker="parent_child", use_parent=True)
    assert off.query_pipeline.use_parent is False
    assert on.query_pipeline.use_parent is True


# =============================================================
# RAGStack: use_vlm re-runs ingestion
# =============================================================
def test_flipping_use_vlm_reindexes_from_the_other_caption_mode(vector_stub, cache):
    stack = _stack(cache, use_vlm=True)
    stack.ingest("fake.pdf")
    assert "19.3 billion" in _corpus_text(stack), "VLM arm should embed the caption"

    raw = stack.reconfigure(use_vlm=False)
    assert "19.3 billion" not in _corpus_text(raw), "raw arm must not embed the caption"
    assert FakeIngestion.TABLE_MD in _corpus_text(raw), "raw arm keeps the table"
    assert raw.status()["fingerprint"] != stack.status()["fingerprint"]


def test_flipping_use_vlm_back_reuses_the_pooled_pipeline(vector_stub, cache):
    """Both caption modes share one cache/pool, so flipping back is free."""
    stack = _stack(cache, use_vlm=True)
    stack.ingest("fake.pdf")
    raw = stack.reconfigure(use_vlm=False)
    back = raw.reconfigure(use_vlm=True)

    assert back.ingestion is stack.ingestion, "should reuse the pooled VLM pipeline"
    assert "19.3 billion" in _corpus_text(back)


def test_retriever_change_does_not_reingest(vector_stub, cache):
    """The whole point of fingerprint(): a retriever swap costs zero API calls."""
    stack = _stack(cache, use_vlm=True)
    stack.ingest("fake.pdf")
    assert stack.ingestion.calls == 1

    stack.reconfigure(retriever="bm25")
    stack.reconfigure(retriever="vector")
    assert stack.ingestion.calls == 1, "retriever is query-side; no re-ingest"


def test_splitter_change_replays_ingest_but_keeps_the_source_log(vector_stub, cache):
    stack = _stack(cache, use_vlm=True)
    stack.ingest("fake.pdf")
    wider = stack.reconfigure(chunk_size=800)

    assert stack.ingestion.calls == 2, "new chunk size must re-run the splitter"
    assert len(wider._sources) == 1, "the replay log must not grow on re-chunk"
    assert len(wider._chunks) <= len(stack._chunks)


def test_chunk_ids_are_stable_across_rechunk(vector_stub, cache):
    """Same PDF + same config => same chunk ids, so re-indexing is idempotent."""
    stack = _stack(cache, use_vlm=True)
    stack.ingest("fake.pdf")
    again = stack.reconfigure(chunk_size=800).reconfigure(chunk_size=400)
    assert [c.chunk_id for c in again._chunks] == [c.chunk_id for c in stack._chunks]


# =============================================================
# HybridRetriever: the small-to-big ablation primitive
# =============================================================
class _Stub:
    def __init__(self, chunks):
        self.chunks = chunks

    def search_with_scores(self, query, k=10):
        return [(c, 1.0 - i * 0.1) for i, c in enumerate(self.chunks[:k])]


def _pc_parts():
    parent = DocumentChunk(
        chunk_id="p0", document_id="d", text="PARENT " * 100,
        page_number=1, metadata={"level": "parent"},
    )
    children = [
        DocumentChunk(
            chunk_id=f"c{i}", document_id="d", text=f"CHILD{i} " * 20,
            parent_chunk_id="p0", page_number=1, metadata={"level": "child"},
        )
        for i in range(4)
    ]
    return parent, children


def _pc_fixture():
    parent, children = _pc_parts()
    return HybridRetriever(
        _Stub(children), _Stub([]), parent_store={"p0": parent},
    )


def test_hybrid_without_parent_expansion_returns_children():
    hyb = _pc_fixture()
    out = hyb.retrieve("revenue", k=3, use_parent=False)
    assert out and all(c.metadata["level"] == "child" for c in out)
    assert all(c.chunk_id != "p0" for c in out)


def test_hybrid_with_parent_expansion_returns_parents():
    hyb = _pc_fixture()
    out = hyb.retrieve("revenue", k=3, use_parent=True)
    assert [c.chunk_id for c in out] == ["p0"], "children collapse onto one parent"


def test_hybrid_parent_expansion_is_off_by_default():
    """Ablation runs must opt in explicitly, or the 'off' arm is a lie."""
    hyb = _pc_fixture()
    assert all(c.metadata["level"] == "child" for c in hyb.retrieve("revenue", k=3))


def test_hybrid_accepts_a_callable_parent_store():
    """RAGStack must be able to hand over a store that is filled *after* the
    retriever is built, which is the only ordering that works for reconfigure()."""
    store: dict = {}
    parent, children = _pc_parts()
    hyb = HybridRetriever(
        _Stub(children), _Stub([]), parent_store=lambda: store
    )
    assert all(c.metadata["level"] == "child"
               for c in hyb.retrieve("revenue", k=3, use_parent=True)), \
        "empty store must degrade to children, not crash"
    store["p0"] = parent
    assert [c.chunk_id for c in hyb.retrieve("revenue", k=3, use_parent=True)] == ["p0"]


def test_reconfigure_keeps_parent_expansion_working(vector_stub, cache):
    """Regression: reconfigure() builds the backends *before* the chunking that
    fills _parents, so capturing the dict by value left the HybridRetriever
    pointing at an orphan dict and expansion silently no-opped — the ablation
    reported identical numbers for expand=True and expand=False."""
    stack = _stack(cache, chunker="parent_child")
    stack.ingest("fake.pdf")
    for use_parent in (False, True):
        s = stack.reconfigure(use_parent=use_parent)
        hits = s.hybrid.retrieve("revenue", k=3, use_parent=use_parent)
        assert hits, "retriever returned nothing"
        levels = {c.metadata.get("level") for c in hits}
        assert levels == ({"parent"} if use_parent else {"child"}), \
            f"use_parent={use_parent} returned {levels}"
