"""Tests for chunk -> page resolution (no PDF, no network, no API key)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import DocumentBlock, DocumentChunk
from src.core.page_map import (
    build_block_index,
    chunk_page_span,
    chunk_pages,
    pages_of,
    unattributed,
)


def _block(block_id: str, page, text: str = "x") -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id, block_type="paragraph", text=text, page_number=page
    )


def _chunk(**kw) -> DocumentChunk:
    base = {"chunk_id": "c1", "document_id": "d1", "text": "t"}
    base.update(kw)
    return DocumentChunk(**base)


def test_source_block_ids_win_over_page_number():
    """A chunk straddling pages 4-5 records only one page_number. Resolving via
    source_block_ids is what stops gold page 5 being scored as a miss."""
    index = {"b1": _block("b1", 4), "b2": _block("b2", 5)}
    c = _chunk(source_block_ids=["b1", "b2"], page_number=4)
    assert chunk_pages(c, index) == {4, 5}


def test_falls_back_to_page_number_without_block_ids():
    assert chunk_pages(_chunk(page_number=7), {}) == {7}


def test_unresolvable_block_ids_fall_back_instead_of_returning_empty():
    """Blocks from a different DocumentCache generation: ids don't resolve, so we
    must still return the lossy page rather than nothing."""
    c = _chunk(source_block_ids=["stale"], page_number=3)
    assert chunk_pages(c, {}) == {3}


def test_no_page_information_returns_empty_set():
    assert chunk_pages(_chunk(), {}) == set()
    assert chunk_page_span(_chunk(), {}) == (None, None)


def test_none_pages_are_dropped_not_counted():
    index = {"b1": _block("b1", None), "b2": _block("b2", 6)}
    c = _chunk(source_block_ids=["b1", "b2"], page_number=6)
    assert chunk_pages(c, index) == {6}


def test_pages_of_unions_across_chunks():
    index = {"a": _block("a", 2), "b": _block("b", 3), "c": _block("c", 3)}
    chunks = [
        _chunk(chunk_id="1", source_block_ids=["a"], page_number=2),
        _chunk(chunk_id="2", source_block_ids=["b", "c"], page_number=3),
    ]
    assert pages_of(chunks, index) == {2, 3}


def test_unattributed_flags_chunks_with_no_page_signal():
    chunks = [_chunk(chunk_id="1", page_number=2), _chunk(chunk_id="2")]
    assert [c.chunk_id for c in unattributed(chunks)] == ["2"]


def test_build_block_index_keys_by_block_id():
    from src.core.models import Document

    doc = Document(
        title="t", source_type="pdf",
        blocks=[_block("x1", 1), _block("x2", 2)],
    )
    index = build_block_index(doc)
    assert set(index) == {"x1", "x2"}
    assert index["x2"].page_number == 2