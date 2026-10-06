"""Tests for what survives the ChromaDB round-trip.

Hermetic: `_lc_to_chunk` is a pure function over a Document, so this needs no
embeddings and no API key. The round-trip itself is lossy by default, and the two
fields it used to drop both feed correctness elsewhere:

  source_block_ids -> exact page attribution for the retriever axis
                      (src/core/page_map.py falls back to the single lossy
                      page_number when they are missing)
  metadata         -> tells a parent chunk from a child chunk, which is how the
                      parent-expansion ablation is verified
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.documents import Document as LCDocument

from src.core.models import DocumentChunk
from src.retrievers.vector import VectorRetriever

_CHUNK = DocumentChunk(
    chunk_id="chk_1",
    document_id="doc_1",
    text="revenue was 34,639",
    parent_chunk_id="chk_parent_0",
    heading_path=["Q1 2026", "Financial Summary"],
    page_number=13,
    source_block_ids=["blk_a", "blk_b"],
    metadata={"chunker": "parent_child", "level": "child"},
)


def _stored_metadata(chunk: DocumentChunk) -> dict:
    """Mirror of the metadata VectorRetriever.index writes, without Chroma.

    Kept as an explicit literal rather than importing internals so that changing
    what index() persists has to change this test too.
    """
    import json

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "parent_chunk_id": chunk.parent_chunk_id or "",
        "heading_path": " > ".join(chunk.heading_path),
        "page_number": chunk.page_number or 0,
        "source_block_ids": json.dumps(chunk.source_block_ids),
        "chunk_meta": json.dumps(chunk.metadata),
    }


def test_round_trip_preserves_source_block_ids():
    back = VectorRetriever._lc_to_chunk(
        LCDocument(page_content=_CHUNK.text, metadata=_stored_metadata(_CHUNK))
    )
    assert back.source_block_ids == ["blk_a", "blk_b"]


def test_round_trip_preserves_chunk_metadata():
    back = VectorRetriever._lc_to_chunk(
        LCDocument(page_content=_CHUNK.text, metadata=_stored_metadata(_CHUNK))
    )
    assert back.metadata["level"] == "child"
    assert back.metadata["chunker"] == "parent_child"


def test_round_trip_preserves_identity_and_hierarchy():
    back = VectorRetriever._lc_to_chunk(
        LCDocument(page_content=_CHUNK.text, metadata=_stored_metadata(_CHUNK))
    )
    assert back.chunk_id == "chk_1"
    assert back.document_id == "doc_1"
    assert back.parent_chunk_id == "chk_parent_0"
    assert back.heading_path == ["Q1 2026", "Financial Summary"]
    assert back.page_number == 13


def test_legacy_rows_without_the_new_keys_degrade_instead_of_raising():
    """Caches and collections written before this change have no source_block_ids
    / chunk_meta keys; reading them must not blow up."""
    legacy = {
        "chunk_id": "chk_old", "document_id": "doc_1",
        "parent_chunk_id": "p0", "heading_path": "", "page_number": 4,
    }
    back = VectorRetriever._lc_to_chunk(
        LCDocument(page_content="text", metadata=legacy)
    )
    assert back.source_block_ids == []
    assert back.metadata == {}
    assert back.page_number == 4


def test_malformed_json_metadata_degrades_to_empty():
    m = _stored_metadata(_CHUNK)
    m["source_block_ids"] = "{not json"
    m["chunk_meta"] = "[1, 2, 3]"
    back = VectorRetriever._lc_to_chunk(
        LCDocument(page_content=_CHUNK.text, metadata=m)
    )
    assert back.source_block_ids == []
    assert back.metadata == {}