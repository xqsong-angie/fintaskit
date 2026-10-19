"""
chunk -> page resolution.

Retrieval metrics are scored against `gold_pages` in the golden set, so every
retrieved chunk has to be attributed to the pages it actually covers. A chunk's
own `page_number` is not enough:

  - `fixed_size` chunks straddle page breaks routinely, but record one page
    (the first non-None among the blocks they overlap).
  - `parent_child` parents are ~800 tokens and routinely straddle a break.

The lossless signal is `source_block_ids`, which points back into
`Document.blocks`. Every chunker populates it, so resolving a chunk's true page
set is a dict lookup. `page_number` stays as the fallback for chunks produced
outside the chunkers (hand-built, deserialised from an older cache).

Why this matters for the eval: with a single page per chunk, a chunk covering
pages 21-22 counts only for page 21, so gold page 22 is scored as a miss and
page-level recall is understated. See src/evaluators/component_eval.py.
"""
from __future__ import annotations

from src.core.models import Document, DocumentBlock, DocumentChunk


def build_block_index(doc: Document) -> dict[str, DocumentBlock]:
    """block_id -> DocumentBlock, for resolving chunks back to pages."""
    return {b.block_id: b for b in doc.blocks}


def chunk_pages(
    chunk: DocumentChunk,
    blocks_by_id: dict[str, DocumentBlock],
) -> set[int]:
    """Every page this chunk covers.

    Prefers `source_block_ids` (exact) and falls back to `page_number` (lossy).
    Returns an empty set when neither is available, so callers must treat an
    empty set as "unattributable" rather than "page unknown".
    """
    if chunk.source_block_ids:
        pages = {
            blocks_by_id[bid].page_number
            for bid in chunk.source_block_ids
            if bid in blocks_by_id and blocks_by_id[bid].page_number is not None
        }
        if pages:
            return pages
    if chunk.page_number is not None:
        return {chunk.page_number}
    return set()


def chunk_page_span(
    chunk: DocumentChunk,
    blocks_by_id: dict[str, DocumentBlock],
) -> tuple[int | None, int | None]:
    """(min_page, max_page) for display. (None, None) when unattributable."""
    pages = chunk_pages(chunk, blocks_by_id)
    if not pages:
        return (None, None)
    return (min(pages), max(pages))


def pages_of(
    chunks: list[DocumentChunk],
    blocks_by_id: dict[str, DocumentBlock],
) -> set[int]:
    """Union of every page covered by a retrieved chunk list."""
    out: set[int] = set()
    for c in chunks:
        out |= chunk_pages(c, blocks_by_id)
    return out


def unattributed(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Chunks with no page information at all — always a bug, worth asserting on."""
    return [
        c for c in chunks
        if not c.source_block_ids and c.page_number is None
    ]