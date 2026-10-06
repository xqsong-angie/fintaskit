"""
S4 §4.5 Strategy D: Parent-Child (small-to-big retrieval).

Core insight:
  "The best chunk size for retrieval is not the best chunk size for generation."

Children are tiny (~150 tokens) → precise embedding match.
Parents are big (~800 tokens) → enough context for the LLM to reason.
Each child has parent_chunk_id pointing back to its parent.
"""
from __future__ import annotations

from typing import NamedTuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.interfaces import BaseChunker
from src.core.models import Document, DocumentChunk

from ._token_utils import get_token_counter


class _Located(NamedTuple):
    """Where a split piece sits in the document, and what it covers."""

    page: int | None
    end: int
    block_ids: list[str]
    start: int = -1


class ParentChildChunker(BaseChunker):
    name = "parent_child"
    
    def __init__(
        self,
        parent_size: int = 800,
        child_size: int = 150,
        parent_overlap: int = 80,
        child_overlap: int = 20,
        model: str = "gpt-4o",
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.model = model
    
    def chunk(self, doc: Document) -> list[DocumentChunk]:
        """Returns children only (what goes in vector DB)."""
        _, children = self.chunk_with_parents(doc)
        return children
    
    def chunk_with_parents(
        self, doc: Document
    ) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
        token_count = get_token_counter(self.model)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size,
            chunk_overlap=self.parent_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=token_count,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size,
            chunk_overlap=self.child_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=token_count,
        )
        
        # Build full doc text (with heading prefixes inline) AND track block char positions
        parts: list[str] = []
        # Each entry: (char_start, char_end, page_number, block_id)
        block_pos: list[tuple[int, int, int | None, str]] = []
        cursor = 0
        for block in doc.blocks:
            if block.block_type in ("header", "footer"):
                continue
            prefix = " > ".join(block.heading_path)
            text = block.get_embed_text()
            if not text:
                continue
            if block.block_type in ("h1", "h2", "h3"):
                segment = f"\n## {text}\n"
            else:
                segment = f"[{prefix}]\n{text}\n\n" if prefix else f"{text}\n\n"
            parts.append(segment)
            block_pos.append(
                (cursor, cursor + len(segment), block.page_number, block.block_id)
            )
            cursor += len(segment)
        full_text = "".join(parts)
        block_pages: list[int | None] = [pg for _, _, pg, _ in block_pos]

        def _span(start: int, end: int) -> tuple[int | None, list[str]]:
            """Blocks overlapping [start, end) -> (page, block_ids).

            `page` keeps the previous behaviour: the page covering most of the
            span. `block_ids` is the lossless signal. A parent can straddle a
            page break, so one page_number cannot attribute a chunk to the pages
            it really covers; src/core/page_map.py resolves the full set.
            """
            touched = [
                (s, e, pg, bid)
                for s, e, pg, bid in block_pos
                if s < end and e > start
            ]
            ids = [bid for _, _, _, bid in touched]
            ranked = [pg for _, _, pg, _ in touched if pg is not None]
            if ranked:
                return max(set(ranked), key=ranked.count), ids
            if touched:
                nearest = min(touched, key=lambda t: abs(t[0] - start))
                for j in range(block_pos.index(nearest), -1, -1):
                    if block_pages[j] is not None:
                        return block_pages[j], ids
            return None, ids

        def _locate(haystack: str, needle: str, search_from: int) -> int:
            """Offset of `needle` in `haystack` at or after `search_from`, or -1.

            Three things make this harder than a plain `str.find`:
              1. the splitters strip whitespace, so we search the stripped form;
              2. consecutive splits overlap, so piece N+1 *starts before* the end
                 of piece N. Searching from `search_from` with the piece's own
                 prefix therefore fails for every other piece. We anchor on the
                 tail instead, which is always past the previous cursor, and back
                 out the true start from where the anchor landed.
              3. a piece can straddle two blocks, so its exact text may not appear
                 verbatim after the splitter normalised whitespace.
            """
            idx = haystack.find(needle, search_from)

            if idx < 0:
                compact = " ".join(needle.split())
                if compact != needle:
                    idx = haystack.find(compact, search_from)

            if idx < 0:
                for tail in (60, 150, 300):
                    if len(needle) <= tail:
                        continue
                    pos = haystack.find(needle[-tail:], search_from)
                    if pos != -1:
                        return pos - (len(needle) - tail)

            return idx

        def _page_for(chunk_text: str, search_from: int = 0) -> _Located:
            needle = chunk_text.strip()
            idx = _locate(full_text, needle, search_from)
            if idx < 0:
                # Keep making progress, otherwise the search_from scan stalls.
                return _Located(
                    None,
                    min(len(full_text), search_from + max(1, len(needle) // 2)),
                    [],
                )
            end = idx + len(needle)
            page, ids = _span(idx, end)
            return _Located(page, end, ids, idx)
        
        parents: list[DocumentChunk] = []
        children: list[DocumentChunk] = []
        parent_search_pos = 0
        
        #Set unique parent_id to satisfy indexing idempotence
        doc_key = getattr(doc, "source_hash", None) or doc.document_id

        for p_idx, parent_text in enumerate(parent_splitter.split_text(full_text)):
            parent_loc = _page_for(parent_text, parent_search_pos)
            parent_search_pos = parent_loc.end

            #Set unique parent_id
            parent_id = f"chk_{self.name}_p_{doc_key[:12]}_{p_idx}"

            parent = DocumentChunk(
                chunk_id=parent_id,
                document_id=doc.document_id,
                text=parent_text,
                source_block_ids=parent_loc.block_ids,
                page_number=parent_loc.page,
                metadata={"chunker": self.name, "level": "parent"},
            )
            parents.append(parent)

            # Children are located inside the parent's own text and then mapped
            # back into document coordinates, so each reports the page its text
            # actually sits on rather than the parent's first page. The two
            # diverge for every child after a page break, which is precisely where
            # page-level retrieval metrics are sensitive.
            p_start = parent_loc.start
            child_off = 0
            for c_idx, child_text in enumerate(child_splitter.split_text(parent_text)):

                #Set unique child_id
                child_id = f"chk_{self.name}_c_{doc_key[:12]}_{p_idx}_{c_idx}"

                needle = child_text.strip()
                off = _locate(parent_text, needle, child_off) if p_start >= 0 else -1
                if off >= 0:
                    g_start = p_start + off
                    g_end = g_start + len(needle)
                    page, ids = _span(g_start, g_end)
                    child_loc = _Located(page, g_end, ids, g_start)
                    # Splits overlap, so the next child can start before this one
                    # ends; re-searching from `off` is what lets _locate's tail
                    # anchor recover it.
                    child_off = off
                else:
                    child_loc = _page_for(child_text, parent_loc.start)

                children.append(DocumentChunk(
                    chunk_id=child_id,
                    document_id=doc.document_id,
                    text=child_text,
                    parent_chunk_id=parent.chunk_id,
                    source_block_ids=child_loc.block_ids,
                    page_number=child_loc.page,
                    metadata={"chunker": self.name, "level": "child"},
                ))
        
        return parents, children
