"""
S4 §4.5 Strategy D: Parent-Child (small-to-big retrieval).

Core insight:
  "The best chunk size for retrieval is not the best chunk size for generation."

Children are tiny (~150 tokens) → precise embedding match.
Parents are big (~800 tokens) → enough context for the LLM to reason.
Each child has parent_chunk_id pointing back to its parent.
"""
from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.interfaces import BaseChunker
from src.core.models import Document, DocumentChunk
from ._token_utils import get_token_counter


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
        # Each entry: (char_start, char_end, page_number)
        block_pos: list[tuple[int, int, int | None]] = []
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
            block_pos.append((cursor, cursor + len(segment), block.page_number))
            cursor += len(segment)
        full_text = "".join(parts)
        block_pages: list[int | None] = [pg for _, _, pg in block_pos]

        def _page_for(chunk_text: str, search_from: int = 0) -> tuple[int | None, int]:
            """Locate chunk in full_text and return (page_number, end_pos) for next search.

            Two things make this harder than a plain `str.find`:
              1. the splitters strip whitespace, so we search the stripped form;
              2. consecutive splits overlap, so chunk N+1 *starts before* the end
                 of chunk N. Searching from `search_from` with the chunk's own
                 prefix therefore fails for every other chunk. We anchor on the
                 chunk's tail instead, which is always past the previous cursor,
                 and back out the true start from where the anchor landed.
            """
            needle = chunk_text.strip()
            idx = full_text.find(needle, search_from)

            if idx < 0:
                compact = " ".join(needle.split())
                idx = full_text.find(compact, search_from)

            if idx < 0:
                for tail in (60, 150, 300):
                    if len(needle) <= tail:
                        continue
                    anchor = needle[-tail:]
                    pos = full_text.find(anchor, search_from)
                    if pos != -1:
                        idx = pos - (len(needle) - tail)
                        break

            if idx < 0:
                # Keep making progress, otherwise the search_from scan stalls.
                return (None, min(len(full_text), search_from + max(1, len(needle) // 2)))

            chunk_end = idx + len(needle)

            # Pages overlapping this span, in document order. An 800-token parent
            # can straddle a page break, so pick the page covering most of it.
            pages: list[int | None] = [
                pg for s, e, pg in block_pos if s < chunk_end and e > idx
            ]
            ranked = [pg for pg in pages if pg is not None]
            if ranked:
                return (max(set(ranked), key=ranked.count), chunk_end)

            # No page recorded for this span — fall back to the nearest known page.
            touched = [i for i, (s, e, _) in enumerate(block_pos) if s < chunk_end and e > idx]
            if touched:
                nearest = min(touched, key=lambda i: abs(block_pos[i][0] - idx))
                for j in range(nearest, -1, -1):
                    if block_pages[j] is not None:
                        return (block_pages[j], chunk_end)
            return (None, chunk_end)
        
        parents: list[DocumentChunk] = []
        children: list[DocumentChunk] = []
        parent_search_pos = 0
        
        #Set unique parent_id to satisfy indexing idempotence
        doc_key = getattr(doc, "source_hash", None) or doc.document_id

        for p_idx, parent_text in enumerate(parent_splitter.split_text(full_text)):
            parent_page, parent_search_pos = _page_for(parent_text, parent_search_pos)

            #Set unique parent_id
            parent_id = f"chk_{self.name}_p_{doc_key[:12]}_{p_idx}"

            parent = DocumentChunk(
                chunk_id=parent_id,
                document_id=doc.document_id,
                text=parent_text,
                page_number=parent_page,
                metadata={"chunker": self.name, "level": "parent"},
            )
            parents.append(parent)

            for c_idx, child_text in enumerate(child_splitter.split_text(parent_text)):

                #Set unique child_id
                child_id = f"chk_{self.name}_c_{doc_key[:12]}_{p_idx}_{c_idx}"

                # Children inherit parent's page (close enough — they're contiguous within parent)
                children.append(DocumentChunk(
                    chunk_id=child_id,
                    document_id=doc.document_id,
                    text=child_text,
                    parent_chunk_id=parent.chunk_id,
                    page_number=parent_page,
                    metadata={"chunker": self.name, "level": "child"},
                ))
        
        return parents, children
