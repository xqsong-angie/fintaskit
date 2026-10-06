"""
HybridRetriever: composes Vector + BM25 + RRF + (optional) parent-child swap.

This is what production RAG looks like in 2026:
  1. Run vector and BM25 in parallel (both retrieve top-fetch_k)
  2. Merge with RRF (k=60) into a single ranked list
  3. If parent-child enabled, swap each child for its parent (deduped)
  4. Return top-k
"""
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Optional

from langsmith import traceable

from src.core.interfaces import BaseRetriever
from src.core.models import DocumentChunk
from .rrf import rrf_merge


class HybridRetriever(BaseRetriever):
    name = "hybrid"
    
    def __init__(
        self,
        vector,                         # VectorRetriever
        bm25,                           # BM25Retriever
        parent_store: Mapping[str, DocumentChunk]
        | Callable[[], Mapping[str, DocumentChunk]] | None = None,
        rrf_k: int = 60,
    ):
        """
        `parent_store` maps parent chunk_id -> parent chunk. It may be a callable,
        and callers that own the store (RAGStack) should pass one: they must
        build the retriever before the chunking that fills the store exists, so a
        dict captured by value would still be empty at query time and parent
        expansion would silently no-op.
        """
        self.vector = vector
        self.bm25 = bm25
        self._parent_store = parent_store
        self.rrf_k = rrf_k

    @property
    def parent_store(self) -> Mapping[str, DocumentChunk]:
        store = self._parent_store
        if callable(store):
            store = store()
        return store if store is not None else {}

    @parent_store.setter
    def parent_store(self, value) -> None:
        self._parent_store = value
    
    def add_parents(self, new_parents: list[DocumentChunk]) -> None:
        """add new parents to parent_store"""
        for p in new_parents:
            self.parent_store[p.chunk_id] = p

    def index(self, chunks: list[DocumentChunk],parents: Optional[list[DocumentChunk]] = None) -> None:
        """Increasingly indexing into both vector + BM25 and add parent chunks"""
        self.vector.index(chunks)
        self.bm25.index(chunks)
        if parents:
            self.add_parents(parents)
    
    @traceable(name="hybrid_retrieve")
    def retrieve(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        use_parent: bool = False,
    ) -> list[DocumentChunk]:
        """`use_parent=True` swaps each retrieved child for its parent.

        Defaults to False so that "no parent expansion" is the behaviour you get
        without asking for it. The caller (QueryPipeline) always passes this
        explicitly, from PipelineConfig.use_parent.
        """
        vec_scored = self.vector.search_with_scores(query, k=fetch_k)
        bm25_scored = self.bm25.search_with_scores(query, k=fetch_k)
        
        fused = rrf_merge([vec_scored, bm25_scored], k=self.rrf_k, top_n=fetch_k)
        
        if use_parent and self.parent_store:
            return self._swap_to_parents(fused, k)
        return [c for c, _ in fused[:k]]
    
    def _swap_to_parents(
        self, fused: list[tuple[DocumentChunk, float]], k: int
    ) -> list[DocumentChunk]:
        seen: set[str] = set()
        out: list[DocumentChunk] = []
        for child, _ in fused:
            pid = child.parent_chunk_id
            if pid:
                if pid in seen or pid not in self.parent_store:
                    continue
                seen.add(pid)
                out.append(self.parent_store[pid])
            else:
                out.append(child)
            if len(out) >= k:
                break
        return out
