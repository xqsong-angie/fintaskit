"""
Config-driven assembly of the whole RAG stack.

One place that builds (chunker -> retriever -> generator -> QueryPipeline) from
a PipelineConfig. Notebooks 03/05 and the Gradio UI all go through this, so the
config you benchmark is literally the config the UI serves.

Usage:
    stack = RAGStack(PipelineConfig(retriever="hybrid"), cache, tracker)
    stack.ingest("data/uploads/wells_fargo.pdf")
    result = stack.query("What was the CET1 capital ratio?")
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from src.chunkers import FixedSizeChunker, ParentChildChunker, RecursiveChunker
from src.core.cache import CacheBundle
from src.core.models import Document, DocumentChunk
from src.generators import RAGGenerator
from src.observability import CostTracker
from src.pipelines import IngestionPipeline, QueryPipeline
from src.retrievers import BM25Retriever, HybridRetriever, VectorRetriever

RETRIEVERS = ("vector", "bm25", "hybrid")
CHUNKERS = ("recursive", "fixed_size", "parent_child")


@dataclass
class PipelineConfig:
    """Everything that can change between an eval run and a UI session."""

    retriever: str = "hybrid"
    chunker: str = "recursive"
    use_parent: bool = False

    chunk_size: int = 400
    chunk_overlap: int = 60
    parent_size: int = 800
    child_size: int = 150

    quick_k: int = 3
    deep_k: int = 8
    model: str = "gpt-5-mini"
    temperature: float = 0.0

    # Embedded Chroma is single-writer: two processes on the same
    # (persist_dir, collection) will reset each other's index. Give every
    # entry point its own pair — ui_demo and the API both run in development.
    persist_dir: str = "./chroma_db_default"
    collection: str = "stack_default"

    def validate(self) -> "PipelineConfig":
        if self.retriever not in RETRIEVERS:
            raise ValueError(f"retriever must be one of {RETRIEVERS}, got {self.retriever!r}")
        if self.chunker not in CHUNKERS:
            raise ValueError(f"chunker must be one of {CHUNKERS}, got {self.chunker!r}")
        if self.use_parent and self.chunker != "parent_child":
            raise ValueError("use_parent=True requires chunker='parent_child'")
        return self

    def fingerprint(self) -> str:
        """Identity of the *index*. Changing anything here invalidates chunks."""
        return f"{self.chunker}|{self.chunk_size}|{self.chunk_overlap}|{self.parent_size}|{self.child_size}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestResult:
    document_id: str
    title: str
    source_hash: str
    n_blocks: int
    n_chunks: int
    n_chunks_total: int
    cost_usd: float
    cache_hit: bool
    duplicate: bool = False


class RAGStack:
    """Holds the index (chunks + parents) and rebuilds the query side per config.

    The expensive part (parsing, embedding, BM25 stats) lives in the index.
    Switching retriever/generator/model is cheap, so the UI can A/B them without
    re-embedding anything.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        cache: CacheBundle,
        tracker: CostTracker,
        ingestion: Optional[IngestionPipeline] = None,
    ):
        self.cfg = cfg.validate()
        self.cache = cache
        self.tracker = tracker
        self.ingestion = ingestion or IngestionPipeline(cache=cache, cost_tracker=tracker)

        # ---- index state (survives retriever/model changes) ----
        self._docs: dict[str, Document] = {}          # source_hash -> parsed Document
        self._chunks: list[DocumentChunk] = []
        self._parents: dict[str, DocumentChunk] = {}
        self._by_hash: dict[str, str] = {}          # source_hash -> document_id
        self._indexed_fingerprint: Optional[str] = None

        self._build_backends()
        self._build_query_side()

    # ------------------------------------------------------------------ setup
    def _build_backends(self):
        """Vector + BM25 always exist; the retriever picks which one answers."""
        self.vector = VectorRetriever(
            persist_dir=self.cfg.persist_dir,
            collection=self.cfg.collection,
            embeddings_cache_dir=self.cache.embeddings_dir,
        )
        self.bm25 = BM25Retriever()
        self.hybrid = HybridRetriever(self.vector, self.bm25, parent_store=self._parents)

    def _build_query_side(self):
        """Retriever + generator + LangGraph pipeline. Cheap; rebuilt on change."""
        self.retriever = {"vector": self.vector, "bm25": self.bm25, "hybrid": self.hybrid}[
            self.cfg.retriever
        ]
        self.generator = RAGGenerator(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            cost_tracker=self.tracker,
        )
        self.query_pipeline = QueryPipeline(
            self.retriever,
            self.generator,
            quick_k=self.cfg.quick_k,
            deep_k=self.cfg.deep_k,
            cost_tracker=self.tracker,
        )

    # ----------------------------------------------------------------- ingest
    def _chunk_document(self, doc: Document) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
        if self.cfg.chunker == "recursive":
            return RecursiveChunker(
                chunk_size=self.cfg.chunk_size, overlap=self.cfg.chunk_overlap
            ).chunk(doc), []
        if self.cfg.chunker == "fixed_size":
            return FixedSizeChunker(
                size=self.cfg.chunk_size, overlap=self.cfg.chunk_overlap
            ).chunk(doc), []

        parents, children = ParentChildChunker(
            parent_size=self.cfg.parent_size, child_size=self.cfg.child_size
        ).chunk_with_parents(doc)
        return children, parents

    def ingest(self, source: str | Path, **kwargs) -> IngestResult:
        report = self.ingestion.ingest(source, **kwargs)
        doc = report.document
        doc_hash = doc.source_hash or doc.document_id

        if doc_hash in self._by_hash:
            return IngestResult(
                document_id=doc.document_id, title=doc.title, source_hash=doc_hash,
                n_blocks=len(doc.blocks), n_chunks=0, n_chunks_total=len(self._chunks),
                cost_usd=report.total_cost_usd, cache_hit=report.parse_cache_hit,
                duplicate=True,
            )

        chunks, parents = self._chunk_document(doc)
        self._docs[doc_hash] = doc
        self._chunks.extend(chunks)
        self._parents.update({p.chunk_id: p for p in parents})
        self._by_hash[doc_hash] = doc.document_id
        self._reindex()

        return IngestResult(
            document_id=doc.document_id, title=doc.title, source_hash=doc_hash,
            n_blocks=len(doc.blocks), n_chunks=len(chunks),
            n_chunks_total=len(self._chunks), cost_usd=report.total_cost_usd,
            cache_hit=report.parse_cache_hit,
        )

    def _reindex(self):
        """Full rebuild from self._chunks.

        Cheaper than it sounds: embeddings come from cache.embeddings_dir, so
        re-indexing the same chunks costs zero API calls. Doing a full rebuild
        (instead of add_documents) is what keeps vector and BM25 in sync.
        """
        self.vector.reset()
        self.bm25.reset()
        if self._chunks:
            self.vector.index(self._chunks)
            self.bm25.index(self._chunks)
        self._indexed_fingerprint = self.cfg.fingerprint()

    def _rechunk_all(self):
        """Rebuild every chunk from the parsed Documents we kept in memory.

        No re-parsing and no re-captioning: only the splitter runs. Embeddings
        hit cache.embeddings_dir, so a chunk_size sweep costs ~0 API calls.
        """
        self._chunks, self._parents, self._by_hash = [], {}, {}
        for doc_hash, doc in self._docs.items():
            chunks, parents = self._chunk_document(doc)
            self._chunks.extend(chunks)
            self._parents.update({p.chunk_id: p for p in parents})
            self._by_hash[doc_hash] = doc.document_id
        self._reindex()

    def reconfigure(self, **overrides) -> "RAGStack":
        """Return a stack bound to new_cfg, reusing the corpus already parsed."""
        new_cfg = PipelineConfig(**{**self.cfg.to_dict(), **overrides}).validate()

        stack = RAGStack.__new__(RAGStack)
        stack.cfg = new_cfg
        stack.cache = self.cache
        stack.tracker = self.tracker
        stack.ingestion = self.ingestion
        stack._docs = self._docs
        stack._chunks, stack._parents, stack._by_hash = [], {}, {}
        stack._indexed_fingerprint = None
        stack._build_backends()

        if new_cfg.fingerprint() != self._indexed_fingerprint:
            stack._rechunk_all()
        else:
            stack._chunks = self._chunks
            stack._parents = self._parents
            stack._by_hash = self._by_hash
            stack._indexed_fingerprint = self._indexed_fingerprint
            stack._reindex()

        stack._build_query_side()
        return stack

    # ------------------------------------------------------------------ query
    def query(
        self, question: str, document_ids: Optional[list[str]] = None
    ) -> dict[str, Any]:
        if not self._chunks:
            raise RuntimeError("No documents indexed. Call stack.ingest(path) first.")
        unknown = [d for d in (document_ids or []) if d not in self._by_hash.values()]
        if unknown:
            raise ValueError(f"unknown document_id(s): {unknown}")
        result = self.query_pipeline.query(question, document_ids=document_ids)
        lookup = {c.chunk_id: c for c in result["chunks"]}
        result["citations"] = [
            {
                "marker": i + 1,
                "chunk_id": cid,
                "document_id": lookup[cid].document_id,
                "page_number": lookup[cid].page_number,
                "heading_path": lookup[cid].heading_path,
                "text": lookup[cid].text,
            }
            for i, cid in enumerate(result.get("citations", []))
            if cid in lookup
        ]
        result["cost_usd"] = self.tracker.total
        result["config"] = self.cfg.to_dict()
        return result

    def documents(self) -> list[dict[str, Any]]:
        """Indexed documents, for a UI document picker."""
        by_doc: dict[str, dict[str, Any]] = {}
        for doc_hash, doc_id in self._by_hash.items():
            n = sum(1 for c in self._chunks if c.document_id == doc_id)
            by_doc[doc_id] = {
                "document_id": doc_id,
                "source_hash": doc_hash,
                "n_chunks": n,
            }
        for doc in self._docs.values():
            if doc.document_id in by_doc:
                by_doc[doc.document_id]["title"] = doc.title
        return sorted(by_doc.values(), key=lambda d: d["document_id"])

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        return {
            "n_documents": len(self._by_hash),
            "n_chunks": len(self._chunks),
            "n_parents": len(self._parents),
            "fingerprint": self.cfg.fingerprint(),
            "cost_usd": self.tracker.total,
            "config": self.cfg.to_dict(),
        }


def build_stack(
    cfg: PipelineConfig,
    cache: Optional[CacheBundle] = None,
    tracker: Optional[CostTracker] = None,
) -> RAGStack:
    """Convenience factory with repo defaults (matches server.py AppState)."""
    cache = cache or CacheBundle.from_root("cache", enabled=True)
    tracker = tracker or CostTracker()
    return RAGStack(cfg, cache, tracker)