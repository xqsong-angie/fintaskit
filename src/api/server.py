"""
FastAPI service wrapping RAGStack (Updated with VLM & Cross-Encoder Reranker).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.app_factory import PipelineConfig, RAGStack
from src.captioners.vlm_captioner import NoOpCaptioner, GPT4oCaptioner
from src.core.cache import CacheBundle
from src.core.config import configure_langsmith, settings
from src.observability import CostTracker
from src.pipelines import IngestionPipeline
from src.retrievers.reranked import RerankedHybridRetriever

# =============================================================
# Request / Response models (保持原样不变)
# =============================================================
class IngestRequest(BaseModel):
    path: str = Field(..., description="Absolute or repo-relative path to a PDF")
    max_pages: Optional[int] = None
    page_range: Optional[tuple[int, int]] = None


class IngestResponse(BaseModel):
    document_id: str
    title: str
    n_blocks: int
    n_chunks: int
    n_chunks_total: int
    cost_usd: float
    cache_hit: bool
    duplicate: bool


class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[list[str]] = Field(
        None,
        description=(
            "Scope retrieval to these documents. Omit to search the whole "
            "collection — which is how cross-document leakage happens."
        ),
    )


class CitationModel(BaseModel):
    chunk_id: str
    document_id: Optional[str] = None
    text_preview: str
    page_number: Optional[int] = None
    heading_path: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    refused: bool
    query_type: Optional[str] = None
    stages: list[str] = []
    n_chunks_retrieved: int
    cost_usd: float = 0.0


class DocumentModel(BaseModel):
    document_id: str
    title: Optional[str] = None
    n_chunks: int
    source_hash: Optional[str] = None


# =============================================================
# App state & Factory
# =============================================================
def build_app(
    config: Optional[PipelineConfig] = None,
    cache: Optional[CacheBundle] = None,
    ingestion: Optional[IngestionPipeline] = None,
    use_reranker: bool = True,
) -> FastAPI:
    """Factory pattern — lets tests or production build an isolated app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_langsmith()
        yield

    app = FastAPI(
        title="Fintech RAG Lab API",
        version="1.3.0",
        description="RAG service over financial filings with VLM + Reranker.",
        lifespan=lifespan,
    )

    tracker = CostTracker()
    cache = cache or CacheBundle.from_root("cache", enabled=True)
    
    # 1. 默认使用最佳评估配置 (deep_k=10, persist_dir 指向已索引的 tmp_chroma_vlm_k10)
    config = config or PipelineConfig(
        use_vlm=True,
        deep_k=10,
        quick_k=10,
        persist_dir="tmp_chroma_vlm_k10",
        collection="api_stack",
    )
    
    ingestion = ingestion or IngestionPipeline(cache=cache, cost_tracker=tracker)
    stack = RAGStack(config, cache, tracker, ingestion=ingestion)

    # 2. 注入 Reranker 过滤器 (Top-10 -> Top-4)
    if use_reranker:
        print("[API Initialization] Injecting Cross-Encoder Reranker (bge-reranker-base)...")
        reranked_retriever = RerankedHybridRetriever(
            base_retriever=stack.retriever,
            top_n=4,
            model_name="BAAI/bge-reranker-base",
        )
        # 同时替换 stack 和 internal query_pipeline 中的 retriever
        stack.retriever = reranked_retriever
        stack.query_pipeline.retriever = reranked_retriever

    app.state.stack = stack

    # ---- health ----
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "n_documents_indexed": stack.status()["n_documents"],
            "n_chunks": stack.status()["n_chunks"],
            "openai_key_set": settings.has_openai_key,
            "cost_usd": stack.tracker.total,
            "config": stack.cfg.to_dict(),
            "reranker_enabled": use_reranker,
        }

    # ---- documents ----
    @app.get("/documents", response_model=list[DocumentModel])
    def documents():
        return [DocumentModel(**d) for d in stack.documents()]

    # ---- ingest ----
    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest):
        path = Path(req.path)
        if not path.exists():
            raise HTTPException(404, f"file not found: {path}")
        try:
            result = stack.ingest(path, max_pages=req.max_pages, page_range=req.page_range)
        except Exception as e:
            raise HTTPException(400, f"ingest failed: {type(e).__name__}: {e}")
        return IngestResponse(
            document_id=result.document_id,
            title=result.title,
            n_blocks=result.n_blocks,
            n_chunks=result.n_chunks,
            n_chunks_total=result.n_chunks_total,
            cost_usd=result.cost_usd,
            cache_hit=result.cache_hit,
            duplicate=result.duplicate,
        )

    # ---- query ----
    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        if not stack.status()["n_documents"]:
            raise HTTPException(400, "No documents indexed. POST /ingest first.")
        try:
            result = stack.query(req.question, document_ids=req.document_ids)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"query failed: {type(e).__name__}: {e}")

        return QueryResponse(
            answer=result["answer"],
            citations=[
                CitationModel(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    text_preview=c["text"][:150] + ("..." if len(c["text"]) > 150 else ""),
                    page_number=c["page_number"],
                    heading_path=c["heading_path"],
                )
                for c in result["citations"]
            ],
            refused=result["refused"],
            query_type=result.get("query_type"),
            stages=result.get("stages", []),
            n_chunks_retrieved=len(result["chunks"]),
            cost_usd=result["cost_usd"],
        )

    return app


# Module-level default instance for uvicorn
app = build_app(use_reranker=True)