"""
FastAPI service wrapping RAGStack.

Endpoints:

  POST /ingest     body: {"path": "...", "max_pages": int?, "page_range": [int,int]?}
                   returns: {"document_id", "title", "n_chunks", "cost_usd", "duplicate"}
  POST /query      body: {"question": "...", "document_ids": [str]?}
                   returns: {"answer", "citations", "stages", "query_type"}
  GET  /documents  returns: indexed documents (for a UI document picker)
  GET  /health     returns: {"status", "n_documents_indexed", ...}

Assembly lives in app_factory.RAGStack — this module is only the HTTP shell, so
the notebooks, the Gradio UI and this API all serve the same pipeline config.

Run from repo root:
    uvicorn src.api.server:app --reload --port 8000

Production hardening (auth, rate limiting, async job queue) is out of scope for
the lab.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.app_factory import PipelineConfig, RAGStack
from src.captioners.vlm_captioner import NoOpCaptioner
from src.core.cache import CacheBundle
from src.core.config import configure_langsmith, settings
from src.observability import CostTracker
from src.pipelines import IngestionPipeline


# =============================================================
# Request / Response models
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
# App state
# =============================================================
def build_app(
    config: Optional[PipelineConfig] = None,
    cache: Optional[CacheBundle] = None,
    ingestion: Optional[IngestionPipeline] = None,
) -> FastAPI:
    """Factory pattern — lets tests build an isolated app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_langsmith()
        yield

    app = FastAPI(
        title="Fintech RAG Lab API",
        version="1.1.0",
        description="RAG service over financial filings. POST /ingest, then POST /query.",
        lifespan=lifespan,
    )

    tracker = CostTracker()
    cache = cache or CacheBundle.from_root("cache", enabled=True)
    ingestion = ingestion or IngestionPipeline(cache=cache, cost_tracker=tracker)
    config = config or PipelineConfig(persist_dir="./chroma_api", collection="api_stack")
    stack = RAGStack(config, cache, tracker, ingestion=ingestion)
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


# Module-level app for `uvicorn src.api.server:app`
app = build_app(ingestion=IngestionPipeline(captioner=NoOpCaptioner(), cache=CacheBundle.from_root("cache")))