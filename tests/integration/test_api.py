"""
Integration tests for the FastAPI server.

Uses fastapi.testclient.TestClient — no real server needed. Builds the app via
build_app() with a NoOp captioner and fake retrievers/generator so we don't hit OpenAI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build an isolated app with fake retrievers + generator in tmp_path."""
    monkeypatch.chdir(tmp_path)

    from tests.integration.make_test_pdf import make_test_pdf
    pdf_path = tmp_path / "test.pdf"
    make_test_pdf(pdf_path)

    from src.api.app_factory import PipelineConfig
    from src.api.server import build_app
    from src.captioners.vlm_captioner import NoOpCaptioner
    from src.core.cache import CacheBundle
    from src.observability import CostTracker
    from src.pipelines.ingestion import IngestionPipeline
    from src.pipelines.query import QueryPipeline
    from src.retrievers import HybridRetriever
    from src.core.models import DocumentChunk

    tracker = CostTracker()
    cache = CacheBundle.from_root(tmp_path / "cache", enabled=True)
    ingestion = IngestionPipeline(captioner=NoOpCaptioner(), cache=cache, cost_tracker=tracker)

    app = build_app(PipelineConfig(), cache=cache, ingestion=ingestion)
    stack = app.state.stack

    # Swap the real retrievers/generator for fakes so no OpenAI call happens.
    class _MutableFakeVector:
        def __init__(self):
            self.chunks_to_return = []
        def reset(self):
            self.chunks_to_return = []
        def index(self, chunks):
            self.chunks_to_return = list(chunks)
        def search_with_scores(self, query, k=10):
            return [(c, 1.0) for c in self.chunks_to_return[:k]]
        def retrieve(self, query, k=10):
            return self.chunks_to_return[:k]

    class _MutableFakeBM25(_MutableFakeVector):
        pass

    stack.vector = _MutableFakeVector()
    stack.bm25 = _MutableFakeBM25()
    stack.hybrid = HybridRetriever(stack.vector, stack.bm25)

    from tests.unit.test_generator_query import FakeGenerator

    stack.query_pipeline = QueryPipeline(
        stack.hybrid,
        FakeGenerator({
            "answer": "Mocked answer about [^1].",
            "citations": [],  # filled by RAGStack from the chunk lookup
            "refused": False,
            "n_sources_used": 1,
        }),
    )

    return TestClient(app), pdf_path


def test_health_endpoint(client):
    test_client, _ = client
    r = test_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "n_documents_indexed" in body


def test_ingest_then_query(client):
    test_client, pdf_path = client
    
    # Ingest
    r = test_client.post("/ingest", json={"path": str(pdf_path)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_blocks"] > 0
    assert body["n_chunks"] > 0
    
    # Query
    r2 = test_client.post("/query", json={"question": "What was net income?"})
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert "answer" in body2
    assert body2["n_chunks_retrieved"] >= 0


def test_query_without_ingest_fails(client):
    test_client, _ = client
    r = test_client.post("/query", json={"question": "What?"})
    assert r.status_code == 400


def test_ingest_missing_file_404(client):
    test_client, _ = client
    r = test_client.post("/ingest", json={"path": "/nonexistent.pdf"})
    assert r.status_code == 404
