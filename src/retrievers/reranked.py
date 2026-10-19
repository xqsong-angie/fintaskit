from sentence_transformers import CrossEncoder
from src.core.interfaces import BaseRetriever
from src.core.models import DocumentChunk


class RerankedHybridRetriever(BaseRetriever):

    def __init__(
        self,
        base_retriever: BaseRetriever,
        top_n: int = 4,
        model_name: str = "BAAI/bge-reranker-base",
    ):
        self.base_retriever = base_retriever
        self.top_n = top_n
        # 加载 Cross-Encoder 模型
        self.reranker = CrossEncoder(model_name)

    def index(self, chunks: list[DocumentChunk]) -> None:
        """透传 index 给底层的 base_retriever"""
        self.base_retriever.index(chunks)

    def reset(self) -> None:
        """透传 reset 给底层的 base_retriever"""
        if hasattr(self.base_retriever, "reset"):
            self.base_retriever.reset()

    def retrieve(
        self, query: str, k: int = 10, **kwargs
    ) -> list[DocumentChunk]:
        # 1. 粗筛阶段：从 base_retriever 拿 Top-k (例如 k=10)
        initial_chunks = self.base_retriever.retrieve(query, k=k, **kwargs)
        if not initial_chunks:
            return []

        # 2. 构造 (Query, Chunk_Text) 匹配对
        pairs = [[query, chunk.text] for chunk in initial_chunks]

        # 3. Cross-Encoder 计算相关性得分
        scores = self.reranker.predict(pairs)

        # 4. 按 Reranker 得分倒序排列，并截取 Top-N (例如 top_n=4)
        scored_chunks = sorted(
            zip(scores, initial_chunks), key=lambda x: x[0], reverse=True
        )
        reranked_chunks = [
            chunk for score, chunk in scored_chunks[: self.top_n]
        ]

        return reranked_chunks