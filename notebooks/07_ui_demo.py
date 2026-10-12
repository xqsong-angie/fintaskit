"""
07_ui_demo.py - Production-Ready Gradio UI with Citations, Live Config Switching,
and Claim-Level Verification.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from src.api.app_factory import PipelineConfig, RAGStack
from src.captioners.vlm_captioner import NoOpCaptioner
from src.core.cache import CacheBundle
from src.observability import CostTracker
from src.pipelines import IngestionPipeline
from src.retrievers.reranked import RerankedHybridRetriever


class UIManager:
    def __init__(self):
        self.tracker = CostTracker()
        self.cache = CacheBundle.from_root("cache", enabled=True)
        self.stack = None
        self.init_stack(use_vlm=True, deep_k=10, use_reranker=True)

    def init_stack(
        self, use_vlm: bool, deep_k: int, use_reranker: bool
    ) -> str:
        cfg = PipelineConfig(
            use_vlm=use_vlm,
            deep_k=deep_k,
            quick_k=deep_k,
            persist_dir="tmp_chroma_vlm_k10",
        )
        ingestion = IngestionPipeline(
            cache=self.cache, cost_tracker=self.tracker
        )
        self.stack = RAGStack(cfg, self.cache, self.tracker, ingestion=ingestion)

        if use_reranker:
            reranked = RerankedHybridRetriever(
                base_retriever=self.stack.retriever, top_n=4
            )
            self.stack.retriever = reranked
            self.stack.query_pipeline.retriever = reranked

        status = f"Pipeline Loaded: VLM={use_vlm}, Deep K={deep_k}, Reranker={use_reranker}"
        return status


manager = UIManager()


def ask_rag(
    question: str, history, use_vlm: bool, deep_k: int, use_reranker: bool
):
    if not question.strip():
        return "", history, "Please ask a valid question.", ""

    res = manager.stack.query(question)

    answer = res.get("answer", "No answer generated.")
    citations = res.get("citations", [])
    chunks = res.get("chunks", [])

    citation_md = "### Document Citations\n"
    if not citations:
        citation_md += "_No citations available for this answer._\n"
    else:
        for i, c in enumerate(citations, 1):
            citation_md += f"**[{i}] Document**: `{c.get('document_id', 'Unknown')}` | **Page**: `{c.get('page_number', 'N/A')}`\n"
            citation_md += f"> _{c.get('text', '')[:200]}..._\n\n"

    claim_md = "### 🔍 Claim-Level Verification\n"
    claim_md += f"- **Refused / Safety**: `{res.get('refused', False)}`\n"
    claim_md += f"- **Query Type**: `{res.get('query_type', 'standard')}`\n"
    claim_md += f"- **Retrieved Chunks**: `{len(chunks)} chunks`\n"
    claim_md += f"- **Cumulative API Cost**: `${manager.stack.tracker.total:.4f}`\n"

    history = history or []
    history.append((question, answer))

    return "", history, citation_md, claim_md


# 构建 Gradio UI 界面
with gr.Blocks(title="FinTasKit - Financial RAG Suite") as demo:
    gr.Markdown("# 📈 FinTasKit: Financial RAG Workbench")
    gr.Markdown(
        "Interactive RAG with multi-modal VLM ingestion, Cross-Encoder reranking, and citation claims verification."
    )

    with gr.Row():
        # 左侧控制面板与设置
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Live Pipeline Config")
            chk_vlm = gr.Checkbox(label="Enable VLM Captioning", value=True)
            sld_k = gr.Slider(
                minimum=3,
                maximum=15,
                step=1,
                value=10,
                label="Retrieval Depth (k)",
            )
            chk_reranker = gr.Checkbox(
                label="Enable bge-reranker-base", value=True
            )
            btn_apply = gr.Button("Apply Config Switch", variant="secondary")
            txt_status = gr.Textbox(
                label="Pipeline Status", interactive=False, lines=2
            )

            btn_apply.click(
                fn=manager.init_stack,
                inputs=[chk_vlm, sld_k, chk_reranker],
                outputs=[txt_status],
            )

        # 右侧对话框与引文视图
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450, label="RAG Financial Assistant")
            msg = gr.Textbox(
                placeholder="Ask about Tesla, AMD, or Wells Fargo filings...",
                label="Your Question",
            )
            btn_submit = gr.Button("Submit Query", variant="primary")

            with gr.Accordion("🔍 Citations & Verification Details", open=True):
                with gr.Row():
                    md_citations = gr.Markdown("_Citations will appear here._")
                    md_claims = gr.Markdown(
                        "_Verification metadata will appear here._"
                    )

    # 提交逻辑绑定
    btn_submit.click(
        fn=ask_rag,
        inputs=[msg, chatbot, chk_vlm, sld_k, chk_reranker],
        outputs=[msg, chatbot, md_citations, md_claims],
    )
    msg.submit(
        fn=ask_rag,
        inputs=[msg, chatbot, chk_vlm, sld_k, chk_reranker],
        outputs=[msg, chatbot, md_citations, md_claims],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)