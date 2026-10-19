"""
07_ui_demo.py - Production-Ready Gradio UI with Dynamic Document Mapping
"""
from pathlib import Path
import os
import sys

# 保证根目录导入路径正常
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from src.api.app_factory import PipelineConfig, RAGStack
from src.core.cache import CacheBundle
from src.evaluators.component_eval import DOC_FILES
from src.observability import CostTracker
from src.pipelines import IngestionPipeline
from src.retrievers.reranked import RerankedHybridRetriever


class UIManager:
    def __init__(self):
        self.tracker = CostTracker()
        self.cache = CacheBundle.from_root("cache", enabled=True)
        self.stack = None
        self.doc_map = {}  # 存储 可读名称/key -> 真实 document_id 的映射

    def init_stack(self, use_vlm: bool, deep_k: int, use_reranker: bool) -> tuple[str, list[str]]:
        cfg = PipelineConfig(
            use_vlm=use_vlm,
            deep_k=deep_k,
            quick_k=deep_k,
            persist_dir="tmp_chroma_vlm_k10",
        )
        ingestion = IngestionPipeline(cache=self.cache, cost_tracker=self.tracker)
        self.stack = RAGStack(cfg, self.cache, self.tracker, ingestion=ingestion)

        # 1. 自动 Ingest 并建立名称到真实 document_id 的映射
        self.doc_map = {}
        for doc_key, pdf_path in DOC_FILES.items():
            if Path(pdf_path).exists():
                res = self.stack.ingest(pdf_path)
                # 建立双向映射，例如 'wells_fargo' -> 'doc_c0da709f46a9'
                # 以及 title -> 'doc_c0da709f46a9'
                real_id = res.document_id
                self.doc_map[doc_key] = real_id
                if getattr(res, "title", None):
                    self.doc_map[res.title] = real_id

        if use_reranker:
            reranked = RerankedHybridRetriever(
                base_retriever=self.stack.retriever, top_n=deep_k
            )
            self.stack.retriever = reranked
            self.stack.query_pipeline.retriever = reranked

        available_choices = list(self.doc_map.keys())
        msg = f"Pipeline Loaded ({len(available_choices)} docs indexed): VLM={use_vlm}, Deep K={deep_k}"
        return msg, available_choices


manager = UIManager()


def init_pipeline_with_key(api_key: str, use_vlm: bool, deep_k: int, use_reranker: bool):
    if api_key and api_key.strip():
        os.environ["OPENAI_API_KEY"] = api_key.strip()

    if not os.getenv("OPENAI_API_KEY"):
        return "❌ Error: Please enter a valid OpenAI API Key.", gr.update()

    try:
        status_msg, choices = manager.init_stack(use_vlm=use_vlm, deep_k=deep_k, use_reranker=use_reranker)
        return f"✅ Success! {status_msg}", gr.update(choices=choices, value=choices)
    except Exception as e:
        return f"❌ Initialization Failed: {str(e)}", gr.update()


def ask_rag(
    question: str, 
    history, 
    api_key: str, 
    selected_docs: list[str], 
    use_vlm: bool, 
    deep_k: int, 
    use_reranker: bool
):
    if api_key and api_key.strip():
        os.environ["OPENAI_API_KEY"] = api_key.strip()

    if not os.getenv("OPENAI_API_KEY"):
        return "", history, "❌ Missing API Key", "Please input OpenAI API Key on the left sidebar first."

    if not question.strip():
        return "", history, "Please ask a valid question.", ""

    if manager.stack is None:
        try:
            manager.init_stack(use_vlm, deep_k, use_reranker)
        except Exception as e:
            return "", history, "❌ Stack Init Error", str(e)

    # 🔑 映射前端选中的 Key -> 转换为 ChromaDB 识别的真实 document_id
    real_doc_ids = []
    if selected_docs:
        for doc_item in selected_docs:
            if doc_item in manager.doc_map:
                real_doc_ids.append(manager.doc_map[doc_item])
            else:
                real_doc_ids.append(doc_item)
    
    doc_scope = real_doc_ids if real_doc_ids else None

    try:
        res = manager.stack.query(question, document_ids=doc_scope)
        answer = res.get("answer", "No answer generated.")
        citations = res.get("citations", [])
        chunks = res.get("chunks", [])

        # 引文渲染
        citation_md = "### Document Citations\n"
        if not citations:
            citation_md += "_No citations available for this answer._\n"
        else:
            for i, c in enumerate(citations, 1):
                citation_md += f"**[{i}] Document**: `{c.get('document_id', 'Unknown')}` | **Page**: `{c.get('page_number', 'N/A')}`\n"
                citation_md += f"> _{c.get('text', '')[:200]}..._\n\n"

        # 验证渲染
        claim_md = "### 🔍 Claim-Level Verification\n"
        claim_md += f"- **Refused / Safety**: `{res.get('refused', False)}`\n"
        claim_md += f"- **Query Type**: `{res.get('query_type', 'standard')}`\n"
        claim_md += f"- **Retrieved Chunks**: `{len(chunks)} chunks`\n"
        claim_md += f"- **Cumulative API Cost**: `${manager.stack.tracker.total:.4f}`\n"

        history = history or []
        if len(history) > 0 and isinstance(history[0], dict):
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
        else:
            history.append((question, answer))

        return "", history, citation_md, claim_md

    except Exception as e:
        return "", history, "❌ Query Failed", f"Error details: {str(e)}"


# 构建 Gradio UI 界面
with gr.Blocks(title="FinTasKit - Financial RAG Suite") as demo:
    gr.Markdown("# 📈 FinTasKit: Financial RAG Workbench")
    gr.Markdown("Interactive RAG with multi-modal VLM ingestion, Cross-Encoder reranking, and citation claims verification.")

    with gr.Row():
        # 左侧控制面板与 Key 设置
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Live Pipeline Config")

            txt_api_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-...",
                type="text",
                value=os.getenv("OPENAI_API_KEY", ""),
                info="Paste your sk-... key here."
            )

            # 🔑 动态的 CheckboxGroup
            dropdown_docs = gr.CheckboxGroup(
                choices=["wells_fargo", "tesla", "amd"],
                value=["wells_fargo", "tesla", "amd"],
                label="Target Document Scope",
                info="Select specific filings to prevent cross-document retrieval leakage."
            )

            chk_vlm = gr.Checkbox(label="Enable VLM Captioning", value=True)
            sld_k = gr.Slider(minimum=3, maximum=15, step=1, value=10, label="Retrieval Depth (k)")
            chk_reranker = gr.Checkbox(label="Enable bge-reranker-base", value=True)

            btn_apply = gr.Button("Apply Config & Key", variant="primary")
            txt_status = gr.Textbox(label="Pipeline Status", interactive=False, lines=2)

            btn_apply.click(
                fn=init_pipeline_with_key,
                inputs=[txt_api_key, chk_vlm, sld_k, chk_reranker],
                outputs=[txt_status, dropdown_docs],
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
                    md_claims = gr.Markdown("_Verification metadata will appear here._")

    # 提交逻辑绑定
    btn_submit.click(
        fn=ask_rag,
        inputs=[msg, chatbot, txt_api_key, dropdown_docs, chk_vlm, sld_k, chk_reranker],
        outputs=[msg, chatbot, md_citations, md_claims],
    )
    msg.submit(
        fn=ask_rag,
        inputs=[msg, chatbot, txt_api_key, dropdown_docs, chk_vlm, sld_k, chk_reranker],
        outputs=[msg, chatbot, md_citations, md_claims],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)