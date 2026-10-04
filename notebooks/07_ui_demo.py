"""
Gradio UI on top of RAGStack (src/api/app_factory.py).

Run from repo root:
    python notebooks/07_ui_demo.py
Then open http://127.0.0.1:7860

The Sidebar mirrors PipelineConfig field-for-field. Changing a chunker param
re-chunks the corpus; changing retriever/model only rebuilds the query side.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import gradio as gr

from src.api.app_factory import CHUNKERS, RETRIEVERS, PipelineConfig, build_stack
from src.core.config import configure_langsmith
from src.evaluators.hallucination import HallucinationDetector
from src.observability import CostTracker

configure_langsmith()

UPLOADS = ROOT / "data" / "uploads"

# Own Chroma location so this never collides with the API server (embedded
# Chroma is single-writer).
STACK = build_stack(
    PipelineConfig(persist_dir="./chroma_ui", collection="ui_stack"),
    tracker=CostTracker(),
)
DETECTOR: HallucinationDetector | None = None
LAST: dict = {"answer": "", "chunks": [], "citations": []}


# ------------------------------------------------------------------ config
def cfg_from_ui(retriever, chunker, use_parent, chunk_size, overlap,
                parent_size, child_size, model) -> PipelineConfig:
    return PipelineConfig(
        retriever=retriever, chunker=chunker, use_parent=use_parent,
        chunk_size=int(chunk_size), chunk_overlap=int(overlap),
        parent_size=int(parent_size), child_size=int(child_size), model=model,
    )


def sync_stack(*sidebar) -> str:
    """Rebuild the query side (and re-chunk if chunker params changed)."""
    global STACK
    try:
        cfg = cfg_from_ui(*sidebar)
        STACK = STACK.reconfigure(**cfg.to_dict())
    except Exception as e:
        return f"config error: {type(e).__name__}: {e}"
    st = STACK.status()
    return (
        f"{st['n_documents']} docs / {st['n_chunks']} chunks | "
        f"`{st['fingerprint']}` | spent ${st['cost_usd']:.4f}"
    )


# ------------------------------------------------------------------ ingest
def do_ingest(paths, *sidebar):
    if not paths:
        return "pick at least one PDF first", sync_stack(*sidebar)

    lines = []
    for p in paths:
        src = Path(p)
        if not src.is_absolute():
            src = UPLOADS / src.name
            if src.suffix.lower() != ".pdf":
                src = src.with_suffix(".pdf")
        try:
            r = STACK.ingest(src)
        except Exception as e:
            lines.append(f"FAIL {src.name}: {type(e).__name__}: {e}")
            continue
        tag = "already indexed" if r.duplicate else f"+{r.n_chunks} chunks"
        lines.append(f"{src.name}: {tag} (cache_hit={r.cache_hit}, ${r.cost_usd:.4f})")
    return "\n".join(lines), sync_stack(*sidebar)


# ------------------------------------------------------------------- query
def do_query(question, history, *sidebar):
    global LAST
    history = list(history or [])
    if not question.strip():
        yield history, "", "", LAST
        return

    sync_stack(*sidebar)
    if STACK.status()["n_chunks"] == 0:
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": "Nothing indexed yet — ingest a PDF first."}]
        yield history, "", "", LAST
        return

    try:
        res = STACK.query(question)
    except Exception as e:
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": f"query failed: {type(e).__name__}: {e}"}]
        yield history, "", "", LAST
        return

    LAST = {"answer": res["answer"], "chunks": res["chunks"], "citations": res["citations"]}

    history += [{"role": "user", "content": question},
                {"role": "assistant", "content": res["answer"]}]

    sources = "\n\n".join(
        f"**[{c['marker']}] p{c['page_number']}** — {' > '.join(c['heading_path'])}\n\n"
        f"> {c['text'][:600]}"
        for c in res["citations"]
    )
    meta = (
        f"- query_type: `{res['query_type']}`\n"
        f"- chunks retrieved: {len(res['chunks'])}\n"
        f"- stages: `{' → '.join(res['stages'])}`\n"
        f"- cost so far: `${res['cost_usd']:.4f}`"
    )
    yield history, sources, meta, LAST


# ------------------------------------------------------- hallucination check
def do_verify(_btn, last):
    global DETECTOR
    if not last or not last.get("answer") or not last.get("chunks"):
        return "ask a question first"
    if DETECTOR is None:
        DETECTOR = HallucinationDetector(cost_tracker=STACK.tracker)
    try:
        report = DETECTOR.detect(answer=last["answer"], chunks=last["chunks"])
    except Exception as e:
        return f"verification failed: {type(e).__name__}: {e}"
    return gr.Dataframe(
        [[c.claim, c.verdict, c.reasoning] for c in report.claims],
        headers=["claim", "verdict", "reasoning"],
        label=(f"{report.n_entailed} entailed / {report.n_refuted} refuted / "
               f"{report.n_unsupported} unsupported"),
    )


# ---------------------------------------------------------------------- UI
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Fintaskit") as demo:
        gr.Markdown("# Fintaskit\nHybrid retrieval (vector + BM25 + RRF) over real financial filings.")

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Corpus")
                file_in = gr.File(label="PDFs", file_count="multiple",
                                  file_types=[".pdf"], type="filepath")
                ingest_btn = gr.Button("Ingest", variant="primary")
                ingest_out = gr.Textbox(label="Ingest log", lines=5)

                gr.Markdown("### Pipeline config")
                r_retriever = gr.Radio(RETRIEVERS, value="hybrid", label="Retriever")
                r_chunker = gr.Dropdown(CHUNKERS, value="recursive", label="Chunker")
                r_use_parent = gr.Checkbox(False, label="small-to-big (child → parent)")
                r_chunk_size = gr.Slider(100, 1200, value=400, step=50, label="chunk_size")
                r_overlap = gr.Slider(0, 300, value=60, step=10, label="overlap")
                r_parent_size = gr.Slider(200, 2400, value=800, step=100, label="parent_size")
                r_child_size = gr.Slider(50, 600, value=150, step=25, label="child_size")
                r_model = gr.Dropdown(["gpt-5-mini", "gpt-4o-mini", "gpt-4o"],
                                      value="gpt-5-mini", label="LLM")
                status = gr.Textbox(label="Stack status", interactive=False)

            with gr.Column(scale=3):
                last_state = gr.State({})
                chat = gr.Chatbot(label="Q&A", height=480, type="messages")
                q_in = gr.Textbox(placeholder="What was the CET1 capital ratio?", label="Question")
                with gr.Row():
                    send = gr.Button("Ask", variant="primary")
                    verify_btn = gr.Button("Verify last answer")
                faith_out = gr.Markdown()
                with gr.Accordion("Retrieved sources", open=True):
                    sources_md = gr.Markdown()
                with gr.Accordion("Run metadata", open=True):
                    meta_md = gr.Markdown()

        sidebar = (r_retriever, r_chunker, r_use_parent, r_chunk_size, r_overlap,
                   r_parent_size, r_child_size, r_model)

        ingest_btn.click(do_ingest, [file_in, *sidebar], [ingest_out, status])
        send.click(do_query, [q_in, chat, *sidebar], [chat, sources_md, meta_md, last_state])
        q_in.submit(do_query, [q_in, chat, *sidebar], [chat, sources_md, meta_md, last_state])
        verify_btn.click(do_verify, [verify_btn, last_state], faith_out)
        for w in (r_retriever, r_chunker, r_use_parent, r_model):
            w.change(sync_stack, sidebar, status)

    return demo


if __name__ == "__main__":
    for p in sorted(UPLOADS.glob("*.pdf")):
        STACK.ingest(p)
    print("preloaded:", STACK.status())
    build_ui().launch(server_name="127.0.0.1", server_port=7860)