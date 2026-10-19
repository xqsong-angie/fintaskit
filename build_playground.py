"""
Interactive RAG Evaluation Studio & Benchmark Playground
Built with Gradio for Fintaskit.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import plotly.graph_objects as go
import time
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
from src.api.app_factory import PipelineConfig, build_stack
from src.core.config import configure_langsmith
from src.evaluators.hallucination import HallucinationDetector
from src.observability import CostTracker

configure_langsmith()

UPLOADS = ROOT / "data" / "uploads"
PRESET_FILES = [p.name for p in sorted(UPLOADS.glob("*.pdf"))] if UPLOADS.exists() else []

# 全局初始化 Stack
TRACKER = CostTracker()
STACK = build_stack(
    PipelineConfig(persist_dir="./chroma_ui", collection="ui_playground"),
    tracker=TRACKER,
)
DETECTOR: HallucinationDetector | None = None


# ---------------------------------------------------------------- design system
INK = "#141726"
INK_SOFT = "#5B6478"
LINE = "#E6E9F4"
LINE_SOFT = "#EFF1F7"
INDIGO = "#4F46E5"

THEME = gr.themes.Base(
    primary_hue=colors.indigo,
    secondary_hue=colors.sky,
    neutral_hue=colors.zinc,
    radius_size=sizes.radius_lg,
    text_size=sizes.text_md,
    font=(fonts.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(
        fonts.GoogleFont("JetBrains Mono"),
        "ui-monospace",
        "SFMono-Regular",
        "monospace",
    ),
).set(
    body_background_fill="#F4F6FC",
    body_text_color=INK,
    body_text_color_subdued="#6E7793",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#EEF1FA",
    border_color_primary="#FFFFFF",
    border_color_primary_dark="transparent",
    color_accent=INDIGO,
    color_accent_soft="#EEF0FF",
    link_text_color=INDIGO,
    block_background_fill="#FFFFFF",
    block_border_color=LINE,
    block_label_text_color="#414A63",
    block_label_text_weight="600",
    block_title_text_color=INK,
    block_shadow="0 1px 2px rgba(20,23,38,.04), 0 14px 34px -22px rgba(20,23,38,.35)",
    block_padding="16px",
    block_radius="16px",
    input_background_fill="#FFFFFF",
    input_background_fill_hover="#FFFFFF",
    input_background_fill_focus="#FFFFFF",
    input_border_color="#DFE3F0",
    input_border_color_hover="#C9D0E4",
    input_border_color_focus=INDIGO,
    input_shadow_focus="0 0 0 3px rgba(79,70,229,.14)",
    input_placeholder_color="#A6AEC4",
    input_radius="11px",
    table_even_background_fill="#FAFBFE",
    table_border_color="#EDEFF7",
    checkbox_border_color_selected=INDIGO,
    checkbox_background_color_selected=INDIGO,
    button_primary_background_fill=INDIGO,
    button_primary_background_fill_hover="#4338CA",
    button_primary_border_color=INDIGO,
    button_primary_border_color_hover="#4338CA",
    button_primary_text_color="#FFFFFF",
    button_primary_shadow="0 1px 2px rgba(79,70,229,.24)",
    button_primary_shadow_hover="0 8px 20px -10px rgba(79,70,229,.65)",
    button_primary_shadow_active="0 1px 2px rgba(79,70,229,.24)",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F3F5FC",
    button_secondary_border_color="#DFE3F0",
    button_secondary_border_color_hover="#C9D0E4",
    button_secondary_text_color="#2A3145",
    button_secondary_shadow="0 1px 2px rgba(20,23,38,.05)",
    button_secondary_shadow_hover="0 6px 16px -10px rgba(20,23,38,.45)",
    button_secondary_shadow_active="0 1px 2px rgba(20,23,38,.05)",
)

APP_CSS = f"""
footer {{ display: none !important; }}

.gradio-container {{
  --ft-ink: {INK};
  --ft-soft: {INK_SOFT};
  --ft-line: {LINE};
  --ft-line-soft: {LINE_SOFT};
  --ft-indigo: {INDIGO};
  --ft-indigo-tint: #EEF0FF;
  --ft-slate-tint: #F1F5F9;
  --ft-radius: 16px;
  --ft-lift: 0 1px 2px rgba(20,23,38,.04), 0 14px 34px -22px rgba(20,23,38,.35);
  max-width: 1440px !important;
}}

/* ---------------- hero ---------------- */
.gradio-container .hero {{
  border-radius: var(--ft-radius);
  border: 1px solid var(--ft-line);
  padding: 22px 26px 20px;
  gap: 12px;
  background:
    radial-gradient(900px 280px at 6% -60%, #E4E1FF 0%, rgba(228,225,255,0) 62%),
    radial-gradient(760px 240px at 92% -70%, #D6EFFF 0%, rgba(214,239,255,0) 60%),
    linear-gradient(180deg, #FFFFFF 0%, #FAFBFF 100%);
  box-shadow: var(--ft-lift);
}}
.gradio-container .hero h1 {{
  margin: 0 0 4px;
  font-size: 30px;
  line-height: 1.18;
  font-weight: 750;
  letter-spacing: -.022em;
  background: linear-gradient(94deg, #191D33 6%, #4338CA 58%, #0284C7 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}}
.gradio-container .hero p {{
  margin: 0;
  max-width: 82ch;
  font-size: 14.5px;
  line-height: 1.6;
  color: var(--ft-soft);
}}


/* ---------------- hero feature dropdowns ---------------- */
.gradio-container .hero .feat-dd {{
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}}
.gradio-container .hero .feat-dd label > span {{
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #8A93AC !important;
}}
.gradio-container .hero .feat-dd input,
.gradio-container .hero .feat-dd .wrap,
.gradio-container .hero .feat-dd .wrap-inner {{
  background: #F5F7FF !important;
  color: #1F243A !important;
  font-size: 13px;
  font-weight: 620;
}}
.gradio-container .hero .feat-dd .wrap {{
  border: 1px solid #DCE2F3 !important;
  border-radius: 999px !important;
  box-shadow: 0 1px 2px rgba(20,23,38,.04) !important;
}}
.gradio-container .hero .feat-dd .wrap:hover {{ border-color: #C9D1EC !important; }}
.gradio-container .hero .feat-dd .wrap:focus-within {{
  border-color: var(--ft-indigo) !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,.14) !important;
}}

/* ---------------- badges ---------------- */
.gradio-container .badge-row {{ gap: 8px !important; flex-wrap: wrap; }}
.gradio-container .badge {{
  padding: 5px 11px;
  border-radius: 999px;
  background: rgba(255,255,255,.86);
  border: 1px solid var(--ft-line);
  box-shadow: 0 1px 2px rgba(20,23,38,.04);
  color: #414A63;
  font-size: 12px;
  font-weight: 620;
  letter-spacing: .005em;
  white-space: nowrap;
}}
.gradio-container .badge p {{ margin: 0 !important; color: inherit !important; }}

/* ---------------- stat tiles ---------------- */
.gradio-container .stats-row {{ gap: 14px !important; }}
.gradio-container .stat {{
  border-radius: 14px;
  border: 1px solid var(--ft-line);
  background: #FFFFFF;
  box-shadow: var(--ft-lift);
  padding: 13px 16px 14px;
  gap: 0 !important;
}}
.gradio-container .stat .ft-k {{
  font-size: 10px;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: #8A93AC;
  font-weight: 700;
}}
.gradio-container .stat .ft-v {{
  font-size: 26px;
  line-height: 1.2;
  font-weight: 750;
  letter-spacing: -.02em;
  color: var(--ft-ink);
  font-variant-numeric: tabular-nums;
}}
.gradio-container .stat .ft-s {{ font-size: 11.5px; color: #98A0B6; }}
.gradio-container .stat.accent {{ border-color: #D9DDFB; background: linear-gradient(180deg,#FBFBFF 0%,#F6F7FF 100%); }}
.gradio-container .stat.accent .ft-v {{ color: var(--ft-indigo); }}

/* ---------------- cards ---------------- */
    .gradio-container .card {{
      border-radius: var(--ft-radius);
      border: 1px solid var(--ft-line);
      background: #FAFBFF;
      box-shadow: var(--ft-lift);
      padding: 16px 18px 18px;
      gap: 12px;
    }}
.gradio-container .card-head,
.gradio-container .card-sub {{ padding: 0 !important; background: transparent !important; }}
.gradio-container .card-head p {{
  margin: 0 !important;
  font-size: 14.5px;
  line-height: 1.45;
  font-weight: 720;
  letter-spacing: -.012em;
  color: var(--ft-ink);
}}
.gradio-container .card-sub p {{
  margin: -4px 0 0 !important;
  font-size: 12.5px;
  line-height: 1.55;
  color: #7C86A0;
}}
.gradio-container .step-tag {{
  display: inline-block;
  margin-right: 8px;
  padding: 1px 7px;
  border-radius: 6px;
  background: var(--ft-indigo-tint);
  color: var(--ft-indigo);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .04em;
  vertical-align: 1.5px;
}}

/* ---------------- result panels ---------------- */
.gradio-container .panel-head {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 !important;
  background: transparent !important;
}}
.gradio-container .panel-head p {{ margin: 0 !important; }}
.gradio-container .panel-head b {{
  font-size: 13.5px;
  font-weight: 700;
  letter-spacing: -.005em;
  color: var(--ft-ink);
}}
.gradio-container .panel-head i {{
  font-style: normal;
  font-size: 12px;
  font-weight: 600;
  color: #8A93AC;
}}
.gradio-container .panel-head::before {{
  content: "";
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--dot, var(--ft-indigo));
  box-shadow: 0 0 0 3px var(--dot-tint, var(--ft-indigo-tint));
  flex: none;
}}
.gradio-container .card-a {{ --dot: #A6B0C4; --dot-tint: var(--ft-slate-tint); }}
.gradio-container .card-b {{ --dot: var(--ft-indigo); --dot-tint: var(--ft-indigo-tint); }}

.gradio-container .result {{
  max-height: 440px;
  overflow-y: auto;
  padding-right: 6px;
  font-size: 13.5px;
  line-height: 1.62;
  color: #2B3244;
}}
.gradio-container .result h4 {{
  margin: 16px 0 8px;
  font-size: 13px;
  font-weight: 700;
  color: var(--ft-soft);
  letter-spacing: .02em;
}}
.gradio-container .result h4:first-child {{ margin-top: 0; }}
.gradio-container .result blockquote {{
  margin: 0 0 10px;
  padding: 10px 12px;
  background: #F8FAFD;
  border: 1px solid var(--ft-line-soft);
  border-left: 3px solid #CBD3E2;
  border-radius: 10px;
  color: #3C4459;
  font-size: 13px;
}}
.gradio-container .result blockquote p {{ margin: 0; }}
.gradio-container .result strong {{ color: var(--ft-indigo); font-weight: 700; }}
.gradio-container .result em {{ color: #97A0B7; }}
.gradio-container .result::-webkit-scrollbar {{ width: 8px; }}
.gradio-container .result::-webkit-scrollbar-thumb {{
  background: #DCE1EE;
  border-radius: 99px;
  border: 2px solid #FFFFFF;
}}
.gradio-container .empty {{ color: #A2AAC0; font-size: 13px; font-style: italic; }}

/* ---------------- tabs ---------------- */
.gradio-container .tabs > .tab-wrapper {{ gap: 0 !important; }}
.gradio-container .tabs > .tab-wrapper > .tab-container {{
  gap: 6px;
  background: transparent;
  border: none;
  padding: 2px 0 14px;
}}
.gradio-container .tabs button[role="tab"] {{
  border: 1px solid transparent;
  background: transparent;
  color: var(--ft-soft);
  font-size: 13.5px;
  font-weight: 650;
  padding: 9px 16px;
  border-radius: 11px;
  transition: background .16s ease, color .16s ease, box-shadow .16s ease;
}}
.gradio-container .tabs button[role="tab"]:hover {{
  background: #EDF0F9;
  color: var(--ft-ink);
}}
.gradio-container .tabs button[role="tab"][aria-selected="true"] {{
  background: #FFFFFF;
  color: var(--ft-indigo);
  border-color: var(--ft-line);
  box-shadow: 0 1px 2px rgba(20,23,38,.05), 0 10px 24px -18px rgba(20,23,38,.55);
}}

/* ---------------- controls ---------------- */
.gradio-container .card button {{
  border-radius: 11px;
  font-weight: 650;
  font-size: 13.5px;
  letter-spacing: .002em;
  transition: transform .12s ease, box-shadow .16s ease, background .16s ease;
}}
.gradio-container .card button.primary {{
  background: linear-gradient(180deg, #5B54EA 0%, #4A41DE 100%);
  border-color: #4A41DE;
  color: #FFFFFF;
  box-shadow: 0 1px 2px rgba(79,70,229,.28), 0 10px 22px -14px rgba(79,70,229,.85);
}}
.gradio-container .card button.primary:hover {{
  background: linear-gradient(180deg, #655EEF 0%, #4F46E5 100%);
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(79,70,229,.24), 0 14px 28px -14px rgba(79,70,229,.9);
}}
.gradio-container .card button.primary:active {{ transform: translateY(0); }}
.gradio-container .card button.secondary {{
  color: #2A3145;
  background: #FFFFFF;
  border-color: #DFE3F0;
}}
.gradio-container .card button.secondary:hover {{ background: #F3F5FC; }}

.gradio-container .card label > span,
.gradio-container .card .block-label {{
  font-size: 12px !important;
  font-weight: 640 !important;
  letter-spacing: .01em;
  color: #2F3652 !important;
}}

/* 浅色下拉 + 深色文字 */
.gradio-container .card .dropdown,
.gradio-container .card .dropdown > div,
.gradio-container .card .dropdown input,
.gradio-container .card select,
.gradio-container .card .multiselect {{
  background: #F5F7FF !important;
  color: #1F243A !important;
}}
.gradio-container .card .dropdown input::placeholder {{
  color: #7E869F !important;
}}
.gradio-container .card .dropdown .label-wrap,
.gradio-container .card .dropdown .wrap,
.gradio-container .card .dropdown .border {{
  border-color: #DCE2F3 !important;
}}
.gradio-container .card .dropdown:hover .border,
.gradio-container .card .dropdown:hover .label-wrap,
.gradio-container .card .dropdown:focus-within .border {{
  border-color: #C9D1EC !important;
}}
.gradio-container .card .dropdown .option,
.gradio-container .card .dropdown .item,
.gradio-container .card .dropdown .dropdown-item {{
  background: #FFFFFF !important;
  color: #1F243A !important;
}}

/* ---------------- console (ingestion log) ---------------- */
.gradio-container .console textarea,
.gradio-container .console input {{
  background: #F8FAFD !important;
  border-color: var(--ft-line-soft) !important;
  border-radius: 11px !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  line-height: 1.75 !important;
  color: #39415A !important;
  box-shadow: none !important;
}}
.gradio-container .console textarea:focus,
.gradio-container .console textarea:focus-within {{
  border-color: #C7CDF7 !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,.10) !important;
}}

/* ---------------- accordion ---------------- */
.gradio-container .card button.label-wrap {{
  border: 1px dashed #DCE1F0;
  border-radius: 11px;
  background: #FBFCFF;
  color: #525B72;
  font-size: 12.5px;
  font-weight: 620;
  padding: 9px 12px;
}}
.gradio-container .card button.label-wrap:hover {{ border-color: #C2C9E4; background: #F5F7FE; }}

/* ---------------- dataframe ---------------- */
.gradio-container .verify-card .table-wrap {{ border-radius: 12px; overflow: auto; }}
.gradio-container .verify-card table {{ font-size: 13px; }}
.gradio-container .verify-card th {{
  background: #F7F9FD !important;
  color: #525B72 !important;
  font-weight: 700 !important;
  font-size: 11.5px !important;
  letter-spacing: .06em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--ft-line) !important;
}}
.gradio-container .verify-card td {{ border-bottom: 1px solid #F1F3F9 !important; }}
"""


# ------------------------------------------------------------------- data logic
def _stats_html() -> tuple[str, str, str]:
    st = STACK.status()
    return (
        '<div class="ft-k">Documents</div>'
        f'<div class="ft-v">{st["n_documents"]}</div>'
        '<div class="ft-s">indexed PDFs</div>',
        '<div class="ft-k">Chunks</div>'
        f'<div class="ft-v">{st["n_chunks"]:,}</div>'
        '<div class="ft-s">vectorised segments</div>',
        '<div class="ft-k">Session Spend</div>'
        f'<div class="ft-v">${st["cost_usd"]:.4f}</div>'
        '<div class="ft-s">across all API calls</div>',
    )
 
 
def run_ingest(selected_preset, uploaded_files):
    """处理文档加载：优先读预置 Preset PDF（全走 Cache，零 API 消耗）"""
    targets = []
    if selected_preset:
        targets.append(UPLOADS / selected_preset)
    if uploaded_files:
        for p in uploaded_files:
            targets.append(Path(p))
 
    if not targets:
        return "⚠️ Please select a preset document or upload a PDF first.", *_stats_html()
 
    log_lines = []
    for src in targets:
        try:
            r = STACK.ingest(src)
            tag = "already indexed (Cache Hit)" if r.duplicate or r.cache_hit else f"+{r.n_chunks} chunks"
            log_lines.append(f"✅ `{src.name}` — {tag} · ${r.cost_usd:.4f}")
        except Exception as e:
            log_lines.append(f"❌ `{src.name}` — {type(e).__name__}: {e}")
 
    return "\n".join(log_lines), *_stats_html()
 
 
def _bucket(verdict: str) -> str:
    """verdict 归类为 good / bad / other。先判负面词，因为 'unsupported' 里含 'support'。
    ⚠️ 关键词是按常见写法猜的，请对照 HallucinationDetector 真实的 verdict 取值调整。"""
    v = str(verdict).lower()
    if any(k in v for k in ("unsupport", "contradict", "halluc", "not_", "false")):
        return "bad"
    if any(k in v for k in ("support", "entail", "faithful", "true")):
        return "good"
    return "other"
 
 
def _metrics_html(claims, latency, cost, n_cites):
    """顶部指标：Faithfulness 大数字 + 分段条，加 Latency / Cost / Citations 三块"""
    def tile(k, v, s=""):
        return (f'<div class="stat"><div class="ft-k">{k}</div>'
                f'<div class="ft-v">{v}</div><div class="ft-s">{s}</div></div>')
 
    bar = ""
    if not claims:
        score_block = ('<div class="ft-k">Faithfulness</div><div class="ft-v">—</div>'
                       '<div class="ft-s">no claims to verify</div>')
    else:
        buckets = [_bucket(c.verdict) for c in claims]
        n = len(buckets)
        good, bad = buckets.count("good"), buckets.count("bad")
        other = n - good - bad
        score_block = (
            '<div class="ft-k">Faithfulness</div>'
            f'<div class="ft-v">{good / n:.0%}</div>'
            f'<div class="ft-s">{good}/{n} claims supported</div>'
        )
 
        def seg(w, color):
            return f'<div style="width:{w / n * 100:.1f}%;background:{color}"></div>' if w else ""
 
        bar = (
            '<div style="display:flex;height:10px;border-radius:99px;overflow:hidden;'
            'background:#EEF1FA;margin-top:14px">'
            f'{seg(good, "#10B981")}{seg(other, "#F59E0B")}{seg(bad, "#EF4444")}</div>'
            '<div class="ft-s" style="margin-top:6px;color:#98A0B6">'
            f'<span style="color:#10B981">●</span> supported {good} &nbsp; '
            f'<span style="color:#F59E0B">●</span> other {other} &nbsp; '
            f'<span style="color:#EF4444">●</span> unsupported {bad}</div>'
        )
 
    return (
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px">'
        f'<div class="stat accent">{score_block}</div>'
        + tile("Latency", f"{latency:.1f}s", "retrieval + generation")
        + tile("Cost", f"${cost:.4f}", "this run")
        + tile("Citations", n_cites, "retrieved sources")
        + "</div>" + bar
    )
 
 
HIST_COLS = ["#", "Retriever", "Chunker", "VLM", "Parent", "Faithfulness", "Latency", "Cost", "Query"]
 
 
def _hist_rows(history):
    return [[h[c] for c in HIST_COLS] for h in history]
 
 
# ---- 雷达图：5 个维度，都归一化到 0~1，越大越好 ----
# Faithfulness / No hallucination 来自 claim 校验；后三项是启发式打分，阈值可按需调整
RADAR_AXES = ["Faithfulness", "No hallucination", "Citation coverage", "Speed", "Cost efficiency"]
CITE_TARGET = 5        # 引用数达到这个值算满分
LATENCY_BUDGET = 30.0  # 秒；超过则 Speed = 0
COST_BUDGET = 0.05     # 美元；超过则 Cost efficiency = 0
RADAR_COLORS = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6"]
 
 
def _scores(claims, latency, cost, n_cites):
    n = len(claims)
    if n:
        buckets = [_bucket(c.verdict) for c in claims]
        faith = buckets.count("good") / n
        clean = 1 - buckets.count("bad") / n
    else:
        faith = clean = 0.0
    return {
        "Faithfulness": faith,
        "No hallucination": clean,
        "Citation coverage": min(n_cites / CITE_TARGET, 1.0),
        "Speed": max(0.0, 1 - latency / LATENCY_BUDGET),
        "Cost efficiency": max(0.0, 1 - cost / COST_BUDGET),
    }
 
 
def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
 
 
def _radar_fig(history):
    """所有已保存实验叠在一张雷达图上，最新一次加粗并填充"""
    runs = [h for h in history if h.get("_scores")][-6:]
    fig = go.Figure()
    for i, h in enumerate(runs):
        color = RADAR_COLORS[i % len(RADAR_COLORS)]
        latest = i == len(runs) - 1
        vals = [h["_scores"][a] for a in RADAR_AXES]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1],
            theta=RADAR_AXES + RADAR_AXES[:1],
            name=f'#{h["#"]} {h["Retriever"]} · {h["Chunker"]}',
            fill="toself" if latest else "none",
            fillcolor=_rgba(color, 0.16),
            line=dict(color=color, width=3 if latest else 1.5),
            marker=dict(size=6 if latest else 4),
        ))
    if not runs:
        fig.add_annotation(text="Run an experiment to see the radar",
                           showarrow=False, font=dict(size=13, color="#A2AAC0"))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0],
                            tickfont=dict(size=10, color="#98A0B6"),
                            gridcolor="#E6E9F4", linecolor="#E6E9F4"),
            angularaxis=dict(tickfont=dict(size=12, color="#141726"),
                             gridcolor="#E6E9F4", linecolor="#E6E9F4"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        margin=dict(l=50, r=50, t=20, b=40),
        height=440,
    )
    return fig
 
 
def _fail(msg, history):
    """出错/空输入时的统一返回，保持输出个数和 exp_outputs 一致"""
    return msg, "", "", _radar_fig(history), [], _hist_rows(history), history, *_stats_html()
 
 
def run_experiment(question, chunker, retriever, vlm, parent, history):
    """Save & Run：按当前配置检索+生成 → 幻觉检测 → 指标 + 历史记录"""
    global DETECTOR
    history = history or []
 
    if not question or not question.strip():
        return _fail("*Please enter a query.*", history)
    if STACK.status()["n_chunks"] == 0:
        return _fail("> ⚠️ No documents indexed yet. Click **Ingest & Index** first.", history)
 
    cost0 = STACK.status()["cost_usd"]
    t0 = time.time()
 
    # Pass only the fields the header controls, so the stack keeps whatever
    # chunk_size / model it was built with instead of being reset to defaults.
    try:
        res = STACK.reconfigure(
            retriever=retriever,
            chunker=chunker,
            use_vlm=(vlm == "On"),
            use_parent=bool(parent),
        ).query(question)
    except Exception as e:
        return _fail(f"> ⚠️ Query failed: {e}", history)
    latency = time.time() - t0
 
    raw_answer = res.get("answer", "")
    answer = raw_answer or "*No answer returned.*"
    citations = res.get("citations", [])
    sources = "\n\n".join(
        f"**[{c['marker']}] Page {c['page_number']}** — {' > '.join(c['heading_path'])}\n\n> {c['text'][:500]}"
        for c in citations
    ) or "*No citations returned.*"
 
    # ---- 幻觉检测 ----
    claims, rows = [], []
    if raw_answer:
        if DETECTOR is None:
            DETECTOR = HallucinationDetector(cost_tracker=STACK.tracker)
        try:
            report = DETECTOR.detect(answer=raw_answer, chunks=res.get("chunks", []))
            claims = report.claims
            rows = [[c.claim, c.verdict, c.reasoning] for c in claims]
        except Exception as e:
            rows = [["Error", "Failed", str(e)]]
 
    cost = STACK.status()["cost_usd"] - cost0
    good = sum(_bucket(c.verdict) == "good" for c in claims)
    score = f"{good / len(claims):.0%}" if claims else "—"
 
    history = history + [{
        "#": len(history) + 1,
        "Retriever": retriever,
        "Chunker": chunker,
        "VLM": vlm,
        "Parent": "on" if parent else "off",
        "Faithfulness": score,
        "Latency": f"{latency:.1f}s",
        "Cost": f"${cost:.4f}",
        "Query": question[:40],
        "_scores": _scores(claims, latency, cost, len(citations)),
    }]
 
    metrics = _metrics_html(claims, latency, cost, len(citations))
    return (answer, sources, metrics, _radar_fig(history), rows,
            _hist_rows(history), history, *_stats_html())
 
 
# ---- 界面构建 (Gradio Blocks) ----
def build_playground():
    with gr.Blocks(title="Fintaskit - RAG Eval Studio", theme=THEME, css=APP_CSS) as demo:
        history_state = gr.State([])
 
        # ---------- Hero：配置栏 ----------
        with gr.Column(elem_classes="hero"):
            gr.Markdown(
                "# FinTasKit · Financial Multimodal RAG & Evaluation Studio\n"
                "Pick a configuration, save it, and inspect the answer, sources and "
                "faithfulness metrics for complex financial PDFs."
            )
            with gr.Row(elem_classes="badge-row"):
                retriever_opt = gr.Dropdown(
                    choices=["hybrid", "vector", "bm25"],
                    value="hybrid",
                    label="Retrieval",
                    elem_classes="feat-dd",
                    min_width=170,
                )
                vlm_opt = gr.Dropdown(
                    choices=["On", "Off"],
                    value="On",
                    label="VLM Chart Captioning",
                    elem_classes="feat-dd",
                    min_width=170,
                )
                chunker_opt = gr.Dropdown(
                    choices=["fixed_size", "recursive", "parent_child"],
                    value="parent_child",
                    label="Chunker",
                    elem_classes="feat-dd",
                    min_width=170,
                )
                # Only meaningful for parent_child, which is the only chunker
                # that emits a parent tier. Toggled off automatically elsewhere.
                parent_opt = gr.Checkbox(
                    value=True,
                    label="Parent expansion",
                    info="small-to-big: retrieve children, generate from parents",
                    interactive=True,
                )
 
                def sync_parent(chunker, parent):
                    if chunker == "parent_child":
                        return gr.update(interactive=True, value=parent)
                    return gr.update(interactive=False, value=False)
 
                chunker_opt.change(
                    sync_parent,
                    inputs=[chunker_opt, parent_opt],
                    outputs=[parent_opt],
                )
 
        # ---------- Stat bar ----------
        with gr.Row(elem_classes="stats-row"):
            stat_tiles = []
            for slot in range(3):
                with gr.Column(elem_classes="stat accent" if slot == 2 else "stat"):
                    stat_tiles.append(gr.HTML(_stats_html()[slot]))
 
        with gr.Row(equal_height=False):
            # ---------- 左侧：数据集 ----------
            with gr.Column(scale=1, min_width=310):
                with gr.Column(elem_classes="card"):
                    gr.Markdown(
                        '<span class="step-tag">STEP 1</span>Dataset',
                        elem_classes="card-head",
                        sanitize_html=False,
                    )
                    gr.Markdown(
                        "*Pre-cached benchmark PDFs — ingestion runs at zero API cost.*",
                        elem_classes="card-sub",
                    )
                    doc_source = gr.Radio(
                        choices=["Preset", "Upload"],
                        value="Preset",
                        label="Document source",
                        interactive=True,
                    )
                    preset_dropdown = gr.Dropdown(
                        choices=PRESET_FILES,
                        value=PRESET_FILES[0] if PRESET_FILES else None,
                        label="Benchmark document",
                        interactive=True,
                    )
                    file_uploader = gr.File(
                        label="PDF files",
                        file_count="multiple",
                        file_types=[".pdf"],
                        visible=False,
                        interactive=True,
                    )
                    ingest_btn = gr.Button("Ingest & Index", variant="primary")
                    ingest_log = gr.Textbox(
                        label="Ingestion log",
                        interactive=False,
                        lines=4,
                        max_lines=9,
                        elem_classes="console",
                        value="*Waiting for the first ingest…*",
                    )
 
            # ---------- 右侧：实验台（不再有 Tabs） ----------
            with gr.Column(scale=3):
                with gr.Column(elem_classes="card"):
                    gr.Markdown(
                        '<span class="step-tag">STEP 2</span>Save configuration & run',
                        elem_classes="card-head",
                        sanitize_html=False,
                    )
                    gr.Markdown(
                        "*Choose a configuration in the header, enter a query, then save — "
                        "answer, sources and faithfulness metrics are generated in one go.*",
                        elem_classes="card-sub",
                    )
                    query_input = gr.Textbox(
                        placeholder="e.g. What was total operating revenue in Q3?",
                        label="Benchmark query",
                        lines=2,
                    )
                    save_btn = gr.Button("Save & Run", variant="primary")
 
                with gr.Tabs():
                    # TAB 1：答案 + 来源
                    with gr.Tab("Answer & Sources"):
                        with gr.Row():
                            with gr.Column(elem_classes="card card-b"):
                                gr.Markdown("<b>Answer</b>", elem_classes="panel-head", sanitize_html=False)
                                answer_md = gr.Markdown(
                                    "*The answer will appear here…*", elem_classes="result"
                                )
                            with gr.Column(elem_classes="card card-a"):
                                gr.Markdown("<b>Sources</b>", elem_classes="panel-head", sanitize_html=False)
                                sources_md = gr.Markdown(
                                    "*Retrieved chunks will appear here…*", elem_classes="result"
                                )
 
                    # TAB 2：指标可视化
                    with gr.Tab("Metrics & Radar"):
                        metrics_html = gr.HTML()
 
                        with gr.Column(elem_classes="card"):
                            gr.Markdown(
                                '<span class="step-tag">STEP 3</span>Evaluation radar',
                                elem_classes="card-head",
                                sanitize_html=False,
                            )
                            gr.Markdown(
                                "*Every saved run is overlaid; the latest one is highlighted. "
                                "All axes are normalised to 0–1, larger is better.*",
                                elem_classes="card-sub",
                            )
                            radar_plot = gr.Plot(value=_radar_fig([]), show_label=False)
 
                        with gr.Column(elem_classes="card verify-card"):
                            gr.Markdown(
                                '<span class="step-tag">LOG</span>Saved experiments',
                                elem_classes="card-head",
                                sanitize_html=False,
                            )
                            history_df = gr.Dataframe(
                                headers=HIST_COLS, wrap=True, interactive=False,
                            )
 
                        with gr.Column(elem_classes="card verify-card"):
                            with gr.Accordion("Claim-level details", open=False):
                                claims_df = gr.Dataframe(
                                    headers=["Claim", "Verdict", "Reasoning"],
                                    wrap=True,
                                    interactive=False,
                                )
 
        # ---------- 事件绑定 ----------
        def toggle_doc_source(src):
            return gr.update(visible=(src == "Preset")), gr.update(visible=(src == "Upload"))
 
        doc_source.change(
            toggle_doc_source,
            inputs=[doc_source],
            outputs=[preset_dropdown, file_uploader],
        )
 
        def run_ingest_choice(src, selected_preset, uploaded_files):
            if src == "Upload":
                return run_ingest(None, uploaded_files)
            return run_ingest(selected_preset, None)
 
        ingest_btn.click(
            run_ingest_choice,
            inputs=[doc_source, preset_dropdown, file_uploader],
            outputs=[ingest_log, *stat_tiles],
        )
 
        exp_inputs = [query_input, chunker_opt, retriever_opt, vlm_opt, parent_opt, history_state]
        exp_outputs = [answer_md, sources_md, metrics_html, radar_plot, claims_df,
                       history_df, history_state, *stat_tiles]
 
        save_btn.click(run_experiment, inputs=exp_inputs, outputs=exp_outputs)
        query_input.submit(run_experiment, inputs=exp_inputs, outputs=exp_outputs)
 
        demo.load(_stats_html, outputs=stat_tiles)
 
    return demo
 
 
# 放在模块级：这样 `gradio app.py` 热重载模式也能找到 demo
if UPLOADS.exists():
    for _p in sorted(UPLOADS.glob("*.pdf")):
        try:
            STACK.ingest(_p)
        except Exception:
            pass
    print("[Pre-warmed Cache]: Successfully loaded pre-cached PDF benchmarks.")
 
demo = build_playground()
 
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)