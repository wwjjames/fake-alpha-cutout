"""Polished local Streamlit MVP for pseudo-transparency alpha matting."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from mvp_inference import checkerboard_preview, ensure_model_files, fake_probability, load_models, predict_alpha, render_rgba, select_device

ROOT = Path(__file__).resolve().parent
STAGE_A = ROOT / "pseudo_transparency_detection/models/400_3depth(baseline).pth"
STAGE_B = ROOT / "matting_models/stage_b_alpha_v1/best.pt"

st.set_page_config(page_title="Cutout Lab — 伪透明抠图", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="正在唤醒本地模型…")
def cached_models() -> tuple:
    ensure_model_files(ROOT, STAGE_A, STAGE_B)
    device = select_device()
    stage_a, stage_b, config = load_models(STAGE_A, STAGE_B, device)
    return device, stage_a, stage_b, config


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def clear_stale_result(file_id: str) -> None:
    if st.session_state.get("file_id") != file_id:
        for key in ("result", "alpha", "probability", "decision", "threshold_used", "used_matting"):
            st.session_state.pop(key, None)
        st.session_state["file_id"] = file_id


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { --ink:#edf2ff; --muted:#93a0bd; --line:rgba(197,210,255,.14); --panel:rgba(17,24,43,.74); --teal:#55ebd2; --violet:#9788ff; }
        .stApp { background:radial-gradient(circle at 8% -15%,#2b385d 0,transparent 31%),radial-gradient(circle at 92% 6%,#2d1f62 0,transparent 28%),#090d19; color:var(--ink); font-family:'Manrope',sans-serif; }
        .block-container { max-width:1320px; padding:2.5rem 3rem 4rem; }
        #MainMenu, footer, header { visibility:hidden; }
        h1, h2, h3, p, [data-testid="stMarkdownContainer"] { font-family:'Manrope',sans-serif; }
        [data-testid="stFileUploader"], [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--line); border-radius:18px; background:var(--panel); }
        [data-testid="stFileUploader"] { padding:1.05rem 1.15rem .75rem; }
        [data-testid="stFileUploaderDropzone"] { background:rgba(81,103,166,.08); border:1px dashed rgba(133,155,236,.38); border-radius:13px; min-height:190px; }
        [data-testid="stFileUploaderDropzone"] * { color:var(--ink) !important; }
        [data-testid="stFileUploader"] button, [data-testid="stBaseButton-secondary"] { border-radius:10px !important; border:1px solid rgba(138,157,237,.35) !important; color:#dfe7ff !important; background:#151e35 !important; }
        [data-testid="stBaseButton-primary"] { border:0 !important; border-radius:12px !important; background:linear-gradient(100deg,var(--teal),#81b9ff) !important; color:#07101d !important; font-weight:800 !important; min-height:46px; }
        [data-testid="stMetric"] { background:rgba(14,20,36,.77); border:1px solid var(--line); border-radius:15px; padding:1rem 1.1rem; }
        [data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:.76rem !important; letter-spacing:.05em; text-transform:uppercase; }
        [data-testid="stMetricValue"] { color:var(--ink) !important; font-size:1.45rem !important; }
        [data-testid="stImage"] img { border-radius:14px; border:1px solid var(--line); }
        [data-testid="stRadio"] { background:rgba(11,16,30,.62); border:1px solid var(--line); border-radius:13px; padding:.32rem .7rem; }
        .eyebrow { color:var(--teal); font:500 .72rem 'DM Mono',monospace; letter-spacing:.14em; text-transform:uppercase; margin:0 0 .85rem; }
        .hero { display:flex; justify-content:space-between; gap:1.5rem; align-items:flex-start; padding:1.2rem 0 3.1rem; }
        .hero h1 { margin:0; color:var(--ink); font-size:clamp(2.3rem,5vw,4.7rem); line-height:.99; letter-spacing:-.075em; font-weight:800; }
        .hero h1 em { color:var(--teal); font-style:normal; }
        .hero-copy { max-width:625px; color:var(--muted); font-size:1rem; line-height:1.75; margin:1.25rem 0 0; }
        .device-chip { display:inline-flex; align-items:center; gap:.55rem; margin-top:.25rem; padding:.62rem .8rem; border:1px solid var(--line); border-radius:999px; color:#c7d2ed; background:rgba(11,17,31,.6); font:.72rem 'DM Mono',monospace; white-space:nowrap; }
        .pulse { width:7px; height:7px; border-radius:50%; background:var(--teal); box-shadow:0 0 14px var(--teal); }
        .step { color:var(--muted); font:500 .69rem 'DM Mono',monospace; letter-spacing:.1em; text-transform:uppercase; margin:0 0 .5rem; }
        .section-title { font-size:1.25rem; font-weight:700; color:var(--ink); margin:0 0 1rem; }
        .instruction-card { border:1px solid var(--line); border-radius:18px; background:linear-gradient(140deg,rgba(29,38,66,.77),rgba(14,19,34,.79)); padding:1.25rem; height:100%; box-sizing:border-box; }
        .instruction-card h3 { margin:0 0 .65rem; font-size:1rem; }
        .instruction-card p { color:var(--muted); margin:0; font-size:.88rem; line-height:1.65; }
        .mode-note { color:var(--muted); font-size:.84rem; line-height:1.65; margin:1rem 0 0; }
        .status { border:1px solid rgba(85,235,210,.28); border-radius:15px; background:rgba(32,103,103,.14); padding:.95rem 1.05rem; color:#d7fff7; margin:.65rem 0 1.1rem; }
        .soft-warning { color:#d4c9ff; font-size:.8rem; line-height:1.6; border-left:2px solid var(--violet); padding-left:.75rem; margin-top:1rem; }
        @media(max-width:780px) { .block-container{padding:1.4rem 1rem 2.5rem;} .hero{display:block;padding-bottom:2rem;} .device-chip{margin-top:1.25rem;} }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()
try:
    device, stage_a, stage_b, config = cached_models()
except Exception as error:
    st.error(f"模型加载失败：{error}")
    st.stop()

st.markdown(f"""
<div class="hero"><div><p class="eyebrow">LOCAL ALPHA MATTING · TWO-STAGE PIPELINE</p>
<h1>把<em>伪透明</em><br>变成真正透明。</h1>
<p class="hero-copy">检测生成图里的棋盘格伪透明背景，恢复连续 alpha，并在你的设备上导出 RGBA PNG。文件不会离开本机。</p>
</div><div class="device-chip"><span class="pulse"></span>{device.type.upper()} 已就绪 · Stage A + Stage B</div></div>
""", unsafe_allow_html=True)

left, right = st.columns((1.3, 0.7), gap="large")
with left:
    st.markdown('<p class="step">01 / INPUT</p><p class="section-title">放入一张需要检查的图片</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("上传图片", type=("jpg", "jpeg", "png", "webp"), label_visibility="collapsed")
with right:
    st.markdown('<p class="step">02 / INTENT</p><p class="section-title">选择处理方式</p>', unsafe_allow_html=True)
    mode = st.radio("模式", ("自动识别", "强制抠图", "保留原图"), horizontal=True, label_visibility="collapsed")
    if mode == "自动识别":
        threshold = st.slider("判定为伪透明的阈值", 0.0, 1.0, 0.5, 0.01)
        note = "Stage A 高于阈值时进入 Stage B；不确定时可改用强制抠图。"
    elif mode == "强制抠图":
        threshold, note = 0.5, "跳过分类器，直接生成连续 alpha。适合你已确认是伪透明图的场景。"
    else:
        threshold, note = 0.5, "不修改现有 alpha；适合原本就是真透明或不需要抠图的图片。"
    st.markdown(f'<p class="mode-note">{note}</p>', unsafe_allow_html=True)
    st.markdown('<p class="soft-warning">当前版本只恢复 alpha。玻璃、薄纱等半透明区域的 RGB 去污染将作为后续能力。</p>', unsafe_allow_html=True)

if uploaded is None:
    st.markdown("<br>", unsafe_allow_html=True)
    cards = st.columns(3)
    content = (("自动分流", "先由 Stage A 判断是否是伪透明背景。"), ("连续 alpha", "Stage B 预测 0–255 的透明度，而不是二值遮罩。"), ("本地导出", "预览 alpha 后下载保持原尺寸的 RGBA PNG。"))
    for column, (title, copy) in zip(cards, content):
        with column:
            st.markdown(f'<div class="instruction-card"><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)
    st.stop()

file_id = f"{uploaded.name}:{uploaded.size}"
clear_stale_result(file_id)
try:
    source = Image.open(uploaded)
    source.load()
except Exception as error:
    st.error(f"无法读取图片：{error}")
    st.stop()

rgb = source.convert("RGB")
original_alpha = source.getchannel("A") if "A" in source.getbands() else Image.new("L", source.size, 255)
st.markdown("<br>", unsafe_allow_html=True)
source_col, action_col = st.columns((1.12, 0.88), gap="large")
with source_col:
    st.markdown('<p class="step">SOURCE</p><p class="section-title">输入图像</p>', unsafe_allow_html=True)
    st.image(rgb, caption=f"{uploaded.name} · {rgb.width} × {rgb.height}", use_container_width=True)
with action_col:
    st.markdown('<p class="step">03 / PROCESS</p><p class="section-title">准备好后开始处理</p>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-card"><h3>双模型协作</h3><p>Auto 模式先显示分类概率，再按你的阈值决定是否执行 alpha matting。</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("开始生成透明 PNG", type="primary", use_container_width=True)

if run:
    with st.spinner("Stage A 正在检查伪透明特征…"):
        probability = fake_probability(rgb, stage_a, device)
    if mode == "自动识别":
        use_matting = probability >= threshold
        decision = "检测到伪透明背景，已生成连续 alpha。" if use_matting else "Auto 判定为普通图片，已保留原图 alpha。"
    elif mode == "强制抠图":
        use_matting, decision = True, "已跳过 Stage A，直接生成连续 alpha。"
    else:
        use_matting, decision = False, "已按你的选择保留原图。"
    if use_matting:
        with st.spinner("Stage B 正在恢复连续 alpha…"):
            alpha = predict_alpha(rgb, stage_b, config["image_size"], device)
    else:
        alpha = original_alpha
    st.session_state.update(result=render_rgba(rgb, alpha), alpha=alpha, probability=probability, decision=decision, threshold_used=threshold, used_matting=use_matting)

if "result" in st.session_state:
    st.markdown("<br><p class=\"step\">RESULT / REVIEW</p>", unsafe_allow_html=True)
    st.markdown(f'<div class="status">✦ {st.session_state["decision"]}</div>', unsafe_allow_html=True)
    stats = st.columns(3)
    stats[0].metric("伪透明概率", f"{st.session_state['probability']:.1%}")
    stats[1].metric("本次阈值", f"{st.session_state['threshold_used']:.0%}")
    stats[2].metric("输出模式", "连续 Alpha" if st.session_state["used_matting"] else "原始 Alpha")
    st.markdown("<br>", unsafe_allow_html=True)
    alpha_col, preview_col = st.columns(2, gap="large")
    with alpha_col:
        st.markdown('<p class="section-title">Alpha 通道</p>', unsafe_allow_html=True)
        st.image(st.session_state["alpha"], caption="白色为保留，黑色为透明", clamp=True, use_container_width=True)
    with preview_col:
        st.markdown('<p class="section-title">透明效果预览</p>', unsafe_allow_html=True)
        st.image(checkerboard_preview(st.session_state["result"]), caption="在中性棋盘格上预览导出结果", use_container_width=True)
    stem = Path(uploaded.name).stem
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("下载 RGBA PNG", png_bytes(st.session_state["result"]), file_name=f"{stem}_cutout.png", mime="image/png", type="primary", use_container_width=True)
