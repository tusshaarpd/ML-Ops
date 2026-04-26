"""
MLOps Showcase Platform – Main Entry Point
Run with: streamlit run app.py
"""
import streamlit as st
from config import APP_TITLE, APP_ICON, CUSTOM_CSS
from utils.logging_utils import setup_logging
from utils.model_registry import init_db

# ── Bootstrap ──────────────────────────────────────────────────────────────────
setup_logging()
init_db()

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": f"**{APP_TITLE}** — End-to-end MLOps platform built with Streamlit.",
    },
)

# Inject global CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar Branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 10px 0 20px;'>
            <span style='font-size:3rem;'>🚀</span><br>
            <span style='color:#93C5FD; font-size:1.3rem; font-weight:700;'>MLOps Platform</span><br>
            <span style='color:#64748B; font-size:0.75rem;'>DataOps Corp · v2.0.0</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<span style='color:#94A3B8; font-size:0.85rem;'>Navigate using the pages above</span>",
        unsafe_allow_html=True,
    )

# ── Landing Page ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 60px 20px 40px;'>
        <h1 style='font-size:3rem; color:#1E40AF; margin-bottom:8px;'>
            🚀 MLOps Showcase Platform
        </h1>
        <p style='font-size:1.2rem; color:#475569; max-width:700px; margin:0 auto;'>
            End-to-end machine learning lifecycle management — from raw data to
            production monitoring — across six real-world business use cases.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Feature Cards ─────────────────────────────────────────────────────────────
cols = st.columns(3)
features = [
    ("📊", "6 Business Use Cases",
     "Sentiment · Spam · Churn · Fraud · Sales Forecast · HR Attrition"),
    ("🤖", "Smart Training Engine",
     "Multi-algorithm training, CV, hyperparameter config, auto best-model selection"),
    ("🔍", "Explainable AI",
     "Feature importance, local explanations, SHAP-style charts for every prediction"),
    ("📡", "Live Monitoring",
     "Real-time performance tracking, data drift detection with PSI & KS tests"),
    ("🗂️", "Model Registry",
     "Version control, stage promotions, rollbacks, and full audit trail in SQLite"),
    ("🛡️", "Governance & Compliance",
     "Approval workflows, bias metrics, responsible AI reports, audit logs"),
]
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,#EFF6FF,#F0FDFA);
                        border:1px solid #BFDBFE; border-radius:12px;
                        padding:20px; margin-bottom:16px; height:140px;'>
                <span style='font-size:1.8rem;'>{icon}</span>
                <h4 style='color:#1E40AF; margin:8px 0 4px;'>{title}</h4>
                <p style='color:#475569; font-size:0.85rem; margin:0;'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<p style='text-align:center; color:#9CA3AF; font-size:0.85rem;'>"
    "👈 Use the sidebar to navigate between platform sections"
    "</p>",
    unsafe_allow_html=True,
)
