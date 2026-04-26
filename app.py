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
    menu_items={"About": f"**{APP_TITLE}** — End-to-end MLOps platform built with Streamlit."},
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:16px 0 20px;'>
            <div style='font-size:2.8rem; line-height:1;'>🚀</div>
            <div style='color:#93c5fd; font-size:1.25rem; font-weight:800;
                        letter-spacing:0.5px; margin-top:8px;'>MLOps Platform</div>
            <div style='color:#94a3b8; font-size:0.72rem; margin-top:4px;'>
                DataOps Corp &nbsp;·&nbsp; v2.0.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style='color:#bfdbfe; font-size:0.82rem; font-weight:700;
                    letter-spacing:0.5px; margin-bottom:6px;'>📌 NAVIGATION</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='font-size:0.82rem; color:#cbd5e1; line-height:2;'>
        🏠 <b style='color:#e2e8f0;'>Home</b> — KPI dashboard & alerts<br>
        🎯 <b style='color:#e2e8f0;'>Use Cases</b> — Dataset exploration<br>
        🤖 <b style='color:#e2e8f0;'>Training</b> — Train & compare models<br>
        🔮 <b style='color:#e2e8f0;'>Predictions</b> — Live inference & API<br>
        🎛️ <b style='color:#e2e8f0;'>Control Center</b> — Pipeline health<br>
        📡 <b style='color:#e2e8f0;'>Monitoring</b> — Drift & performance<br>
        🗂️ <b style='color:#e2e8f0;'>Registry</b> — Version management<br>
        🛡️ <b style='color:#e2e8f0;'>Governance</b> — Compliance & bias
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.78rem; color:#94a3b8; line-height:1.7;'>
        <b style='color:#bfdbfe;'>💡 How to use</b><br>
        1. Start at <b style='color:#e2e8f0;'>Use Cases</b> to explore data<br>
        2. Go to <b style='color:#e2e8f0;'>Training</b> to train a model<br>
        3. Try <b style='color:#e2e8f0;'>Predictions</b> for live inference<br>
        4. Monitor in <b style='color:#e2e8f0;'>Monitoring</b> &amp; <b style='color:#e2e8f0;'>Registry</b><br>
        5. Review <b style='color:#e2e8f0;'>Governance</b> for compliance
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.75rem; color:#64748b; text-align:center;'>
        All data is <b style='color:#94a3b8;'>synthetic</b><br>
        Reproducible seed: <code style='background:rgba(255,255,255,0.1);
        color:#93c5fd; padding:1px 5px; border-radius:3px;'>42</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding:50px 20px 30px;'>
        <div style='font-size:3.2rem; margin-bottom:10px;'>🚀</div>
        <h1 style='font-size:2.6rem; color:#1E40AF; margin-bottom:10px;'>
            MLOps Showcase Platform
        </h1>
        <p style='font-size:1.1rem; color:#475569; max-width:680px;
                  margin:0 auto 20px; line-height:1.7;'>
            A <b>production-grade, end-to-end MLOps platform</b> demonstrating the complete
            machine learning lifecycle — data ingestion, training, evaluation, versioning,
            deployment simulation, drift detection, and governance — across
            <b>six real-world business use cases</b>.
        </p>
        <div style='display:inline-block; background:#EFF6FF; border:1px solid #BFDBFE;
                    border-radius:20px; padding:6px 20px; font-size:0.85rem; color:#1E40AF;
                    font-weight:600;'>
            Built with Python · Streamlit · scikit-learn · Plotly · SQLite
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── What This Platform Covers ─────────────────────────────────────────────────
st.markdown(
    """
    <div class='explain-box'>
    <b>📖 What is MLOps?</b> MLOps (Machine Learning Operations) is the practice of
    streamlining and automating the machine learning lifecycle in production. It covers
    everything from <b>data validation</b> and <b>model training</b> to <b>deployment</b>,
    <b>monitoring</b>, and <b>retraining</b>. This platform demonstrates all of these
    stages in one unified interface using realistic synthetic datasets and real ML algorithms.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Feature Cards ─────────────────────────────────────────────────────────────
st.subheader("🧩 Platform Capabilities")
features = [
    ("🎯", "6 Business Use Cases",
     "Sentiment Analysis · Spam Detection · Customer Churn · Fraud Detection · Sales Forecasting · HR Attrition",
     "#EFF6FF", "#BFDBFE"),
    ("🤖", "Multi-Algorithm Training",
     "Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes, XGBoost, Isolation Forest with cross-validation",
     "#F0FDFA", "#99F6E4"),
    ("💡", "Explainable AI (XAI)",
     "Global feature importance, local token attributions for text models, probability confidence distributions",
     "#FFF7ED", "#FED7AA"),
    ("📡", "Drift & Performance Monitoring",
     "PSI (Population Stability Index) + KS test for data drift; accuracy/F1 trend charts over 30 days",
     "#FDF4FF", "#E9D5FF"),
    ("🗂️", "Model Registry",
     "SQLite-backed versioning with Development → Staging → Production → Archived lifecycle, rollback, and audit trail",
     "#F0FDF4", "#BBF7D0"),
    ("🛡️", "Governance & Compliance",
     "Multi-level approval workflows, bias/fairness metrics (Demographic Parity, Equal Opportunity), model cards",
     "#FEF2F2", "#FECACA"),
]
row1 = st.columns(3)
row2 = st.columns(3)
for i, (icon, title, desc, bg, border) in enumerate(features):
    col = (row1 if i < 3 else row2)[i % 3]
    with col:
        st.markdown(
            f"""
            <div style='background:{bg}; border:1.5px solid {border};
                        border-radius:12px; padding:18px 16px; margin-bottom:12px; min-height:130px;'>
                <div style='font-size:1.6rem; margin-bottom:6px;'>{icon}</div>
                <div style='font-weight:700; font-size:0.95rem; color:#1E40AF;
                            margin-bottom:5px;'>{title}</div>
                <div style='font-size:0.82rem; color:#475569; line-height:1.5;'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── ML Lifecycle Flow ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🔄 ML Lifecycle at a Glance")
stages = [
    ("📥", "Data\nIngestion", "#DBEAFE"),
    ("✅", "Data\nValidation", "#D1FAE5"),
    ("🔧", "Feature\nEngineering", "#FEF3C7"),
    ("🏋️", "Model\nTraining", "#EDE9FE"),
    ("📊", "Evaluation\n& Registry", "#FCE7F3"),
    ("🚀", "Deployment\n& Serving", "#DCFCE7"),
    ("📡", "Monitoring\n& Drift", "#FEF9C3"),
    ("🔁", "Retraining\nTrigger", "#FFEDD5"),
]
cols = st.columns(len(stages))
for col, (icon, label, bg) in zip(cols, stages):
    with col:
        st.markdown(
            f"""
            <div style='background:{bg}; border-radius:10px; padding:12px 6px;
                        text-align:center; font-size:0.78rem; font-weight:600;
                        color:#374151; line-height:1.5;'>
                <div style='font-size:1.4rem;'>{icon}</div>
                {label}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Use Cases Quick View ──────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Business Use Cases Overview")
use_cases = [
    ("💬", "Sentiment Analysis",   "Text → Positive/Neutral/Negative",   "TF-IDF + LR / NB",     "Classification"),
    ("📧", "Spam Detection",        "Message → Ham/Spam",                 "TF-IDF + Naive Bayes",  "Classification"),
    ("📉", "Customer Churn",        "Telecom customer → Churn/Retained",  "Random Forest / GB",    "Classification"),
    ("🔍", "Fraud Detection",       "Transaction → Fraud/Legit",          "RF / Isolation Forest", "Classification"),
    ("📈", "Sales Forecasting",     "Date features → Daily sales volume", "RF Regressor / Linear", "Regression"),
    ("👥", "HR Attrition",          "Employee → Leave/Stay",              "Gradient Boosting",     "Classification"),
]
uc_cols = st.columns(3)
for i, (icon, name, desc, algo, task) in enumerate(use_cases):
    with uc_cols[i % 3]:
        task_color = "#1E40AF" if task == "Classification" else "#059669"
        st.markdown(
            f"""
            <div style='border:1px solid #E5E7EB; border-radius:10px; padding:14px;
                        margin-bottom:10px; background:#FAFAFA;'>
                <div style='font-size:1.3rem;'>{icon}
                    <span style='font-size:0.78rem; background:{task_color}22;
                                 color:{task_color}; padding:1px 8px; border-radius:8px;
                                 font-weight:600; margin-left:6px;'>{task}</span>
                </div>
                <div style='font-weight:700; color:#1E40AF; margin:6px 0 3px;
                            font-size:0.92rem;'>{name}</div>
                <div style='font-size:0.8rem; color:#6B7280;'>{desc}</div>
                <div style='font-size:0.78rem; color:#9CA3AF; margin-top:4px;'>
                    🤖 {algo}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    """
    <p style='text-align:center; color:#9CA3AF; font-size:0.85rem;'>
    👈 <b>Select a page from the sidebar</b> to start exploring the platform.
    &nbsp;|&nbsp; All ML models train in seconds on synthetic data with seed=42 for reproducibility.
    </p>
    """,
    unsafe_allow_html=True,
)
