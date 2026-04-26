"""
Executive Dashboard — KPI overview, active alerts, recent activity, health scores.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import CUSTOM_CSS
from utils.model_registry import get_all_models, get_production_models, get_audit_log
from utils.metrics import simulate_metric_history
from utils.visualizations import metric_trend_chart, gauge_chart, sparkline, PALETTE

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar explanation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:10px 0 14px;'>
            <div style='font-size:2rem;'>📊</div>
            <div style='color:#93c5fd; font-weight:700; font-size:1rem;'>Executive Dashboard</div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.82rem; color:#cbd5e1; line-height:1.8;'>
        <b style='color:#bfdbfe;'>📌 What you see here</b><br>
        • Live KPI cards for all production models<br>
        • Active alerts & drift incidents<br>
        • 30-day accuracy / F1 trend chart<br>
        • Use-case coverage matrix<br>
        • Recent audit log events
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.78rem; color:#94a3b8; line-height:1.6;'>
        <b style='color:#bfdbfe;'>💡 Key metrics explained</b><br>
        <b style='color:#e2e8f0;'>Avg Accuracy</b> — mean accuracy of all Production-stage models<br>
        <b style='color:#e2e8f0;'>Drift Incidents</b> — features with PSI &gt; 0.20 this week<br>
        <b style='color:#e2e8f0;'>Retrain Pending</b> — models awaiting retraining approval
        </div>
        """, unsafe_allow_html=True,
    )

st.title("📊 Executive Dashboard")
st.caption("Real-time platform health, model KPIs, and operational alerts — refreshes every 30 seconds.")

st.markdown(
    """
    <div class='explain-box'>
    <b>📖 About this page:</b> The Executive Dashboard gives a bird's-eye view of the entire
    MLOps platform. It aggregates key performance indicators from all production models,
    highlights active operational alerts, and shows a 30-day performance trend.
    Use this page to quickly spot degrading models, drift events, or pending actions
    without diving into individual use-case pages.
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_registry():
    return get_all_models()

@st.cache_data(ttl=30)
def load_prod():
    return get_production_models()

@st.cache_data(ttl=30)
def load_audit():
    return get_audit_log()

registry = load_registry()
prod     = load_prod()
audit    = load_audit()

# ── KPI Metrics ───────────────────────────────────────────────────────────────
total_models   = len(registry)
prod_models    = len(prod)
staging_models = len(registry[registry["stage"] == "Staging"])
avg_accuracy   = prod["accuracy"].dropna().mean() if not prod.empty else 0
drift_alerts   = 3   # simulated
retrain_pend   = 2   # simulated
active_alerts  = 4   # simulated

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
kpis = [
    (c1, "🗂️ Total Models",     total_models,             None,  "normal"),
    (c2, "✅ In Production",     prod_models,              None,  "normal"),
    (c3, "🔬 In Staging",        staging_models,           None,  "normal"),
    (c4, "🎯 Avg Accuracy",      f"{avg_accuracy:.1%}",    None,  "normal"),
    (c5, "🚨 Active Alerts",     active_alerts,            None,  "inverse"),
    (c6, "🌊 Drift Incidents",   drift_alerts,             None,  "inverse"),
    (c7, "🔁 Retrain Pending",   retrain_pend,             None,  "inverse"),
]
for col, label, value, delta, delta_color in kpis:
    with col:
        st.metric(label, value, delta=delta, delta_color=delta_color)

st.divider()

# ── Performance Trend + Health Gauges ────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.subheader("📈 Platform Performance Trend (Last 30 Days)")
    hist = simulate_metric_history(base_acc=0.88, days=30, seed=42)
    fig  = metric_trend_chart(hist, ["accuracy", "f1", "precision", "recall"],
                               title="Aggregate Model Performance")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🩺 System Health")
    for model_name, acc, color in [
        ("Sentiment", 0.912, "#059669"),
        ("Spam Filter", 0.967, "#059669"),
        ("Churn Model", 0.869, "#059669"),
        ("Fraud Detector", 0.954, "#059669"),
        ("Sales Forecaster", None, None),
        ("HR Attrition", 0.882, "#059669"),
    ]:
        score = acc if acc else 0.964
        bar_color = "#059669" if score > 0.85 else "#D97706" if score > 0.75 else "#DC2626"
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; margin-bottom:8px;'>
                <span style='width:140px; font-size:0.85rem; color:#374151;'>{model_name}</span>
                <div style='flex:1; background:#E5E7EB; border-radius:6px; height:10px; margin:0 8px;'>
                    <div style='width:{score*100:.0f}%; background:{bar_color};
                                border-radius:6px; height:10px;'></div>
                </div>
                <span style='font-size:0.8rem; font-weight:700; color:{bar_color};
                             width:48px; text-align:right;'>{score:.1%}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='alert-success'><b>✅ All production models healthy</b></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Alerts ────────────────────────────────────────────────────────────────────
st.subheader("🚨 Active Alerts")
alerts = [
    ("HIGH",   "Drift Detected",    "ChurnPredictor v2.0",
     "PSI=0.23 on 'usage_drop_pct' — exceeds threshold 0.20",
     "2026-04-25 14:32"),
    ("MEDIUM", "Accuracy Drop",     "SentimentClassifier v1.0",
     "Accuracy fell from 0.892 → 0.871 (Δ-2.4%) over 7 days",
     "2026-04-24 09:15"),
    ("MEDIUM", "Data Quality Issue","FraudDetector v1.0",
     "Missing values in 'device_mismatch' column: 4.2% (threshold 3%)",
     "2026-04-23 22:47"),
    ("LOW",    "Retraining Due",    "SalesForecaster v1.0",
     "Scheduled monthly retraining window opens in 3 days",
     "2026-04-23 00:00"),
]
alert_colors = {"HIGH": "alert-danger", "MEDIUM": "alert-warning", "LOW": "alert-info"}
alert_icons  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}

for sev, atype, model, detail, ts in alerts:
    css = alert_colors[sev]
    icon = alert_icons[sev]
    st.markdown(
        f"""
        <div class='{css}' style='margin-bottom:8px;'>
            {icon} <b>[{sev}] {atype}</b> — <em>{model}</em><br>
            <span style='font-size:0.88rem;'>{detail}</span>
            <span style='float:right; font-size:0.78rem; color:#9CA3AF;'>{ts}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Production Models Table ───────────────────────────────────────────────────
st.subheader("🏭 Production Models")
if not prod.empty:
    display_cols = ["model_name", "version", "use_case", "algo_key",
                    "accuracy", "f1", "owner", "updated_at"]
    disp = prod[[c for c in display_cols if c in prod.columns]].copy()
    if "accuracy" in disp.columns:
        disp["accuracy"] = disp["accuracy"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "f1" in disp.columns:
        disp["f1"] = disp["f1"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )
    st.dataframe(disp, use_container_width=True, hide_index=True)
else:
    st.info("No production models yet. Register models from the Training page.")

st.divider()

# ── Recent Activity ───────────────────────────────────────────────────────────
st.subheader("📋 Recent Activity")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Audit Log (Latest)**")
    if not audit.empty:
        disp_audit = audit.head(8)[["timestamp", "action", "model_name", "version",
                                     "to_stage", "actor"]].copy()
        disp_audit["timestamp"] = disp_audit["timestamp"].str[:19]
        st.dataframe(disp_audit, use_container_width=True, hide_index=True)
    else:
        st.info("No audit events yet.")

with col_b:
    st.markdown("**Model Distribution by Stage**")
    if not registry.empty:
        stage_counts = registry["stage"].value_counts()
        fig = go.Figure(go.Pie(
            labels=stage_counts.index.tolist(),
            values=stage_counts.values.tolist(),
            hole=0.45,
            marker=dict(colors=["#059669", "#D97706", "#1E40AF", "#DC2626"]),
            textinfo="percent+label",
        ))
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Use Case Coverage ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🎯 Use Case Coverage")
use_case_stats = [
    ("💬 Sentiment",  "Production", "LR v1.0",  "92.0%", "Stable",   "#059669"),
    ("📧 Spam",       "Production", "NB v1.0",  "96.7%", "Stable",   "#059669"),
    ("📉 Churn",      "Production", "RF v1.0",  "83.5%", "Moderate", "#D97706"),
    ("🔍 Fraud",      "Production", "RF v1.0",  "95.4%", "Stable",   "#059669"),
    ("📈 Sales",      "Production", "RFR v1.0", "R²=0.96","Stable",  "#059669"),
    ("👥 HR",         "Production", "GB v1.0",  "88.2%", "Stable",   "#059669"),
]
cols = st.columns(6)
for col, (name, stage, model, acc, drift, dcolor) in zip(cols, use_case_stats):
    with col:
        st.markdown(
            f"""
            <div style='border:1px solid #E5E7EB; border-radius:10px;
                        padding:14px; text-align:center; background:#FAFAFA;'>
                <div style='font-size:1.4rem;'>{name.split()[0]}</div>
                <div style='font-weight:700; font-size:0.9rem; color:#1E40AF;
                            margin:4px 0;'>{name.split(maxsplit=1)[1]}</div>
                <div style='font-size:0.78rem; color:#059669;'>{stage}</div>
                <div style='font-size:0.85rem; font-weight:600; color:#374151;
                            margin:4px 0;'>{acc}</div>
                <div style='font-size:0.78rem; color:{dcolor};'>Drift: {drift}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
