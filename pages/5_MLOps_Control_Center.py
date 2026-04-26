"""
MLOps Control Center — Full pipeline visualization, stage status, logs, and health scores.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

from config import CUSTOM_CSS
from utils.logging_utils import generate_pipeline_logs, read_log_tail

st.set_page_config(page_title="MLOps Control Center", page_icon="🎛️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:8px 0 12px;'>
            <div style='font-size:1.8rem;'>🎛️</div>
            <div style='color:#93c5fd; font-weight:700; font-size:0.95rem;'>Control Center</div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.8rem; color:#cbd5e1; line-height:1.8;'>
        <b style='color:#bfdbfe;'>📌 What's on this page</b><br>
        🔄 <b style='color:#e2e8f0;'>Pipeline Flow</b> — Sankey diagram showing data through all 9 stages<br>
        🃏 <b style='color:#e2e8f0;'>Stage Cards</b> — Per-stage health, status &amp; last-run time<br>
        📅 <b style='color:#e2e8f0;'>Timeline</b> — Execution duration per stage<br>
        📜 <b style='color:#e2e8f0;'>Log Viewer</b> — Simulated pipeline logs<br>
        🩺 <b style='color:#e2e8f0;'>SLA Board</b> — Compliance vs thresholds<br>
        ▶️ <b style='color:#e2e8f0;'>Simulate Run</b> — Watch the pipeline execute live
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.78rem; color:#94a3b8; line-height:1.6;'>
        <b style='color:#bfdbfe;'>🔑 Status Legend</b><br>
        <span style='color:#059669;'>■</span> <b style='color:#e2e8f0;'>Pass</b> — Stage healthy<br>
        <span style='color:#d97706;'>■</span> <b style='color:#e2e8f0;'>Warn</b> — Needs attention<br>
        <span style='color:#dc2626;'>■</span> <b style='color:#e2e8f0;'>Fail</b> — Action required<br>
        <span style='color:#6b7280;'>■</span> <b style='color:#e2e8f0;'>Pending</b> — Not yet run
        </div>
        """, unsafe_allow_html=True,
    )

st.title("🎛️ MLOps Control Center")
st.caption("End-to-end ML pipeline visibility: stages, health, logs, and execution history.")
st.markdown(
    """
    <div class='explain-box'>
    <b>📖 About this page:</b> The MLOps Control Center provides a unified view of your
    entire ML pipeline — from raw data ingestion through to production monitoring and
    retraining. Each of the <b>9 pipeline stages</b> shows its current health score,
    last execution timestamp, pass/fail status, and duration.
    The <b>Sankey flow diagram</b> shows how data flows between stages.
    Click <b>▶️ Run Full Pipeline</b> at the bottom to watch a simulated end-to-end
    execution with real-time log output.
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Pipeline Stage Definitions ────────────────────────────────────────────────
PIPELINE_STAGES = [
    {
        "id": "data_ingestion",
        "name": "Data Ingestion",
        "icon": "📥",
        "status": "Pass",
        "health": 98,
        "last_run": "2026-04-26 06:00",
        "duration_s": 0.34,
        "description": "Load raw data from source systems",
    },
    {
        "id": "data_validation",
        "name": "Data Validation",
        "icon": "✅",
        "status": "Pass",
        "health": 94,
        "last_run": "2026-04-26 06:01",
        "duration_s": 0.82,
        "description": "Schema, null, outlier, and type checks",
    },
    {
        "id": "feature_store",
        "name": "Feature Store",
        "icon": "🗄️",
        "status": "Pass",
        "health": 99,
        "last_run": "2026-04-26 06:02",
        "duration_s": 0.21,
        "description": "Feature versioning, freshness, lineage",
    },
    {
        "id": "training",
        "name": "Model Training",
        "icon": "🏋️",
        "status": "Pass",
        "health": 97,
        "last_run": "2026-04-25 23:15",
        "duration_s": 4.21,
        "description": "Algorithm training + cross-validation",
    },
    {
        "id": "evaluation",
        "name": "Evaluation",
        "icon": "📊",
        "status": "Pass",
        "health": 96,
        "last_run": "2026-04-25 23:16",
        "duration_s": 1.05,
        "description": "Test-set metrics, threshold checks",
    },
    {
        "id": "registry",
        "name": "Model Registry",
        "icon": "🗂️",
        "status": "Pass",
        "health": 100,
        "last_run": "2026-04-25 23:17",
        "duration_s": 0.15,
        "description": "Versioning, metadata, artifact storage",
    },
    {
        "id": "deployment",
        "name": "Deployment",
        "icon": "🚀",
        "status": "Pass",
        "health": 95,
        "last_run": "2026-04-25 23:20",
        "duration_s": 12.40,
        "description": "Canary → Staging → Production rollout",
    },
    {
        "id": "monitoring",
        "name": "Monitoring",
        "icon": "📡",
        "status": "Warn",
        "health": 78,
        "last_run": "2026-04-26 07:00",
        "duration_s": 0.92,
        "description": "Performance + drift + data quality checks",
    },
    {
        "id": "retraining",
        "name": "Retraining",
        "icon": "🔁",
        "status": "Pending",
        "health": 60,
        "last_run": "2026-04-24 00:00",
        "duration_s": None,
        "description": "Auto-triggered retraining jobs",
    },
]


def status_color(status):
    return {"Pass": "#059669", "Warn": "#D97706", "Fail": "#DC2626",
            "Pending": "#6B7280", "Running": "#1E40AF"}.get(status, "#6B7280")


def health_color(h):
    return "#059669" if h >= 90 else "#D97706" if h >= 70 else "#DC2626"


# ── Live Pipeline Flow ────────────────────────────────────────────────────────
st.subheader("🔄 Pipeline Flow")

# Build Sankey/flow diagram using go.Sankey
stage_names = [s["name"] for s in PIPELINE_STAGES]
n = len(PIPELINE_STAGES)
node_colors = [status_color(s["status"]) for s in PIPELINE_STAGES]

fig_flow = go.Figure(go.Sankey(
    node=dict(
        pad=20, thickness=25,
        line=dict(color="white", width=0.5),
        label=stage_names,
        color=node_colors,
        customdata=[f"{s['health']}% health" for s in PIPELINE_STAGES],
        hovertemplate="%{label}<br>%{customdata}<extra></extra>",
    ),
    link=dict(
        source=list(range(n - 1)),
        target=list(range(1, n)),
        value=[100] * (n - 1),
        color=["rgba(30,64,175,0.25)"] * (n - 1),
    ),
))
fig_flow.update_layout(
    height=220,
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=11, color="#374151"),
)
st.plotly_chart(fig_flow, use_container_width=True)

st.divider()

# ── Stage Cards ────────────────────────────────────────────────────────────────
st.subheader("📋 Stage Status Cards")
cols_per_row = 3
for row_start in range(0, len(PIPELINE_STAGES), cols_per_row):
    row_stages = PIPELINE_STAGES[row_start:row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, stage in zip(cols, row_stages):
        s_color = status_color(stage["status"])
        h_color = health_color(stage["health"])
        dur_str = f"{stage['duration_s']}s" if stage["duration_s"] else "—"
        with col:
            st.markdown(
                f"""
                <div style='border:1.5px solid {s_color}55;border-radius:12px;
                            padding:16px;background:linear-gradient(135deg,{s_color}08,white);
                            margin-bottom:12px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <span style='font-size:1.5rem;'>{stage['icon']}</span>
                        <span style='background:{s_color};color:white;font-size:0.72rem;
                                     font-weight:700;padding:2px 10px;border-radius:12px;'>
                            {stage['status']}
                        </span>
                    </div>
                    <h4 style='color:#1E40AF;margin:8px 0 4px;font-size:0.95rem;'>{stage['name']}</h4>
                    <p style='color:#6B7280;font-size:0.78rem;margin:0 0 8px;'>{stage['description']}</p>
                    <div style='background:#E5E7EB;border-radius:6px;height:8px;margin:8px 0;'>
                        <div style='width:{stage["health"]}%;background:{h_color};
                                    border-radius:6px;height:8px;'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#9CA3AF;'>
                        <span>Health: <b style='color:{h_color};'>{stage['health']}%</b></span>
                        <span>⏱ {dur_str}</span>
                    </div>
                    <div style='font-size:0.72rem;color:#9CA3AF;margin-top:4px;'>
                        Last: {stage['last_run']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()

# ── Timeline ─────────────────────────────────────────────────────────────────
st.subheader("📅 Pipeline Execution Timeline")
timeline_data = []
base_time = datetime(2026, 4, 26, 6, 0)
cumulative = 0
for s in PIPELINE_STAGES:
    dur = s["duration_s"] or 5.0
    timeline_data.append({
        "Stage":    s["name"],
        "Start":    base_time + timedelta(seconds=cumulative),
        "End":      base_time + timedelta(seconds=cumulative + dur),
        "Duration": dur,
        "Status":   s["status"],
    })
    cumulative += dur + 30  # simulate 30s gap between stages

df_tl = pd.DataFrame(timeline_data)
fig_tl = go.Figure()
for i, row in df_tl.iterrows():
    color = status_color(row["Status"])
    fig_tl.add_trace(go.Bar(
        name=row["Stage"],
        y=[row["Stage"]],
        x=[row["Duration"]],
        base=[i * 35],
        orientation="h",
        marker_color=color,
        text=f"{row['Duration']}s",
        textposition="inside",
        showlegend=False,
        hovertemplate=f"<b>{row['Stage']}</b><br>Duration: {row['Duration']}s<br>Status: {row['Status']}<extra></extra>",
    ))
fig_tl.update_layout(
    barmode="overlay",
    title="Today's Pipeline Execution (seconds)",
    height=360,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Elapsed Time (s)",
    yaxis=dict(categoryorder="array", categoryarray=[s["name"] for s in PIPELINE_STAGES][::-1]),
    showlegend=False,
)
st.plotly_chart(fig_tl, use_container_width=True)

st.divider()

# ── Log Viewer ────────────────────────────────────────────────────────────────
st.subheader("📜 Pipeline Logs")
selected_stage = st.selectbox(
    "Select stage to view logs",
    options=[s["id"] for s in PIPELINE_STAGES],
    format_func=lambda x: next(s["name"] for s in PIPELINE_STAGES if s["id"] == x),
)
logs = generate_pipeline_logs(selected_stage)
log_text = "\n".join(logs)
st.code(log_text, language="bash")

with st.expander("📄 View Application Log File (last 50 lines)"):
    app_logs = read_log_tail(50)
    if app_logs:
        st.code("\n".join(app_logs), language="bash")
    else:
        st.info("No log entries yet.")

st.divider()

# ── Health Overview ───────────────────────────────────────────────────────────
st.subheader("🩺 Pipeline Health Overview")
col_a, col_b = st.columns(2)

with col_a:
    health_scores = {s["name"]: s["health"] for s in PIPELINE_STAGES}
    fig_health = go.Figure(go.Bar(
        x=list(health_scores.values()),
        y=list(health_scores.keys()),
        orientation="h",
        marker_color=[health_color(v) for v in health_scores.values()],
        text=[f"{v}%" for v in health_scores.values()],
        textposition="outside",
    ))
    fig_health.update_layout(
        title="Stage Health Scores",
        xaxis=dict(range=[0, 115], title="Health %"),
        height=380, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_health, use_container_width=True)

with col_b:
    # SLAs
    st.markdown("**SLA Compliance**")
    sla_items = [
        ("Data Freshness",         "< 1 hour",      "✅ 0.5h",  True),
        ("Training Latency",       "< 10 min",      "✅ 4.2s",  True),
        ("Inference P95 Latency",  "< 500ms",       "✅ 142ms", True),
        ("Model Accuracy",         "> 80%",         "✅ 87.6%", True),
        ("Drift Check Frequency",  "Every 6 hours", "✅ 6h",    True),
        ("Retraining Trigger",     "PSI > 0.20",    "⚠️ 0.23",  False),
        ("Uptime",                 "99.9%",         "✅ 99.97%",True),
    ]
    for item_name, target, actual, ok in sla_items:
        color = "#059669" if ok else "#D97706"
        st.markdown(
            f"""
            <div style='display:flex;justify-content:space-between;
                        padding:6px 10px;border-bottom:1px solid #F3F4F6;
                        background:{"#F0FDF4" if ok else "#FFFBEB"};
                        border-radius:4px;margin:2px 0;'>
                <span style='font-size:0.85rem;color:#374151;'>{item_name}</span>
                <span style='font-size:0.78rem;color:#9CA3AF;'>Target: {target}</span>
                <span style='font-size:0.85rem;font-weight:700;color:{color};'>{actual}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Simulated Pipeline Run ────────────────────────────────────────────────────
st.subheader("▶️ Simulate Pipeline Execution")
if st.button("🚀 Run Full Pipeline Now", type="primary"):
    progress = st.progress(0)
    status   = st.empty()
    log_container = st.empty()
    all_logs = []

    for i, stage in enumerate(PIPELINE_STAGES):
        pct = int((i / len(PIPELINE_STAGES)) * 100)
        progress.progress(pct, text=f"Running: {stage['name']}…")
        status.info(f"⚙️ Executing **{stage['icon']} {stage['name']}** …")
        time.sleep(0.4)

        stage_logs = generate_pipeline_logs(stage["id"])
        all_logs.extend(stage_logs)
        log_container.code("\n".join(all_logs[-15:]), language="bash")

    progress.progress(100, text="✅ Pipeline complete!")
    status.success("🎉 Full pipeline executed successfully!")
