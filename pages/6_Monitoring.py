"""
Monitoring — Performance tracking, data drift detection, data quality monitoring.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import CUSTOM_CSS, USE_CASES, PSI_MEDIUM_THRESHOLD
from utils.data_generator import get_dataset
from utils.drift_detection import (
    analyze_drift, simulate_drift_batch, generate_psi_history, psi_label, psi_color
)
from utils.metrics import simulate_metric_history, simulate_regression_history
from utils.visualizations import (
    metric_trend_chart, psi_heatmap, distribution_comparison_chart,
    gauge_chart, PALETTE
)

st.set_page_config(page_title="Monitoring", page_icon="📡", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📡 Monitoring Dashboard")
st.caption("Real-time performance tracking, drift detection, and data quality monitoring.")
st.divider()

# ── Sidebar Filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Monitoring Config")
    mon_uc = st.selectbox(
        "Use Case / Model",
        list(USE_CASES.keys()),
        format_func=lambda k: f"{USE_CASES[k]['icon']} {USE_CASES[k]['name']}",
    )
    perf_days = st.slider("Performance History (days)", 7, 90, 30)
    st.divider()
    st.markdown("**Alert Thresholds**")
    acc_threshold = st.slider("Min Accuracy / R²", 0.5, 0.99, 0.80)
    psi_threshold = st.slider("Max PSI Drift", 0.05, 0.50, 0.20)
    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

# ── Performance Metrics ────────────────────────────────────────────────────────
st.subheader("📈 Performance Monitoring")
task = USE_CASES[mon_uc]["task"]

if task == "classification":
    base_acc = {"sentiment": 0.892, "spam": 0.967, "churn": 0.835,
                "fraud": 0.954, "hr": 0.882}.get(mon_uc, 0.88)
    hist = simulate_metric_history(base_acc=base_acc, days=perf_days, seed=hash(mon_uc) % 1000)

    m1, m2, m3, m4, m5 = st.columns(5)
    latest = hist.iloc[-1]
    prev   = hist.iloc[-2]

    m1.metric("Accuracy",  f"{latest.accuracy:.3f}", f"{latest.accuracy - prev.accuracy:+.3f}")
    m2.metric("F1 Score",  f"{latest.f1:.3f}",       f"{latest.f1 - prev.f1:+.3f}")
    m3.metric("Precision", f"{latest.precision:.3f}", f"{latest.precision - prev.precision:+.3f}")
    m4.metric("Recall",    f"{latest.recall:.3f}",    f"{latest.recall - prev.recall:+.3f}")

    alert_triggered = latest.accuracy < acc_threshold
    m5.metric("Alert Status",
              "🔴 TRIGGERED" if alert_triggered else "🟢 OK",
              delta=None)

    if alert_triggered:
        st.markdown(
            f"<div class='alert-danger'>🚨 Accuracy ({latest.accuracy:.3f}) is below threshold "
            f"({acc_threshold:.2f}). Consider retraining.</div>",
            unsafe_allow_html=True,
        )

    fig = metric_trend_chart(hist, ["accuracy", "f1", "precision", "recall"],
                              f"Performance Trend — {USE_CASES[mon_uc]['name']}")
    st.plotly_chart(fig, use_container_width=True)

else:
    hist = simulate_regression_history(base_rmse=42.0, days=perf_days, seed=hash(mon_uc) % 1000)
    latest = hist.iloc[-1]
    prev   = hist.iloc[-2]

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{latest.rmse:.2f}", f"{latest.rmse - prev.rmse:+.2f}")
    m2.metric("MAE",  f"{latest.mae:.2f}",  f"{latest.mae - prev.mae:+.2f}")
    m3.metric("Trend", "⬆️ Degrading" if latest.rmse > prev.rmse else "⬇️ Improving")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["rmse"],
        mode="lines+markers", name="RMSE",
        line=dict(color="#1E40AF", width=2),
        marker=dict(size=4),
    ))
    fig.add_hline(y=acc_threshold * 100, line_dash="dash", line_color="#DC2626",
                  annotation_text=f"Threshold={acc_threshold*100:.0f}")
    fig.update_layout(title="RMSE Trend",
                      height=360, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Drift Detection ────────────────────────────────────────────────────────────
st.subheader("🌊 Data Drift Detection")

@st.cache_data(show_spinner="Computing drift statistics…")
def compute_drift(uc):
    df = get_dataset(uc)
    # Reference = first 70%, current = last 30% + perturbation
    split = int(len(df) * 0.7)
    ref_df = df.iloc[:split].copy()
    cur_df = simulate_drift_batch(df.iloc[split:].copy(), drift_fraction=0.4, seed=42)
    drift_summary = analyze_drift(ref_df, cur_df)
    return ref_df, cur_df, drift_summary

ref_df, cur_df, drift_summary = compute_drift(mon_uc)

# Drift summary table
if not drift_summary.empty:
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.markdown("**Feature Drift Summary (PSI)**")
        styled_rows = []
        for _, row in drift_summary.iterrows():
            color = psi_color(row["PSI"])
            styled_rows.append({
                "Feature":    row["Feature"],
                "PSI":        f"{row['PSI']:.4f}",
                "KS Stat":    f"{row['KS Statistic']:.4f}",
                "KS p-val":   f"{row['KS p-value']:.4f}",
                "Ref Mean":   f"{row['Ref Mean']:.3f}",
                "Cur Mean":   f"{row['Cur Mean']:.3f}",
                "Shift %":    f"{row['Rel Shift %']:.1f}%",
                "Status":     row["Status"],
            })
        drift_disp = pd.DataFrame(styled_rows)
        st.dataframe(drift_disp, use_container_width=True, hide_index=True)

        high_drift = drift_summary[drift_summary["PSI"] > psi_threshold]
        if not high_drift.empty:
            st.markdown(
                f"<div class='alert-danger'>🚨 {len(high_drift)} feature(s) exceed PSI threshold "
                f"({psi_threshold:.2f}): "
                f"{', '.join(high_drift['Feature'].tolist())}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='alert-success'>✅ All features within drift threshold</div>",
                unsafe_allow_html=True,
            )

    with col_b:
        # PSI bar chart
        top_drift = drift_summary.head(10)
        fig = go.Figure(go.Bar(
            y=top_drift["Feature"][::-1].tolist(),
            x=top_drift["PSI"][::-1].tolist(),
            orientation="h",
            marker_color=[psi_color(p) for p in top_drift["PSI"][::-1]],
            text=[f"{v:.3f}" for v in top_drift["PSI"][::-1]],
            textposition="outside",
        ))
        fig.add_vline(x=PSI_MEDIUM_THRESHOLD, line_dash="dash", line_color="#DC2626",
                      annotation_text="Drift threshold")
        fig.update_layout(
            title="PSI by Feature",
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="PSI Score",
        )
        st.plotly_chart(fig, use_container_width=True)

# PSI History Heatmap
num_features = drift_summary["Feature"].tolist()[:8] if not drift_summary.empty else []
if num_features:
    st.markdown("**PSI Drift History (Last 30 Days)**")
    psi_hist = generate_psi_history(num_features[:6], days=30, seed=hash(mon_uc) % 1000)
    fig = psi_heatmap(psi_hist, f"PSI Drift Heatmap — {USE_CASES[mon_uc]['name']}")
    st.plotly_chart(fig, use_container_width=True)

# Distribution comparison for a selected feature
st.markdown("**Distribution Comparison — Reference vs Current**")
num_cols = ref_df.select_dtypes(include="number").columns.tolist()
if num_cols:
    sel_feat = st.selectbox("Select feature to compare", num_cols)
    fig = distribution_comparison_chart(
        ref_df[sel_feat], cur_df[sel_feat], sel_feat,
        f"Distribution Shift — {sel_feat}",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric features to compare for this use case.")

st.divider()

# ── Data Quality Monitoring ───────────────────────────────────────────────────
st.subheader("🔍 Data Quality Monitoring")

@st.cache_data(show_spinner="Computing quality metrics…")
def compute_quality_history(uc, days=30, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    base_missing   = rng.uniform(0.5, 2.5)
    base_dup       = rng.uniform(0.1, 1.5)
    base_outliers  = rng.uniform(0.5, 3.0)
    base_schema    = rng.integers(0, 2, days)

    return pd.DataFrame({
        "date":           dates,
        "missing_pct":    np.clip(base_missing + rng.normal(0, 0.3, days), 0, 10).round(2),
        "duplicate_pct":  np.clip(base_dup     + rng.normal(0, 0.2, days), 0, 5).round(2),
        "outlier_pct":    np.clip(base_outliers + rng.normal(0, 0.4, days), 0, 10).round(2),
        "schema_issues":  base_schema,
    })

dq_hist = compute_quality_history(mon_uc, days=perf_days, seed=hash(mon_uc) % 1000)
latest_dq = dq_hist.iloc[-1]

dq_cols = st.columns(4)
dq_cols[0].metric("Missing Data %",    f"{latest_dq.missing_pct:.2f}%",
                   delta=None, delta_color="inverse")
dq_cols[1].metric("Duplicate Rows %",  f"{latest_dq.duplicate_pct:.2f}%",
                   delta=None, delta_color="inverse")
dq_cols[2].metric("Outlier %",         f"{latest_dq.outlier_pct:.2f}%",
                   delta=None, delta_color="inverse")
dq_cols[3].metric("Schema Violations", int(latest_dq.schema_issues))

# Quality trend chart
fig_dq = go.Figure()
for col, color in [("missing_pct","#1E40AF"), ("duplicate_pct","#D97706"),
                    ("outlier_pct","#DC2626")]:
    fig_dq.add_trace(go.Scatter(
        x=dq_hist["date"], y=dq_hist[col],
        mode="lines", name=col.replace("_pct"," %").title(),
        line=dict(color=color, width=2),
    ))
fig_dq.update_layout(
    title="Data Quality Trend",
    yaxis_title="Percentage (%)",
    height=320, paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_dq, use_container_width=True)

# Quality thresholds
st.markdown("**Quality Thresholds**")
thresholds = [
    ("Missing Data", f"{latest_dq.missing_pct:.2f}%", "< 5%",
     latest_dq.missing_pct < 5),
    ("Duplicate Rows", f"{latest_dq.duplicate_pct:.2f}%", "< 2%",
     latest_dq.duplicate_pct < 2),
    ("Outliers", f"{latest_dq.outlier_pct:.2f}%", "< 5%",
     latest_dq.outlier_pct < 5),
    ("Schema Issues", str(int(latest_dq.schema_issues)), "= 0",
     latest_dq.schema_issues == 0),
]
t_cols = st.columns(4)
for col, (metric, actual, target, ok) in zip(t_cols, thresholds):
    with col:
        color = "#059669" if ok else "#DC2626"
        icon  = "✅" if ok else "❌"
        st.markdown(
            f"""
            <div style='border:1px solid {color}55; border-radius:8px;
                        padding:12px; background:{color}08; text-align:center;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='font-weight:700;font-size:0.9rem;color:#374151;'>{metric}</div>
                <div style='font-size:1.2rem;font-weight:700;color:{color};'>{actual}</div>
                <div style='font-size:0.75rem;color:#9CA3AF;'>Target: {target}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Retraining Recommendation ─────────────────────────────────────────────────
st.subheader("🔁 Retraining Recommendation Engine")

drift_triggered  = not drift_summary.empty and (drift_summary["PSI"] > psi_threshold).any()
perf_triggered   = task == "classification" and latest.accuracy < acc_threshold
quality_triggered = latest_dq.missing_pct > 5

trigger_reasons = []
if drift_triggered:
    trigger_reasons.append("Data drift PSI exceeds threshold")
if perf_triggered:
    trigger_reasons.append("Model accuracy dropped below threshold")
if quality_triggered:
    trigger_reasons.append("Data quality issue: high missing values")

if trigger_reasons:
    st.markdown(
        f"""
        <div class='alert-danger'>
            <b>🔁 Retraining Recommended!</b><br>
            Triggers: {', '.join(trigger_reasons)}<br>
            <i>Estimated effort: ~5 minutes | Estimated accuracy gain: +1.5-3%</i>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("📋 Create Retraining Request", type="primary"):
        st.success("✅ Retraining request submitted. Awaiting ML Engineer approval (job_id=RT-0043).")
else:
    st.markdown(
        "<div class='alert-success'>✅ No retraining needed. All metrics within healthy range.</div>",
        unsafe_allow_html=True,
    )
