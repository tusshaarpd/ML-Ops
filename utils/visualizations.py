"""
Reusable Plotly chart functions used across all pages.
Every function returns a plotly Figure object ready for st.plotly_chart().
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette ─────────────────────────────────────────────────────────────
PRIMARY  = "#1E40AF"
SUCCESS  = "#059669"
WARNING  = "#D97706"
DANGER   = "#DC2626"
INFO     = "#0891B2"
NEUTRAL  = "#6B7280"
PALETTE  = [PRIMARY, INFO, SUCCESS, WARNING, DANGER, NEUTRAL,
            "#7C3AED", "#DB2777", "#D97706", "#0D9488"]


def _base_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=PRIMARY)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        height=height,
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#E5E7EB", borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6", linecolor="#E5E7EB")
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", linecolor="#E5E7EB")
    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def confusion_matrix_chart(cm: list, labels: list, title: str = "Confusion Matrix") -> go.Figure:
    cm_arr = np.array(cm)
    text   = [[str(v) for v in row] for row in cm_arr]

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#EFF6FF"], [0.5, INFO], [1, PRIMARY]],
        showscale=True,
        colorbar=dict(thickness=12, len=0.8),
    ))
    return _base_layout(fig, title, height=380)


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def roc_curve_chart(fpr: list, tpr: list, auc: float, title: str = "ROC Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC (AUC={auc:.3f})",
        line=dict(color=PRIMARY, width=2.5),
        fill="tozeroy", fillcolor="rgba(30,64,175,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color=NEUTRAL, dash="dash", width=1.5),
    ))
    fig.update_xaxes(title_text="False Positive Rate", range=[0, 1])
    fig.update_yaxes(title_text="True Positive Rate", range=[0, 1])
    return _base_layout(fig, title, height=380)


# ── Precision-Recall Curve ────────────────────────────────────────────────────

def pr_curve_chart(precision: list, recall: list, title: str = "Precision-Recall Curve") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=recall, y=precision, mode="lines",
        line=dict(color=SUCCESS, width=2.5),
        fill="tozeroy", fillcolor="rgba(5,150,105,0.08)",
    ))
    fig.update_xaxes(title_text="Recall", range=[0, 1])
    fig.update_yaxes(title_text="Precision", range=[0, 1])
    return _base_layout(fig, title, height=350)


# ── Feature Importance ────────────────────────────────────────────────────────

def feature_importance_chart(feat_imp: dict, title: str = "Feature Importance", top_n: int = 15) -> go.Figure:
    sorted_fi = sorted(feat_imp.items(), key=lambda x: x[1], reverse=False)[-top_n:]
    names  = [k for k, _ in sorted_fi]
    values = [v for _, v in sorted_fi]

    colors = [PRIMARY if v >= max(values) * 0.7 else INFO if v >= max(values) * 0.4 else SUCCESS
              for v in values]

    fig = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_xaxes(title_text="Importance Score")
    return _base_layout(fig, title, height=max(300, 25 * len(names) + 80))


# ── Metric Trend ──────────────────────────────────────────────────────────────

def metric_trend_chart(df: pd.DataFrame, metrics: list, title: str = "Metric Trend") -> go.Figure:
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[metric], mode="lines+markers",
            name=metric.upper(),
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(size=4),
        ))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Score", range=[0.5, 1.0])
    return _base_layout(fig, title, height=380)


# ── Distribution Comparison ───────────────────────────────────────────────────

def distribution_comparison_chart(
    ref: pd.Series, cur: pd.Series,
    feature_name: str,
    title: str = "Distribution Comparison"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ref.dropna(), name="Reference", opacity=0.65,
        marker_color=PRIMARY, nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=cur.dropna(), name="Current Batch", opacity=0.65,
        marker_color=DANGER, nbinsx=30,
    ))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text=feature_name)
    fig.update_yaxes(title_text="Count")
    return _base_layout(fig, title, height=350)


# ── PSI Heatmap ───────────────────────────────────────────────────────────────

def psi_heatmap(drift_df: pd.DataFrame, title: str = "PSI Drift Heatmap") -> go.Figure:
    pivot = drift_df.pivot(index="feature", columns="date", values="psi")
    fig   = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(d.date()) for d in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#D1FAE5"], [0.1, "#FEF3C7"], [0.2, "#FDE68A"],
                    [0.4, "#F97316"], [1, "#DC2626"]],
        zmin=0, zmax=0.4,
        text=np.round(pivot.values, 3),
        texttemplate="%{text}",
        colorbar=dict(title="PSI", thickness=14),
    ))
    return _base_layout(fig, title, height=max(280, 35 * len(pivot) + 80))


# ── Class Distribution ────────────────────────────────────────────────────────

def class_distribution_chart(series: pd.Series, title: str = "Class Distribution") -> go.Figure:
    counts = series.value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=PALETTE[:len(counts)],
        text=counts.values.tolist(),
        textposition="outside",
    ))
    fig.update_xaxes(title_text="Class")
    fig.update_yaxes(title_text="Count")
    return _base_layout(fig, title, height=330)


# ── Pie Chart ─────────────────────────────────────────────────────────────────

def pie_chart(labels: list, values: list, title: str = "") -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.40,
        marker=dict(colors=PALETTE[:len(labels)]),
        textinfo="percent+label",
    ))
    return _base_layout(fig, title, height=340)


# ── Time Series with Forecast ─────────────────────────────────────────────────

def forecast_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "sales",
    title: str = "Sales Forecast",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df[date_col], y=history_df[value_col],
        mode="lines", name="Historical",
        line=dict(color=PRIMARY, width=2),
    ))
    if "lower" in forecast_df.columns and "upper" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df[date_col], forecast_df[date_col][::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(8,145,178,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
        ))
    fig.add_trace(go.Scatter(
        x=forecast_df[date_col], y=forecast_df[value_col],
        mode="lines+markers", name="Forecast",
        line=dict(color=WARNING, width=2.5, dash="dash"),
        marker=dict(size=4),
    ))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales")
    return _base_layout(fig, title, height=420)


# ── Gauge Chart ───────────────────────────────────────────────────────────────

def gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 1) -> go.Figure:
    pct = (value - min_val) / (max_val - min_val + 1e-9)
    color = SUCCESS if pct > 0.7 else (WARNING if pct > 0.4 else DANGER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickfont": {"size": 10}},
            "bar":  {"color": color},
            "steps": [
                {"range": [min_val, min_val + (max_val - min_val) * 0.4], "color": "#FEE2E2"},
                {"range": [min_val + (max_val - min_val) * 0.4,
                           min_val + (max_val - min_val) * 0.7], "color": "#FEF3C7"},
                {"range": [min_val + (max_val - min_val) * 0.7, max_val], "color": "#D1FAE5"},
            ],
        },
        number={"font": {"size": 28}, "valueformat": ".3f"},
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Scatter Matrix ────────────────────────────────────────────────────────────

def scatter_matrix_chart(df: pd.DataFrame, cols: list, color_col: str, title: str = "") -> go.Figure:
    fig = px.scatter_matrix(
        df.sample(min(500, len(df))),
        dimensions=cols[:5],
        color=color_col,
        color_discrete_sequence=PALETTE,
        opacity=0.6,
    )
    fig.update_traces(diagonal_visible=False)
    return _base_layout(fig, title, height=500)


# ── Bar Comparison ────────────────────────────────────────────────────────────

def metric_comparison_bar(metrics_dict: dict, title: str = "Model Comparison") -> go.Figure:
    """Compare multiple models' metrics side-by-side."""
    models   = list(metrics_dict.keys())
    all_keys = list(next(iter(metrics_dict.values())).keys())

    fig = go.Figure()
    for i, key in enumerate(all_keys[:5]):
        vals = [metrics_dict[m].get(key, 0) for m in models]
        fig.add_trace(go.Bar(
            name=key.upper(), x=models, y=vals,
            marker_color=PALETTE[i % len(PALETTE)],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Score", range=[0, 1.1])
    return _base_layout(fig, title, height=380)


# ── Latency Distribution ──────────────────────────────────────────────────────

def latency_histogram(latencies: list, title: str = "Inference Latency Distribution") -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=latencies, nbinsx=30,
        marker_color=INFO, opacity=0.8,
    ))
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    for pct, val, color in [(50, p50, SUCCESS), (95, p95, WARNING)]:
        fig.add_vline(x=val, line_dash="dash", line_color=color,
                      annotation_text=f"P{pct}={val:.0f}ms",
                      annotation_position="top right")
    fig.update_xaxes(title_text="Latency (ms)")
    fig.update_yaxes(title_text="Requests")
    return _base_layout(fig, title, height=330)


# ── KPI Sparkline ─────────────────────────────────────────────────────────────

def sparkline(values: list, color: str = PRIMARY) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=values, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
    ))
    fig.update_layout(
        height=60, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig
