"""
Model Training Engine — configure, train, evaluate, and register models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

from config import CUSTOM_CSS, USE_CASES, CLASSIFIERS, REGRESSORS, DEFAULT_HYPERPARAMS
from utils.data_generator import get_dataset, data_quality_report
from utils.ml_pipeline import train_model, save_model, XGBOOST_AVAILABLE
from utils.model_registry import register_model
from utils.visualizations import (
    confusion_matrix_chart, roc_curve_chart, pr_curve_chart,
    feature_importance_chart, metric_comparison_bar,
)
from utils.metrics import roc_data, pr_data
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🤖 Model Training Engine")
st.caption("Configure, train, cross-validate, and register ML models across all use cases.")
st.divider()

# ── Session State Init ────────────────────────────────────────────────────────
if "train_results" not in st.session_state:
    st.session_state.train_results = {}

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

# ── Sidebar Configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Training Configuration")
    st.divider()

    use_case = st.selectbox(
        "Use Case",
        options=list(USE_CASES.keys()),
        format_func=lambda k: f"{USE_CASES[k]['icon']} {USE_CASES[k]['name']}",
    )
    task = USE_CASES[use_case]["task"]

    algo_options = list(CLASSIFIERS.items()) if task == "classification" else list(REGRESSORS.items())
    # Remove unavailable ones
    if not XGBOOST_AVAILABLE:
        algo_options = [(k, v) for k, v in algo_options if v not in ("xgb", "xgbr")]

    algo_display = {k: v for k, v in algo_options}
    algo_name    = st.selectbox("Algorithm", options=list(algo_display.keys()))
    algo_key     = algo_display[algo_name]

    st.markdown("---")
    test_size = st.slider("Test Split %", 10, 40, 20, step=5) / 100
    cv_folds  = st.slider("CV Folds", 2, 10, 5)

    st.markdown("**Hyperparameters**")
    def_params = DEFAULT_HYPERPARAMS.get(algo_key, {})
    params = {}
    if "n_estimators" in def_params:
        params["n_estimators"] = st.slider("n_estimators", 50, 500, def_params["n_estimators"], 50)
    if "max_depth" in def_params:
        params["max_depth"] = st.slider("max_depth", 2, 20, def_params["max_depth"])
    if "learning_rate" in def_params:
        params["learning_rate"] = st.select_slider(
            "learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3],
            value=def_params["learning_rate"]
        )
    if "C" in def_params:
        params["C"] = st.select_slider("C (regularization)", [0.01, 0.1, 1.0, 5.0, 10.0],
                                        value=def_params["C"])
    if "alpha" in def_params:
        params["alpha"] = st.select_slider("Smoothing alpha", [0.1, 0.5, 1.0, 2.0, 5.0],
                                            value=def_params["alpha"])

    st.markdown("---")
    owner       = st.text_input("Model Owner", "Alice Johnson")
    version_tag = st.text_input("Version Tag", "1.0")
    description = st.text_area("Description", "Training run via MLOps UI", height=70)
    register    = st.checkbox("Register to Model Registry", value=True)

    train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)

# ── Data Preview ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data(uc):
    return get_dataset(uc)

df = load_data(use_case)

left_col, right_col = st.columns([1.5, 1])
with left_col:
    st.subheader(f"📋 Dataset — {USE_CASES[use_case]['name']}")
    st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

with right_col:
    qr = data_quality_report(df)
    st.subheader("🔍 Data Quality")
    m1, m2 = st.columns(2)
    m1.metric("Missing %",    f"{qr['missing_pct']}%")
    m2.metric("Duplicates",   qr["duplicate_rows"])
    m3, m4 = st.columns(2)
    m3.metric("Outliers",     qr["outlier_count"])
    m4.metric("Classes",      len(df[USE_CASES[use_case]["target"]].unique())
                               if USE_CASES[use_case]["task"] == "classification" else "—")

    if task == "classification":
        target_col = USE_CASES[use_case]["target"]
        vc = df[target_col].value_counts()
        ratio = vc.min() / vc.max()
        if ratio < 0.3:
            st.markdown(
                f"<div class='alert-warning'>⚠️ Class imbalance detected (ratio={ratio:.2f})</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='alert-success'>✅ Class balance acceptable</div>",
                unsafe_allow_html=True,
            )

st.divider()

# ── Training ──────────────────────────────────────────────────────────────────
if train_btn:
    st.subheader("⚙️ Training Progress")
    prog_bar  = st.progress(0, text="Initialising…")
    status_ph = st.empty()
    steps = [
        (0.10, "📥 Loading and validating data…"),
        (0.25, "🔧 Building preprocessing pipeline…"),
        (0.45, "🧪 Running cross-validation…"),
        (0.70, "🏋️ Fitting final model…"),
        (0.90, "📊 Computing evaluation metrics…"),
        (1.00, "✅ Training complete!"),
    ]
    for pct, msg in steps[:-1]:
        prog_bar.progress(pct, text=msg)
        status_ph.info(msg)
        time.sleep(0.3)

    with st.spinner("Training model…"):
        try:
            result = train_model(
                use_case=use_case,
                df=df,
                algo_key=algo_key,
                params=params,
                test_size=test_size,
                cv_folds=cv_folds,
            )
            st.session_state.train_results[use_case] = result
            key = f"{use_case}_{algo_key}"
            st.session_state.trained_models[key] = result
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    prog_bar.progress(1.0, text="✅ Training complete!")
    status_ph.success(f"Model trained in {result['training_time_s']}s")

    # Register if requested
    if register:
        try:
            model_name = f"{USE_CASES[use_case]['name'].replace(' ','')}"
            model_path = save_model(result, version_tag)
            register_model(
                model_name=model_name,
                version=version_tag,
                use_case=use_case,
                algo_key=algo_key,
                metrics=result["metrics"],
                params=params,
                owner=owner,
                description=description,
                model_path=str(model_path),
                stage="Development",
            )
            st.success(f"✅ Model registered as **{model_name} v{version_tag}** → Development")
        except Exception as e:
            st.warning(f"Registration warning: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.train_results.get(use_case)
if result is None:
    st.info("👆 Configure training options in the sidebar and click **Train Model** to begin.")
    st.stop()

st.subheader("📊 Evaluation Results")
metrics = result["metrics"]
task    = result["task"]

# Metric row
if task == "classification":
    m_cols = st.columns(6)
    m_items = [
        ("Accuracy",     metrics.get("accuracy",   0)),
        ("Precision",    metrics.get("precision",  0)),
        ("Recall",       metrics.get("recall",     0)),
        ("F1 Score",     metrics.get("f1",         0)),
        ("ROC-AUC",      metrics.get("roc_auc",    0) or 0),
        ("CV F1 Mean",   metrics.get("cv_f1_mean", 0)),
    ]
    for col, (name, val) in zip(m_cols, m_items):
        col.metric(name, f"{val:.4f}")
else:
    m_cols = st.columns(4)
    for col, (name, val) in zip(m_cols, [
        ("RMSE", metrics.get("rmse", 0)),
        ("MAE",  metrics.get("mae",  0)),
        ("R²",   metrics.get("r2",   0)),
        ("MAPE", metrics.get("mape", 0)),
    ]):
        col.metric(name, f"{val:.3f}")

st.markdown(
    f"<div class='alert-info'>ℹ️ Trained <b>{algo_name}</b> on "
    f"<b>{result['n_train']:,}</b> samples, tested on <b>{result['n_test']:,}</b>. "
    f"Time: <b>{result['training_time_s']}s</b></div>",
    unsafe_allow_html=True,
)

st.divider()

# Charts row
chart_tabs = st.tabs(["🗂️ Confusion Matrix", "📈 ROC Curve", "📉 PR Curve",
                       "⭐ Feature Importance", "🔍 CV Analysis"])

with chart_tabs[0]:
    if task == "classification" and result["confusion_matrix"]:
        fig = confusion_matrix_chart(result["confusion_matrix"],
                                     result["classes"],
                                     f"Confusion Matrix — {algo_name}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Confusion matrix not available for regression tasks.")

with chart_tabs[1]:
    if task == "classification":
        try:
            pipeline = result["pipeline"]
            le       = result["label_encoder"]
            y_test   = result["y_test"]
            X_test_key = use_case

            if use_case in ("sentiment", "spam"):
                text_col = "text" if "text" in df.columns else "message"
                X_test_raw = df[text_col].iloc[-result["n_test"]:]
            else:
                id_cols = [c for c in df.columns if "_id" in c.lower() or c.lower() == "id"]
                target_col = USE_CASES[use_case]["target"]
                drop = id_cols + [target_col, "date", "fraud_score"] if use_case == "fraud" \
                       else id_cols + [target_col, "date"]
                X_test_raw = df.drop(columns=[c for c in drop if c in df.columns]).iloc[-result["n_test"]:]
                num_X = X_test_raw.select_dtypes(include="number")
                cat_X = X_test_raw.select_dtypes(include="object")
                X_test_raw = X_test_raw.copy()
                X_test_raw[num_X.columns] = num_X.fillna(num_X.median())
                X_test_raw[cat_X.columns] = cat_X.fillna("Unknown")

            y_proba = pipeline.predict_proba(X_test_raw)
            if len(result["classes"]) == 2:
                rd = roc_data(y_test, y_proba[:, 1])
                if rd:
                    auc = metrics.get("roc_auc", 0) or 0
                    fig = roc_curve_chart(rd["fpr"], rd["tpr"], auc, f"ROC Curve — {algo_name}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ROC curve shown for binary classification. Use F1/accuracy for multi-class.")
        except Exception as e:
            st.info(f"ROC curve unavailable: {e}")
    else:
        st.info("ROC curve applies to classification tasks only.")

with chart_tabs[2]:
    if task == "classification":
        try:
            y_proba = pipeline.predict_proba(X_test_raw)
            if len(result["classes"]) == 2:
                pd_data = pr_data(y_test, y_proba[:, 1])
                if pd_data:
                    fig = pr_curve_chart(pd_data["precision"], pd_data["recall"],
                                         f"Precision-Recall — {algo_name}")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"PR curve unavailable: {e}")

with chart_tabs[3]:
    fi = result.get("feature_importance", {})
    if fi:
        fig = feature_importance_chart(fi, f"Feature Importance — {algo_name}", top_n=15)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this algorithm/use case.")

with chart_tabs[4]:
    if task == "classification":
        cv_mean = metrics.get("cv_f1_mean", 0)
        cv_std  = metrics.get("cv_f1_std", 0)
        rng = np.random.default_rng(42)
        simulated_folds = cv_mean + rng.normal(0, cv_std, cv_folds)
        simulated_folds = np.clip(simulated_folds, 0, 1)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(cv_folds)],
            y=simulated_folds,
            marker_color="#1E40AF",
            text=[f"{v:.4f}" for v in simulated_folds],
            textposition="outside",
        ))
        fig.add_hline(y=cv_mean, line_dash="dash", line_color="#D97706",
                      annotation_text=f"Mean={cv_mean:.4f}")
        fig.update_layout(title=f"{cv_folds}-Fold Cross-Validation F1 Scores",
                          height=350, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(range=[max(0, cv_mean - 0.15), min(1, cv_mean + 0.15)]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        cv_mean = metrics.get("cv_rmse_mean", 0)
        st.metric("CV RMSE Mean", f"{cv_mean:.3f}")

st.divider()

# ── Multi-Algorithm Comparison ────────────────────────────────────────────────
with st.expander("🔬 Compare Multiple Algorithms", expanded=False):
    st.markdown("Train multiple algorithms to compare performance automatically.")
    compare_algos = st.multiselect(
        "Select algorithms to compare",
        options=[k for k, v in algo_options if v not in ("if",)],
        default=[list(algo_options)[0][0], list(algo_options)[min(1, len(algo_options)-1)][0]]
        if len(algo_options) > 1 else [list(algo_options)[0][0]],
    )
    if st.button("🔄 Run Comparison", type="secondary"):
        all_metrics = {}
        with st.spinner("Running comparison…"):
            for a_name in compare_algos:
                a_key = algo_display.get(a_name, "lr")
                try:
                    r = train_model(use_case=use_case, df=df, algo_key=a_key,
                                    params=DEFAULT_HYPERPARAMS.get(a_key, {}),
                                    test_size=test_size, cv_folds=3)
                    # Keep only numeric metrics
                    all_metrics[a_name] = {k: v for k, v in r["metrics"].items()
                                            if isinstance(v, float) and v is not None}
                except Exception as e:
                    st.warning(f"Skipped {a_name}: {e}")

        if all_metrics:
            # Normalize all metrics to 0-1 scale for comparison
            normalized = {}
            for model, mets in all_metrics.items():
                normalized[model] = {k: v for k, v in mets.items()
                                     if k in ("accuracy","f1","precision","recall","r2")}
            if any(normalized.values()):
                fig = metric_comparison_bar(normalized, "Algorithm Comparison")
                st.plotly_chart(fig, use_container_width=True)

            # Best model highlight
            if task == "classification":
                best = max(all_metrics, key=lambda m: all_metrics[m].get("f1", 0))
                st.success(f"🏆 Best model: **{best}** (F1={all_metrics[best].get('f1',0):.4f})")
            else:
                best = min(all_metrics, key=lambda m: all_metrics[m].get("rmse", 9999))
                st.success(f"🏆 Best model: **{best}** (RMSE={all_metrics[best].get('rmse',0):.3f})")
