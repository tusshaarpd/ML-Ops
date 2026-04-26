"""
Live Predictions, Batch Inference, and API Simulation page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import uuid
from datetime import datetime

from config import CUSTOM_CSS, USE_CASES, LATENCY_RANGE_MS
from utils.data_generator import get_dataset
from utils.ml_pipeline import train_model, predict_single, XGBOOST_AVAILABLE
from utils.visualizations import (
    feature_importance_chart, gauge_chart, latency_histogram, pie_chart, PALETTE
)

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🔮 Live Predictions & Deployment Simulation")
st.caption("Single predictions, batch scoring, API simulation, and explainability.")
st.divider()

# ── Model Cache ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models for inference…")
def get_trained_models():
    """Train one representative model per use case for serving."""
    models = {}
    algo_map = {
        "sentiment": "lr",
        "spam":      "nb",
        "churn":     "rf",
        "fraud":     "rf",
        "sales":     "rfr",
        "hr":        "gb",
    }
    for uc, algo in algo_map.items():
        try:
            df = get_dataset(uc)
            result = train_model(uc, df, algo, cv_folds=3)
            models[uc] = result
        except Exception as e:
            st.warning(f"Could not train {uc}: {e}")
    return models

@st.cache_data(show_spinner="Loading datasets…")
def load_all_datasets():
    return {k: get_dataset(k) for k in USE_CASES}

models   = get_trained_models()
datasets = load_all_datasets()

main_tabs = st.tabs([
    "🎯 Single Prediction",
    "📦 Batch Inference",
    "🌐 API Simulation",
    "💡 Explainability",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[0]:
    st.subheader("🎯 Live Single Prediction")
    use_case = st.selectbox(
        "Select Use Case",
        list(USE_CASES.keys()),
        format_func=lambda k: f"{USE_CASES[k]['icon']} {USE_CASES[k]['name']}",
        key="pred_uc"
    )
    result = models.get(use_case)
    if result is None:
        st.error(f"No model available for {use_case}. Go to Training page first.")
        st.stop()

    task = USE_CASES[use_case]["task"]

    # ── Input Form ─────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        if use_case == "sentiment":
            st.markdown("**Enter a customer review:**")
            text = st.text_area(
                "Review Text",
                "The product quality is absolutely amazing and delivery was super fast!",
                height=100,
            )
            input_data = text

        elif use_case == "spam":
            st.markdown("**Enter a message:**")
            text = st.text_area(
                "Message",
                "Claim your FREE prize now! You have been selected as a winner.",
                height=80,
            )
            input_data = text

        elif use_case == "churn":
            st.markdown("**Enter customer details:**")
            c1, c2, c3 = st.columns(3)
            age          = c1.number_input("Age", 18, 80, 35)
            tenure       = c2.number_input("Tenure (months)", 1, 120, 12)
            monthly_bill = c3.number_input("Monthly Bill ($)", 20, 300, 85)
            c4, c5, c6 = st.columns(3)
            complaints   = c4.number_input("Complaints", 0, 20, 3)
            usage_drop   = c5.slider("Usage Drop (%)", 0, 100, 45)
            pay_delays   = c6.number_input("Payment Delays", 0, 12, 2)
            region       = st.selectbox("Region", ["North","South","East","West","Central"])
            plan         = st.selectbox("Plan Type", ["Basic","Standard","Premium","Enterprise"])
            input_data   = {
                "age": age, "region": region, "plan_type": plan,
                "tenure_months": tenure, "monthly_bill": monthly_bill,
                "num_complaints": complaints, "usage_drop_pct": float(usage_drop),
                "payment_delays": pay_delays,
            }

        elif use_case == "fraud":
            st.markdown("**Enter transaction details:**")
            c1, c2, c3 = st.columns(3)
            amount       = c1.number_input("Amount ($)", 1.0, 5000.0, 250.0, step=10.0)
            velocity     = c2.number_input("Velocity Count", 1, 30, 5)
            card_age     = c3.number_input("Card Age (months)", 1, 120, 24)
            c4, c5 = st.columns(2)
            merchant     = c4.selectbox("Merchant Type", ["Retail","Online","Travel","Restaurant","ATM"])
            dev_mismatch = c5.checkbox("Device Mismatch")
            c6, c7, c8 = st.columns(3)
            geo_mismatch = c6.checkbox("Geo Mismatch")
            prev_cb      = c7.checkbox("Previous Chargeback")
            night_txn    = c8.checkbox("Night Transaction")
            intl         = st.checkbox("International Transaction")
            input_data   = {
                "amount": float(amount), "merchant_type": merchant,
                "device_mismatch": int(dev_mismatch), "geo_mismatch": int(geo_mismatch),
                "velocity_count": velocity, "prev_chargeback": int(prev_cb),
                "night_txn": int(night_txn), "card_age_months": card_age,
                "is_international": int(intl),
            }

        elif use_case == "sales":
            st.markdown("**Enter sales context for one day:**")
            c1, c2, c3 = st.columns(3)
            promo        = c1.checkbox("Promotion Active")
            holiday      = c2.checkbox("Holiday Flag")
            price_chg    = c3.selectbox("Price Change", [-1, 0, 1])
            c4, c5 = st.columns(2)
            month        = c4.slider("Month", 1, 12, 6)
            day_of_week  = c5.selectbox("Day of Week", list(range(7)),
                                         format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d])
            input_data   = {
                "promotion": int(promo), "holiday_flag": int(holiday),
                "price_change": price_chg, "day_of_week": day_of_week,
                "month": month, "quarter": (month-1)//3+1,
                "year": 2026, "day_of_year": month * 30,
            }

        elif use_case == "hr":
            st.markdown("**Enter employee details:**")
            c1, c2, c3 = st.columns(3)
            dept         = c1.selectbox("Department", ["Engineering","Sales","Marketing","HR","Finance","Operations"])
            sal_band     = c2.selectbox("Salary Band", ["Low","Mid","High","Executive"])
            tenure_yr    = c3.number_input("Tenure (years)", 0, 20, 3)
            c4, c5, c6 = st.columns(3)
            sat          = c4.slider("Satisfaction (1-5)", 1.0, 5.0, 3.2, step=0.1)
            mgr_changes  = c5.number_input("Manager Changes", 0, 5, 1)
            overtime     = c6.checkbox("Works Overtime")
            c7, c8 = st.columns(2)
            distance     = c7.number_input("Distance (km)", 1, 60, 15)
            training     = c8.number_input("Training Hours", 0, 80, 20)
            input_data   = {
                "department": dept, "salary_band": sal_band, "overtime": int(overtime),
                "satisfaction": float(sat), "tenure_years": tenure_yr,
                "manager_changes": mgr_changes, "promotions": 1,
                "distance_km": distance, "training_hours": training,
            }

        predict_btn = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)

    # ── Run Prediction ─────────────────────────────────────────────────────────
    if predict_btn:
        t0 = time.perf_counter()
        try:
            pred_out = predict_single(result, input_data)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        prediction = pred_out["prediction"]
        proba      = pred_out["probabilities"]

        # Result card
        colors_map = {
            "Positive": "#059669", "Negative": "#DC2626", "Neutral": "#D97706",
            "Spam":     "#DC2626", "Ham":      "#059669",
            "Churned":  "#DC2626", "Retained": "#059669",
            "Fraud":    "#DC2626", "Legit":    "#059669",
            "Leave":    "#DC2626", "Stay":     "#059669",
        }
        color = colors_map.get(str(prediction), "#1E40AF")

        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,{color}15,{color}08);
                        border:2px solid {color}55; border-radius:14px;
                        padding:20px 24px; text-align:center; margin:12px 0;'>
                <h2 style='color:{color}; margin:0;'>Prediction: {prediction}</h2>
                <p style='color:#6B7280; margin:4px 0;'>Inference latency: <b>{latency_ms} ms</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if proba:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Probability Scores:**")
                for cls, prob in sorted(proba.items(), key=lambda x: x[1], reverse=True):
                    bar_color = colors_map.get(str(cls), "#1E40AF")
                    st.markdown(
                        f"""
                        <div style='display:flex;align-items:center;margin:4px 0;'>
                            <span style='width:90px;font-size:0.88rem;color:#374151;'>{cls}</span>
                            <div style='flex:1;background:#E5E7EB;border-radius:6px;height:16px;margin:0 8px;'>
                                <div style='width:{prob*100:.1f}%;background:{bar_color};
                                            border-radius:6px;height:16px;'></div>
                            </div>
                            <span style='font-weight:700;color:{bar_color};width:55px;
                                         text-align:right;font-size:0.88rem;'>{prob:.2%}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with col_b:
                top_cls   = list(proba.keys())
                top_proba = list(proba.values())
                fig = pie_chart(top_cls, top_proba, "Confidence Distribution")
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – BATCH INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.subheader("📦 Batch Inference")
    b_uc = st.selectbox(
        "Use Case for Batch Scoring",
        list(USE_CASES.keys()),
        format_func=lambda k: f"{USE_CASES[k]['icon']} {USE_CASES[k]['name']}",
        key="batch_uc",
    )
    b_result = models.get(b_uc)
    df_full  = datasets[b_uc]

    n_batch = st.slider("Batch Size", 10, 500, 100, step=10)

    if st.button("▶️ Run Batch Scoring", type="primary"):
        if b_result is None:
            st.error("No model for this use case.")
            st.stop()

        batch_df = df_full.sample(n_batch, random_state=99).copy()
        target   = USE_CASES[b_uc]["target"]
        task     = USE_CASES[b_uc]["task"]

        with st.spinner("Running batch inference…"):
            pipeline = b_result["pipeline"]
            le       = b_result["label_encoder"]

            if b_uc in ("sentiment", "spam"):
                text_col = "text" if "text" in batch_df.columns else "message"
                X_batch  = batch_df[text_col].fillna("").astype(str)
            else:
                id_cols = [c for c in batch_df.columns if "_id" in c.lower()]
                drop    = id_cols + [target,"date","fraud_score"] if b_uc=="fraud" \
                          else id_cols + [target,"date"]
                X_batch = batch_df.drop(columns=[c for c in drop if c in batch_df.columns])
                num_X   = X_batch.select_dtypes(include="number")
                cat_X   = X_batch.select_dtypes(include="object")
                X_batch = X_batch.copy()
                X_batch[num_X.columns] = num_X.fillna(num_X.median())
                X_batch[cat_X.columns] = cat_X.fillna("Unknown")

            preds = pipeline.predict(X_batch)
            if le is not None:
                preds = le.inverse_transform(preds)

            try:
                probas = pipeline.predict_proba(X_batch)
                conf   = probas.max(axis=1)
            except Exception:
                conf   = np.ones(len(preds))

            batch_df["prediction"]  = preds
            batch_df["confidence"]  = conf.round(3)
            latencies = np.random.randint(
                LATENCY_RANGE_MS[0], LATENCY_RANGE_MS[1], n_batch
            )
            batch_df["latency_ms"] = latencies

        st.success(f"✅ Scored {n_batch} records in {latencies.sum()} ms total")

        m1, m2, m3 = st.columns(3)
        m1.metric("Records Processed", n_batch)
        m2.metric("Avg Confidence",    f"{conf.mean():.1%}")
        m3.metric("Avg Latency (ms)",  f"{latencies.mean():.0f}")

        # Show results
        display_cols = ["prediction", "confidence", "latency_ms"] + \
                       [c for c in batch_df.columns if c not in
                        ["prediction","confidence","latency_ms"] + [target] and
                        not c.endswith("_id") and c not in ["date","fraud_score"]][:5]
        st.dataframe(batch_df[display_cols].head(20), use_container_width=True, hide_index=True)

        # Distribution of predictions
        from utils.visualizations import class_distribution_chart
        if task == "classification":
            fig = class_distribution_chart(
                pd.Series(preds), "Batch Prediction Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = batch_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Batch Results", csv,
                           f"batch_results_{b_uc}.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – API SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.subheader("🌐 REST API Inference Simulation")
    st.markdown("Simulates a real-time REST endpoint: `/predict` with JSON payload and response.")

    api_uc = st.selectbox(
        "Use Case Endpoint",
        list(USE_CASES.keys()),
        format_func=lambda k: f"{USE_CASES[k]['icon']} {USE_CASES[k]['name']}",
        key="api_uc",
    )

    # Sample payloads
    sample_payloads = {
        "sentiment": {"text": "This product is absolutely wonderful!"},
        "spam":      {"message": "WIN FREE cash prize now call immediately"},
        "churn":     {"age": 42, "region": "North", "plan_type": "Basic",
                      "tenure_months": 6, "monthly_bill": 95.0,
                      "num_complaints": 4, "usage_drop_pct": 65.0,
                      "payment_delays": 3},
        "fraud":     {"amount": 1250.0, "merchant_type": "Online",
                      "device_mismatch": 1, "geo_mismatch": 1,
                      "velocity_count": 12, "prev_chargeback": 0,
                      "night_txn": 1, "card_age_months": 6,
                      "is_international": 1},
        "sales":     {"promotion": 1, "holiday_flag": 0, "price_change": 0,
                      "day_of_week": 4, "month": 6, "quarter": 2,
                      "year": 2026, "day_of_year": 160},
        "hr":        {"department": "Engineering", "salary_band": "Low",
                      "overtime": 1, "satisfaction": 2.1,
                      "tenure_years": 2, "manager_changes": 3,
                      "promotions": 0, "distance_km": 40, "training_hours": 5},
    }

    payload_json = st.text_area(
        "Request Payload (JSON)",
        value=json.dumps(sample_payloads[api_uc], indent=2),
        height=200,
    )

    n_requests = st.slider("Simulate N concurrent requests", 1, 50, 10)
    endpoint   = f"POST /api/v1/predict/{api_uc}"

    if st.button("📡 Send Requests", type="primary"):
        api_result = models.get(api_uc)
        if api_result is None:
            st.error("No model loaded.")
            st.stop()

        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        logs = []
        latencies_all = []
        with st.spinner(f"Sending {n_requests} requests to {endpoint}…"):
            for i in range(n_requests):
                t0 = time.perf_counter()
                try:
                    if api_uc in ("sentiment", "spam"):
                        inp = payload.get("text") or payload.get("message", "")
                    else:
                        inp = payload
                    out     = predict_single(api_result, inp)
                    lat_ms  = round((time.perf_counter() - t0) * 1000
                                    + np.random.randint(*LATENCY_RANGE_MS), 1)
                    latencies_all.append(lat_ms)
                    top_class  = str(out["prediction"])
                    top_prob   = max(out["probabilities"].values()) if out["probabilities"] else 1.0
                    req_id     = str(uuid.uuid4())[:8].upper()
                    logs.append({
                        "Request ID": req_id,
                        "Endpoint":   endpoint,
                        "Prediction": top_class,
                        "Confidence": f"{top_prob:.2%}",
                        "Latency ms": lat_ms,
                        "Status":     "200 OK",
                        "Timestamp":  datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
                    })
                except Exception as e:
                    logs.append({
                        "Request ID": str(uuid.uuid4())[:8].upper(),
                        "Endpoint":   endpoint,
                        "Prediction": "ERROR",
                        "Confidence": "—",
                        "Latency ms": 0,
                        "Status":     "500 ERROR",
                        "Timestamp":  datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
                    })

        log_df = pd.DataFrame(logs)
        success_rate = (log_df["Status"] == "200 OK").mean()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Requests Sent",   n_requests)
        m2.metric("Success Rate",    f"{success_rate:.0%}")
        m3.metric("P50 Latency",     f"{np.percentile(latencies_all, 50):.0f} ms")
        m4.metric("P95 Latency",     f"{np.percentile(latencies_all, 95):.0f} ms")

        st.markdown("**Request Logs:**")
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = latency_histogram(latencies_all, "Request Latency Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            preds_series = log_df[log_df["Prediction"] != "ERROR"]["Prediction"]
            if not preds_series.empty:
                vc = preds_series.value_counts()
                from utils.visualizations import pie_chart as _pie
                fig = _pie(vc.index.tolist(), vc.values.tolist(), "Prediction Distribution")
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[3]:
    st.subheader("💡 Explainable AI — Feature Attribution")
    exp_uc = st.selectbox(
        "Use Case",
        list(USE_CASES.keys()),
        format_func=lambda k: f"{USE_CASES[k]['icon']} {USE_CASES[k]['name']}",
        key="exp_uc",
    )
    exp_result = models.get(exp_uc)

    if exp_result is None:
        st.error("No model for this use case.")
        st.stop()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Global Feature Importance**")
        fi = exp_result.get("feature_importance", {})
        if fi:
            fig = feature_importance_chart(fi, f"Feature Importance — {USE_CASES[exp_uc]['name']}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for text models (see word importance below).")

    with col_b:
        st.markdown("**Model Confidence Distribution**")
        # Simulate confidence scores on sample data
        df_sample = datasets[exp_uc].sample(min(200, len(datasets[exp_uc])), random_state=42)
        task = USE_CASES[exp_uc]["task"]
        target = USE_CASES[exp_uc]["target"]

        if task == "classification":
            try:
                pipeline = exp_result["pipeline"]
                if exp_uc in ("sentiment","spam"):
                    text_col = "text" if "text" in df_sample.columns else "message"
                    X_s = df_sample[text_col].fillna("").astype(str)
                else:
                    id_cols = [c for c in df_sample.columns if "_id" in c.lower()]
                    drop = id_cols + [target,"date","fraud_score"] if exp_uc=="fraud" \
                           else id_cols + [target,"date"]
                    X_s = df_sample.drop(columns=[c for c in drop if c in df_sample.columns])
                    X_s = X_s.copy()
                    X_s[X_s.select_dtypes("number").columns] = \
                        X_s.select_dtypes("number").fillna(X_s.select_dtypes("number").median())
                    X_s[X_s.select_dtypes("object").columns] = \
                        X_s.select_dtypes("object").fillna("Unknown")

                probas = pipeline.predict_proba(X_s)
                max_conf = probas.max(axis=1)

                import plotly.graph_objects as go
                fig = go.Figure(go.Histogram(
                    x=max_conf, nbinsx=20,
                    marker_color="#1E40AF", opacity=0.8,
                ))
                fig.add_vline(x=max_conf.mean(), line_dash="dash",
                              line_color="#D97706",
                              annotation_text=f"Mean={max_conf.mean():.2f}")
                fig.update_layout(
                    title="Model Confidence Distribution",
                    xaxis_title="Max Prediction Confidence",
                    yaxis_title="Frequency",
                    height=360, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Confidence chart unavailable: {e}")

    # Local explanation
    st.markdown("---")
    st.markdown("**Local Explanation** — Why did the model predict this?")

    if USE_CASES[exp_uc]["task"] == "classification":
        classes = exp_result.get("classes", [])
        le      = exp_result.get("label_encoder")

        # Pick a random sample and explain it
        if exp_uc in ("sentiment","spam"):
            sample_text = st.text_input(
                "Enter text for local explanation",
                "This is terrible service and I want my money back",
                key="local_exp_text",
            )
            if sample_text:
                try:
                    pipeline = exp_result["pipeline"]
                    proba    = pipeline.predict_proba(pd.Series([sample_text]))[0]
                    pred     = pipeline.predict(pd.Series([sample_text]))[0]
                    if le:
                        pred = le.inverse_transform([pred])[0]
                        class_labels = le.classes_
                    else:
                        class_labels = classes

                    st.markdown(f"**Prediction: `{pred}`**")
                    for cls, p in zip(class_labels, proba):
                        st.markdown(
                            f"""
                            <div style='display:flex;align-items:center;margin:3px 0;'>
                                <span style='width:80px;font-size:0.85rem;'>{cls}</span>
                                <div style='flex:1;background:#E5E7EB;border-radius:4px;height:14px;margin:0 8px;'>
                                    <div style='width:{p*100:.1f}%;background:#1E40AF;border-radius:4px;height:14px;'></div>
                                </div>
                                <span style='font-weight:700;width:50px;text-align:right;font-size:0.85rem;'>{p:.2%}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Show top contributing words
                    tfidf = pipeline.named_steps["tfidf"]
                    clf   = pipeline.named_steps["clf"]
                    feat_names = tfidf.get_feature_names_out()
                    text_vec = tfidf.transform([sample_text]).toarray()[0]
                    active_feats = {feat_names[i]: float(text_vec[i])
                                    for i in text_vec.nonzero()[0]}
                    if hasattr(clf, "coef_"):
                        coefs = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
                        weighted = {k: v * coefs[list(feat_names).index(k)]
                                    for k, v in active_feats.items()
                                    if k in feat_names}
                        sorted_w = sorted(weighted.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                        st.markdown("**Token contributions:**")
                        for word, score in sorted_w:
                            color = "#059669" if score > 0 else "#DC2626"
                            st.markdown(
                                f"<span style='background:{color}22; color:{color}; "
                                f"padding:3px 8px; border-radius:12px; margin:2px; "
                                f"font-size:0.85rem; display:inline-block;'>"
                                f"{word}: {score:+.3f}</span>",
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    st.error(f"Local explanation error: {e}")
        else:
            st.info(
                "For tabular models, the **Global Feature Importance** chart above shows "
                "which features drive predictions most. "
                "Select a specific customer/transaction in Batch Inference to see individual scores."
            )
    else:
        st.info("Local explanations are available for classification models.")
