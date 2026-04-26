"""
Use Cases page — interactive exploration of all 6 business datasets.
Each tab shows: dataset overview, data quality, class distribution,
sample predictions, and key visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import CUSTOM_CSS, USE_CASES
from utils.data_generator import get_dataset, data_quality_report
from utils.visualizations import (
    class_distribution_chart, pie_chart, feature_importance_chart, PALETTE
)

st.set_page_config(page_title="Use Cases", page_icon="🎯", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🎯 Business Use Cases")
st.caption("Explore the six ML use cases: data distributions, quality metrics, and sample insights.")
st.divider()

# ── Cache all datasets ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating synthetic data…")
def load_all():
    return {k: get_dataset(k) for k in USE_CASES}

datasets = load_all()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "💬 Sentiment",
    "📧 Spam",
    "📉 Churn",
    "🔍 Fraud",
    "📈 Sales",
    "👥 HR Attrition",
])


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    df = datasets["sentiment"]
    st.subheader("💬 Sentiment Analysis — Customer Reviews")
    st.markdown("**Task:** Classify customer reviews as Positive / Neutral / Negative  \n"
                "**Algorithm:** TF-IDF + Logistic Regression (benchmark: Naive Bayes)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews",  len(df))
    c2.metric("Positive",       (df.sentiment == "Positive").sum())
    c3.metric("Neutral",        (df.sentiment == "Neutral").sum())
    c4.metric("Negative",       (df.sentiment == "Negative").sum())

    t1, t2, t3 = st.tabs(["📊 Distribution", "🔍 Data Quality", "📝 Sample Data"])
    with t1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = class_distribution_chart(df["sentiment"], "Sentiment Class Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            counts = df["sentiment"].value_counts()
            fig = pie_chart(counts.index.tolist(), counts.values.tolist(),
                            "Sentiment Share")
            st.plotly_chart(fig, use_container_width=True)

        # Word length distribution
        df["text_len"] = df["text"].str.split().str.len()
        fig = px.histogram(df, x="text_len", color="sentiment",
                           barmode="overlay",
                           color_discrete_sequence=PALETTE,
                           title="Review Length by Sentiment",
                           labels={"text_len": "Word Count"})
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        qr = data_quality_report(df)
        cq1, cq2, cq3, cq4 = st.columns(4)
        cq1.metric("Rows", qr["total_rows"])
        cq2.metric("Missing %", f"{qr['missing_pct']}%")
        cq3.metric("Duplicates", qr["duplicate_rows"])
        cq4.metric("Outliers", qr["outlier_count"])
        st.caption("No numeric outliers expected for text data. Missing applies to optional numeric fields.")

    with t3:
        for sentiment, color in [("Positive","#D1FAE5"), ("Neutral","#FEF3C7"), ("Negative","#FEE2E2")]:
            st.markdown(f"**{sentiment} examples:**")
            samples = df[df.sentiment == sentiment].sample(3, random_state=42)
            for _, row in samples.iterrows():
                st.markdown(
                    f"<div style='background:{color}; padding:8px 12px; border-radius:6px; margin:4px 0; font-size:0.9rem;'>"
                    f"📝 {row['text']}</div>",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPAM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    df = datasets["spam"]
    st.subheader("📧 Spam Detection — SMS & Email Messages")
    st.markdown("**Task:** Binary classification — Spam vs Ham  \n"
                "**Algorithm:** TF-IDF + Multinomial Naive Bayes (benchmark: Logistic Regression)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Messages", len(df))
    c2.metric("Ham",  (df.label == "Ham").sum())
    c3.metric("Spam", (df.label == "Spam").sum())

    t1, t2, t3 = st.tabs(["📊 Distribution", "🔑 Keywords", "📝 Samples"])
    with t1:
        col_a, col_b = st.columns(2)
        with col_a:
            counts = df["label"].value_counts()
            fig = pie_chart(counts.index.tolist(), counts.values.tolist(), "Ham vs Spam Split")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            df["msg_len"] = df["message"].str.split().str.len()
            fig = px.box(df, x="label", y="msg_len",
                         color="label",
                         color_discrete_sequence=["#1E40AF", "#DC2626"],
                         title="Message Length by Class",
                         labels={"msg_len": "Word Count"})
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        spam_keywords = {
            "free": 0.91, "winner": 0.88, "claim": 0.85, "urgent": 0.83,
            "prize": 0.82, "selected": 0.79, "congratulations": 0.77,
            "reward": 0.75, "cash": 0.73, "call immediately": 0.71,
            "limited": 0.68, "offer": 0.65, "bank details": 0.64, "award": 0.61,
        }
        fig = feature_importance_chart(spam_keywords, "Top Spam Signal Words")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        for lbl, color in [("Spam", "#FEE2E2"), ("Ham", "#D1FAE5")]:
            st.markdown(f"**{lbl} examples:**")
            for _, row in df[df.label == lbl].sample(3, random_state=42).iterrows():
                st.markdown(
                    f"<div style='background:{color}; padding:8px 12px; border-radius:6px; margin:4px 0; font-size:0.9rem;'>"
                    f"✉️ {row['message']}</div>",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CUSTOMER CHURN
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    df = datasets["churn"]
    st.subheader("📉 Customer Churn Prediction — Telecom Dataset")
    st.markdown("**Task:** Predict whether a telecom customer will churn  \n"
                "**Features:** Age, Region, Plan, Tenure, Bill, Complaints, Usage Drop, Payment Delays  \n"
                "**Algorithms:** Random Forest · XGBoost · Logistic Regression baseline")

    churn_rate = df["churn"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", len(df))
    c2.metric("Churned",   df["churn"].sum())
    c3.metric("Retained",  (df["churn"] == 0).sum())
    c4.metric("Churn Rate", f"{churn_rate:.1%}")

    t1, t2, t3 = st.tabs(["📊 Distributions", "🔍 Segment Analysis", "📋 Data Preview"])
    with t1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(df, x="tenure_months", color=df["churn"].map({0:"Retained",1:"Churned"}),
                               barmode="overlay",
                               color_discrete_sequence=["#059669","#DC2626"],
                               title="Tenure Distribution by Churn Status",
                               labels={"x": "Tenure (months)", "color": "Status"})
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.box(df, x="plan_type", y="monthly_bill",
                         color=df["churn"].map({0:"Retained",1:"Churned"}),
                         color_discrete_sequence=["#1E40AF","#DC2626"],
                         title="Monthly Bill by Plan & Churn Status")
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # Churn by region
        region_churn = df.groupby("region")["churn"].mean().reset_index()
        region_churn.columns = ["region", "churn_rate"]
        fig = px.bar(region_churn, x="region", y="churn_rate",
                     color="churn_rate",
                     color_continuous_scale=["#D1FAE5","#FEF3C7","#FEE2E2"],
                     title="Churn Rate by Region",
                     text=region_churn["churn_rate"].map("{:.1%}".format))
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        plan_churn = df.groupby("plan_type").agg(
            customers=("churn","count"),
            churn_count=("churn","sum"),
            churn_rate=("churn","mean"),
            avg_bill=("monthly_bill","mean"),
        ).reset_index()
        plan_churn["churn_rate"] = plan_churn["churn_rate"].map("{:.1%}".format)
        plan_churn["avg_bill"]   = plan_churn["avg_bill"].map("${:.0f}".format)
        st.dataframe(plan_churn, use_container_width=True, hide_index=True)

        fig = px.scatter(
            df.sample(500, random_state=42),
            x="usage_drop_pct", y="monthly_bill",
            color=df.sample(500, random_state=42)["churn"].map({0:"Retained",1:"Churned"}),
            size="num_complaints",
            color_discrete_sequence=["#1E40AF","#DC2626"],
            opacity=0.65, title="Usage Drop vs Monthly Bill (size = complaints)",
        )
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Full Dataset", csv,
                           "churn_dataset.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FRAUD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    df = datasets["fraud"]
    st.subheader("🔍 Fraud Detection — Financial Transactions")
    st.markdown("**Task:** Flag fraudulent transactions  \n"
                "**Features:** Amount, Merchant Type, Device/Geo Mismatch, Velocity, Chargeback History  \n"
                "**Algorithms:** Random Forest · Isolation Forest · XGBoost")

    fraud_rate = df["is_fraud"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", len(df))
    c2.metric("Fraud",   df["is_fraud"].sum())
    c3.metric("Legit",   (df["is_fraud"] == 0).sum())
    c4.metric("Fraud Rate", f"{fraud_rate:.2%}")

    t1, t2, t3 = st.tabs(["📊 Distributions", "⚠️ Risk Factors", "📋 High-Risk Transactions"])
    with t1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(df, x="amount", color=df["is_fraud"].map({0:"Legit",1:"Fraud"}),
                               nbins=50, barmode="overlay",
                               color_discrete_sequence=["#1E40AF","#DC2626"],
                               title="Transaction Amount Distribution",
                               labels={"color":"Status"})
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            merchant_fraud = df.groupby("merchant_type")["is_fraud"].mean().reset_index()
            merchant_fraud.columns = ["merchant_type","fraud_rate"]
            fig = px.bar(merchant_fraud, x="merchant_type", y="fraud_rate",
                         color="fraud_rate",
                         color_continuous_scale=["#D1FAE5","#FEE2E2"],
                         title="Fraud Rate by Merchant Type",
                         text=merchant_fraud["fraud_rate"].map("{:.1%}".format))
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        risk_factors = {
            "Previous Chargeback": df[df.prev_chargeback==1]["is_fraud"].mean(),
            "Device Mismatch":     df[df.device_mismatch==1]["is_fraud"].mean(),
            "Geo Mismatch":        df[df.geo_mismatch==1]["is_fraud"].mean(),
            "Night Transaction":   df[df.night_txn==1]["is_fraud"].mean(),
            "International":       df[df.is_international==1]["is_fraud"].mean(),
            "High Velocity (>10)": df[df.velocity_count>10]["is_fraud"].mean(),
        }
        rf_df = pd.DataFrame(list(risk_factors.items()), columns=["Risk Factor","Fraud Rate"])
        rf_df = rf_df.sort_values("Fraud Rate", ascending=True)
        fig = go.Figure(go.Bar(
            y=rf_df["Risk Factor"], x=rf_df["Fraud Rate"],
            orientation="h",
            marker_color=["#DC2626" if v > 0.3 else "#D97706" if v > 0.15 else "#059669"
                          for v in rf_df["Fraud Rate"]],
            text=[f"{v:.1%}" for v in rf_df["Fraud Rate"]],
            textposition="outside",
        ))
        fig.update_layout(title="Fraud Rate by Risk Factor",
                          height=350, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        high_risk = df[df.fraud_score > 0.6].sort_values("fraud_score", ascending=False).head(10)
        display = high_risk[["txn_id","amount","merchant_type","device_mismatch",
                              "geo_mismatch","prev_chargeback","fraud_score","is_fraud"]].copy()
        display["fraud_score"] = display["fraud_score"].map("{:.3f}".format)
        display["amount"]      = display["amount"].map("${:.2f}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SALES FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    df = datasets["sales"]
    st.subheader("📈 Sales Forecasting — Daily Sales Time Series")
    st.markdown("**Task:** Forecast daily sales for next 30/60/90 days  \n"
                "**Features:** Date components, Promotions, Holidays, Price Changes  \n"
                "**Algorithms:** Linear Regression baseline · Random Forest Regressor")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical Days", len(df))
    c2.metric("Avg Daily Sales", f"{df.sales.mean():.0f}")
    c3.metric("Max Sales Day",   f"{df.sales.max():.0f}")
    c4.metric("Promo Days",      df.promotion.sum())

    t1, t2, t3 = st.tabs(["📊 Time Series", "📅 Seasonality", "📋 Preview"])
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["sales"],
            mode="lines", name="Daily Sales",
            line=dict(color="#1E40AF", width=1.5),
        ))
        # Highlight promotions
        promo_days = df[df.promotion == 1]
        fig.add_trace(go.Scatter(
            x=promo_days["date"], y=promo_days["sales"],
            mode="markers", name="Promotion Day",
            marker=dict(color="#DC2626", size=5, symbol="circle"),
        ))
        fig.update_layout(title="2-Year Daily Sales History",
                          height=400, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        col_a, col_b = st.columns(2)
        with col_a:
            monthly = df.groupby("month")["sales"].mean().reset_index()
            months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly["month_name"] = monthly["month"].apply(lambda x: months[x-1])
            fig = px.bar(monthly, x="month_name", y="sales",
                         color="sales",
                         color_continuous_scale=["#BFDBFE","#1E40AF"],
                         title="Average Sales by Month")
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            weekly = df.groupby("day_of_week")["sales"].mean().reset_index()
            weekly["day_name"] = weekly["day_of_week"].apply(lambda x: days[x])
            fig = px.bar(weekly, x="day_name", y="sales",
                         color="sales",
                         color_continuous_scale=["#BFDBFE","#1E40AF"],
                         title="Average Sales by Day of Week")
            fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.dataframe(df.tail(14), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HR ATTRITION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    df = datasets["hr"]
    st.subheader("👥 HR Attrition Prediction — Employee Dataset")
    st.markdown("**Task:** Predict employee likelihood of leaving  \n"
                "**Features:** Department, Salary, Overtime, Satisfaction, Tenure, Manager Changes  \n"
                "**Algorithm:** Gradient Boosting")

    attr_rate = df["attrition"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", len(df))
    c2.metric("Left",           df["attrition"].sum())
    c3.metric("Stayed",         (df["attrition"] == 0).sum())
    c4.metric("Attrition Rate", f"{attr_rate:.1%}")

    t1, t2, t3 = st.tabs(["📊 Distributions", "🏢 Department Analysis", "📋 Preview"])
    with t1:
        col_a, col_b = st.columns(2)
        with col_a:
            dept_attr = df.groupby("department")["attrition"].mean().reset_index()
            fig = px.bar(dept_attr, x="department", y="attrition",
                         color="attrition",
                         color_continuous_scale=["#D1FAE5","#FEE2E2"],
                         title="Attrition Rate by Department",
                         text=dept_attr["attrition"].map("{:.1%}".format))
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.box(df, x="salary_band", y="satisfaction",
                         color=df["attrition"].map({0:"Stayed",1:"Left"}),
                         color_discrete_sequence=["#1E40AF","#DC2626"],
                         title="Satisfaction by Salary Band & Attrition")
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        dept_stats = df.groupby("department").agg(
            employees=("attrition","count"),
            attrition_count=("attrition","sum"),
            attrition_rate=("attrition","mean"),
            avg_satisfaction=("satisfaction","mean"),
            avg_tenure=("tenure_years","mean"),
            overtime_pct=("overtime","mean"),
        ).reset_index()
        dept_stats["attrition_rate"]  = dept_stats["attrition_rate"].map("{:.1%}".format)
        dept_stats["avg_satisfaction"]= dept_stats["avg_satisfaction"].map("{:.2f}".format)
        dept_stats["avg_tenure"]      = dept_stats["avg_tenure"].map("{:.1f} yrs".format)
        dept_stats["overtime_pct"]    = dept_stats["overtime_pct"].map("{:.1%}".format)
        st.dataframe(dept_stats, use_container_width=True, hide_index=True)

    with t3:
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download HR Dataset", csv, "hr_dataset.csv", "text/csv")
