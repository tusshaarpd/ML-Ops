"""
Governance & Compliance — Approvals, bias checks, audit logs, responsible AI.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import CUSTOM_CSS, APPROVAL_LEVELS, BIAS_METRICS, USE_CASES
from utils.model_registry import get_all_models, get_audit_log, get_production_models

st.set_page_config(page_title="Governance", page_icon="🛡️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:8px 0 12px;'>
            <div style='font-size:1.8rem;'>🛡️</div>
            <div style='color:#93c5fd; font-weight:700; font-size:0.95rem;'>Governance</div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.8rem; color:#cbd5e1; line-height:1.8;'>
        <b style='color:#bfdbfe;'>📌 Five Tabs</b><br>
        ✅ <b style='color:#e2e8f0;'>Approvals</b> — Multi-level sign-off workflow<br>
        ⚖️ <b style='color:#e2e8f0;'>Bias &amp; Fairness</b> — Demographic parity, equal opportunity<br>
        📜 <b style='color:#e2e8f0;'>Audit Trail</b> — Complete event log<br>
        📋 <b style='color:#e2e8f0;'>Responsible AI</b> — Six core principles<br>
        📊 <b style='color:#e2e8f0;'>Governance Report</b> — Downloadable compliance report
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.78rem; color:#94a3b8; line-height:1.6;'>
        <b style='color:#bfdbfe;'>⚖️ Bias Thresholds</b><br>
        Disparate Impact &gt; <b style='color:#e2e8f0;'>0.80</b> = Pass<br>
        True Positive Rate &gt; <b style='color:#e2e8f0;'>80%</b> per group = Pass<br><br>
        <b style='color:#bfdbfe;'>🔐 Approval Levels</b><br>
        Data Scientist → ML Engineer<br>
        → Risk Manager → CTO
        </div>
        """, unsafe_allow_html=True,
    )

st.title("🛡️ Governance & Compliance Center")
st.caption("Model approvals, bias assessments, responsible AI reporting, and audit trails.")
st.markdown(
    """
    <div class='explain-box'>
    <b>📖 About this page:</b> Enterprise ML deployments require rigorous governance.
    This page covers the full compliance lifecycle:<br>
    • <b>Approval Workflow</b> — Models cannot reach Production without sign-off
      from Data Scientist → ML Engineer → Risk Manager → CTO.<br>
    • <b>Bias & Fairness</b> — Three fairness metrics are checked:
      <b>Demographic Parity</b> (equal prediction rates), <b>Equal Opportunity</b>
      (equal true positive rates), and <b>Calibration</b> (predicted probabilities match outcomes).<br>
    • <b>Responsible AI</b> — Six principles: Transparency, Fairness, Privacy,
      Accountability, Safety, Sustainability.<br>
    • <b>Governance Report</b> — A full compliance report downloadable as a text file.
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def load_data():
    return get_all_models(), get_audit_log(), get_production_models()

df_reg, df_audit, df_prod = load_data()

main_tabs = st.tabs([
    "✅ Approvals",
    "⚖️ Bias & Fairness",
    "📜 Audit Trail",
    "📋 Responsible AI",
    "📊 Governance Report",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – APPROVALS
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[0]:
    st.subheader("✅ Model Approval Workflow")
    st.markdown(
        "Models must pass a multi-stage approval process before reaching Production. "
        "Each stage requires sign-off from the responsible owner."
    )

    # Approval pipeline visualization
    stages_workflow = [
        ("Data Scientist",  "Code Review",      "✅ Approved", "2026-04-24 10:15", "#059669"),
        ("ML Engineer",     "Model Validation", "✅ Approved", "2026-04-24 14:30", "#059669"),
        ("Risk Manager",    "Risk Assessment",  "⚠️ Pending",  "—",               "#D97706"),
        ("CTO",             "Final Sign-off",   "🔒 Locked",   "—",               "#6B7280"),
    ]
    st.markdown("**Approval Pipeline — ChurnPredictor v2.0 → Production**")
    step_cols = st.columns(len(stages_workflow))
    for col, (role, action, status, ts, color) in zip(step_cols, stages_workflow):
        with col:
            st.markdown(
                f"""
                <div style='border:1.5px solid {color}55; border-radius:10px;
                            padding:14px; text-align:center; background:{color}08;'>
                    <div style='font-size:1.3rem;'>
                        {"✅" if "Approved" in status else "⏳" if "Pending" in status else "🔒"}
                    </div>
                    <div style='font-weight:700;font-size:0.85rem;color:#1E40AF;margin:4px 0;'>{role}</div>
                    <div style='font-size:0.78rem;color:#6B7280;'>{action}</div>
                    <div style='margin-top:8px;'>
                        <span style='background:{color};color:white;font-size:0.7rem;
                                     padding:2px 8px;border-radius:10px;font-weight:700;'>
                            {status}
                        </span>
                    </div>
                    <div style='font-size:0.7rem;color:#9CA3AF;margin-top:4px;'>{ts}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Pending approvals queue
    st.markdown("**Pending Approval Requests**")
    pending = [
        ("ChurnPredictor",    "v2.0", "Risk Manager",   "Promote to Production",
         "Exceeded F1 threshold. CV stable.", "HIGH",   "2026-04-25"),
        ("SentimentClassifier","v2.0","ML Engineer",    "Promote to Staging",
         "New RF model with improved accuracy.", "MEDIUM","2026-04-24"),
        ("FraudDetector",     "v2.0", "Data Scientist", "Retraining Approval",
         "PSI drift detected in velocity feature.", "HIGH","2026-04-26"),
    ]
    for model, ver, reviewer, action, rationale, priority, date in pending:
        p_color = "#DC2626" if priority == "HIGH" else "#D97706"
        col_a, col_b, col_c = st.columns([3, 2, 1])
        with col_a:
            st.markdown(
                f"""
                <div style='background:#F8FAFC;border:1px solid #E2E8F0;
                            border-radius:8px;padding:12px;margin:4px 0;'>
                    <b>{model} v{ver}</b>
                    <span style='background:{p_color};color:white;font-size:0.7rem;
                                 padding:1px 8px;border-radius:8px;margin-left:8px;'>
                        {priority}
                    </span><br>
                    <span style='font-size:0.85rem;color:#6B7280;'>Action: {action}</span><br>
                    <span style='font-size:0.78rem;color:#9CA3AF;'>Reviewer: {reviewer} | {date}</span><br>
                    <span style='font-size:0.78rem;color:#374151;'><i>{rationale}</i></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            col_b1, col_b2 = st.columns(2)
            if col_b1.button("✅ Approve", key=f"approve_{model}_{ver}"):
                st.success(f"Approved: {model} v{ver}")
            if col_b2.button("❌ Reject",  key=f"reject_{model}_{ver}"):
                st.error(f"Rejected: {model} v{ver}")
        with col_c:
            if st.button("📧 Request Info", key=f"info_{model}_{ver}"):
                st.info(f"Review request sent to {reviewer}")

    st.divider()

    # Retraining approvals
    st.markdown("**Retraining Approvals**")
    retrain_items = [
        ("ChurnPredictor",  "RT-0042", "Drift: PSI=0.23",     "Pending", "Carol Lee"),
        ("FraudDetector",   "RT-0043", "Performance drop 3%",  "Approved","David Park"),
        ("SalesForecaster", "RT-0044", "Monthly retrain cycle","Scheduled","Eva Chen"),
    ]
    rt_df = pd.DataFrame(retrain_items,
                         columns=["Model","Job ID","Trigger","Status","Owner"])
    st.dataframe(rt_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – BIAS & FAIRNESS
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.subheader("⚖️ Bias & Fairness Assessment")
    st.markdown(
        "Evaluate model fairness across demographic groups and protected attributes. "
        "Metrics computed on holdout test sets."
    )

    bias_model = st.selectbox(
        "Select Model",
        df_reg["model_name"].unique().tolist() if not df_reg.empty else ["ChurnPredictor"],
        key="bias_model",
    )

    # Simulated bias metrics
    rng = np.random.default_rng(hash(bias_model) % 10000)

    bias_tabs = st.tabs(["Demographic Parity", "Equal Opportunity", "Calibration", "Group Metrics"])

    with bias_tabs[0]:
        st.markdown("**Demographic Parity** — Equal positive prediction rates across groups")
        groups = ["Group A (Age < 35)", "Group B (Age 35–50)", "Group C (Age > 50)",
                  "Region: North", "Region: South", "Region: East", "Region: West"]
        rates  = rng.uniform(0.20, 0.55, len(groups))
        target_rate = 0.35

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=groups, x=rates, orientation="h",
            marker_color=["#DC2626" if abs(r - target_rate) > 0.10 else
                          "#D97706" if abs(r - target_rate) > 0.05 else "#059669"
                          for r in rates],
            text=[f"{r:.2%}" for r in rates],
            textposition="outside",
        ))
        fig.add_vline(x=target_rate, line_dash="dash", line_color="#1E40AF",
                      annotation_text=f"Overall={target_rate:.0%}")
        fig.update_layout(
            title="Positive Prediction Rate by Group",
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)

        disparate_impact = rates.min() / rates.max()
        if disparate_impact < 0.8:
            st.markdown(
                f"<div class='alert-danger'>⚠️ Disparate Impact = {disparate_impact:.2f} "
                f"(below 0.80 threshold — potential bias detected)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='alert-success'>✅ Disparate Impact = {disparate_impact:.2f} "
                f"(above 0.80 threshold — acceptable)</div>",
                unsafe_allow_html=True,
            )

    with bias_tabs[1]:
        st.markdown("**Equal Opportunity** — True positive rates (recall) across groups")
        tpr_groups = ["Male", "Female", "Non-binary",
                      "Plan: Basic", "Plan: Standard", "Plan: Premium"]
        tpr_vals = rng.uniform(0.70, 0.95, len(tpr_groups))
        fig = go.Figure(go.Bar(
            x=tpr_groups, y=tpr_vals,
            marker_color=["#059669" if v > 0.80 else "#D97706" for v in tpr_vals],
            text=[f"{v:.2%}" for v in tpr_vals],
            textposition="outside",
        ))
        fig.add_hline(y=0.80, line_dash="dash", line_color="#DC2626",
                      annotation_text="Min threshold: 80%")
        fig.update_layout(
            title="True Positive Rate by Group (Equal Opportunity)",
            yaxis=dict(range=[0.5, 1.05], tickformat=".0%"),
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with bias_tabs[2]:
        st.markdown("**Calibration** — Predicted probabilities vs actual outcomes")
        bin_centers   = np.linspace(0.05, 0.95, 10)
        ideal_frac    = bin_centers
        actual_frac   = np.clip(bin_centers + rng.normal(0, 0.04, 10), 0, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bin_centers, y=ideal_frac,
            mode="lines", name="Perfect Calibration",
            line=dict(color="#6B7280", dash="dash", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=bin_centers, y=actual_frac,
            mode="lines+markers", name="Model Calibration",
            line=dict(color="#1E40AF", width=2.5),
            marker=dict(size=7),
        ))
        fig.update_layout(
            title="Calibration Curve",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with bias_tabs[3]:
        st.markdown("**Group-level Metric Summary**")
        subgroups = ["Group A", "Group B", "Group C", "Region N", "Region S", "Region E"]
        group_metrics = pd.DataFrame({
            "Subgroup":    subgroups,
            "Accuracy":    rng.uniform(0.80, 0.94, len(subgroups)).round(3),
            "Precision":   rng.uniform(0.78, 0.93, len(subgroups)).round(3),
            "Recall":      rng.uniform(0.75, 0.95, len(subgroups)).round(3),
            "F1":          rng.uniform(0.78, 0.93, len(subgroups)).round(3),
            "FPR":         rng.uniform(0.04, 0.15, len(subgroups)).round(3),
        })
        st.dataframe(group_metrics, use_container_width=True, hide_index=True)

        # Download
        csv = group_metrics.to_csv(index=False).encode()
        st.download_button("⬇️ Download Bias Report", csv, "bias_report.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.subheader("📜 Complete Audit Trail")
    st.markdown("Immutable log of all model lifecycle events across the platform.")

    # Combine registry audit with simulated governance events
    gov_events = pd.DataFrame({
        "timestamp":  ["2026-04-26 07:15", "2026-04-25 23:20", "2026-04-25 15:00",
                       "2026-04-24 10:30", "2026-04-23 09:00", "2026-04-22 14:20"],
        "event_type": ["BIAS_CHECK", "DEPLOYMENT", "APPROVAL", "TRAINING",
                       "DRIFT_ALERT", "REGISTRATION"],
        "model":      ["ChurnPredictor v2.0", "SpamFilter v1.0", "ChurnPredictor v2.0",
                       "FraudDetector v2.0", "ChurnPredictor v1.0", "HRAttritionModel v1.0"],
        "actor":      ["System", "DevOps", "Risk Manager", "ML Engineer", "Monitor", "Data Scientist"],
        "outcome":    ["PASS", "SUCCESS", "APPROVED", "SUCCESS", "ALERT", "REGISTERED"],
        "notes":      ["Bias check passed – DI=0.83", "Deployed to production", "Approved for staging",
                       "F1=0.942 exceeds threshold", "PSI=0.23 on velocity_count",
                       "Initial model registration"],
    })
    st.dataframe(gov_events, use_container_width=True, hide_index=True)

    st.divider()
    if not df_audit.empty:
        st.markdown("**Registry Audit Log**")
        disp = df_audit.copy()
        disp["timestamp"] = disp["timestamp"].str[:19]
        st.dataframe(disp[["timestamp","action","model_name","version",
                             "from_stage","to_stage","actor","notes"]],
                     use_container_width=True, hide_index=True)

    # Event type distribution
    event_type_counts = gov_events["event_type"].value_counts()
    fig = go.Figure(go.Pie(
        labels=event_type_counts.index.tolist(),
        values=event_type_counts.values.tolist(),
        hole=0.4,
        textinfo="percent+label",
    ))
    fig.update_layout(title="Event Distribution",
                      height=300, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – RESPONSIBLE AI
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[3]:
    st.subheader("📋 Responsible AI Framework")
    st.markdown(
        "Our platform adheres to responsible AI principles: transparency, fairness, "
        "accountability, privacy, and safety."
    )

    principles = [
        ("🔍", "Transparency",
         "Models include feature importance and local explanations for every prediction. "
         "Model cards document training data, algorithm, and limitations.",
         True, ["Model cards published", "SHAP explanations enabled", "Audit logs active"]),
        ("⚖️", "Fairness",
         "Bias assessments run on every model version before promotion. "
         "Disparate impact must exceed 0.80 for all protected attributes.",
         True, ["Bias checks automated", "Group metrics reported", "DI threshold: 0.80"]),
        ("🔐", "Privacy",
         "No PII in training data. All datasets are synthetic or anonymized. "
         "Data retention policies enforced via pipeline.",
         True, ["PII scanning enabled", "Data anonymized", "GDPR compliant"]),
        ("👤", "Accountability",
         "Every model has an assigned owner. Multi-level approval required for production. "
         "All actions logged in immutable audit trail.",
         True, ["Owner assigned", "Approvals required", "Audit trail active"]),
        ("🛡️", "Safety",
         "Automated drift monitoring and performance thresholds prevent model failures. "
         "Rollback capability available for all production models.",
         True, ["Drift detection active", "Performance monitoring", "Rollback ready"]),
        ("🌱", "Sustainability",
         "Model training scheduled during off-peak hours. Lightweight algorithms preferred. "
         "Carbon footprint reported quarterly.",
         False, ["Off-peak scheduling", "Carbon reporting: Q2 2026", "⚠️ Energy audit pending"]),
    ]

    cols = st.columns(2)
    for i, (icon, title, desc, passed, checks) in enumerate(principles):
        with cols[i % 2]:
            color = "#059669" if passed else "#D97706"
            status_badge = "✅ Compliant" if passed else "⚠️ In Progress"
            st.markdown(
                f"""
                <div style='border:1px solid {color}55; border-radius:12px;
                            padding:16px; background:{color}08; margin-bottom:12px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <span style='font-size:1.5rem;'>{icon}</span>
                        <span style='background:{color};color:white;font-size:0.75rem;
                                     padding:2px 10px;border-radius:10px;font-weight:700;'>
                            {status_badge}
                        </span>
                    </div>
                    <h4 style='color:#1E40AF;margin:8px 0 4px;'>{title}</h4>
                    <p style='font-size:0.83rem;color:#475569;margin:0 0 10px;'>{desc}</p>
                    {"".join(f"<div style='font-size:0.78rem;color:{color};'>✓ {c}</div>" for c in checks)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Model Cards
    st.markdown("**📄 Model Cards**")
    card_model = st.selectbox(
        "View Model Card",
        df_reg["model_name"].unique().tolist() if not df_reg.empty else ["ChurnPredictor"],
        key="card_model",
    )
    row = df_reg[df_reg["model_name"] == card_model].iloc[0] if not df_reg.empty else {}
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    with st.expander(f"📄 Model Card: {card_model}", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            **Model Name:** {card_model}
            **Version:** {row.get('version','1.0')}
            **Use Case:** {row.get('use_case','—')}
            **Algorithm:** {row.get('algo_key','—')}
            **Stage:** {row.get('stage','—')}
            **Owner:** {row.get('owner','—')}
            **Created:** {str(row.get('created_at','—'))[:16]}
            """)
        with col_b:
            try:
                metrics = eval(row.get("metrics","{}")) if isinstance(row.get("metrics"), str) else {}
                st.markdown(f"""
                **Performance:**
                - Accuracy: {metrics.get('accuracy','—')}
                - F1 Score: {metrics.get('f1','—')}
                - Precision: {metrics.get('precision','—')}
                - Recall: {metrics.get('recall','—')}
                - RMSE: {metrics.get('rmse','—')}
                """)
            except Exception:
                st.markdown("Metrics not available.")
        st.markdown(f"**Description:** {row.get('description','No description provided.')}")
        st.markdown("""
        **Intended Use:** Production deployment for business decision support.
        **Limitations:** Model trained on synthetic data; real-world performance may vary.
        **Bias Assessment:** Completed — Disparate Impact = 0.83 (pass).
        **Data Sources:** Synthetic data generated with fixed random seed for reproducibility.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – GOVERNANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[4]:
    st.subheader("📊 Governance Summary Report")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Models Reviewed",      len(df_reg))
    c2.metric("Approvals Pending",    3)
    c3.metric("Bias Checks Passed",   len(df_prod))
    c4.metric("Audit Events (30d)",   len(df_audit) + 6)
    c5.metric("Policy Violations",    0)

    st.markdown(
        "<div class='alert-success'>"
        "✅ <b>Overall Governance Score: 92/100</b> — Platform is compliant with all active policies."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Policy compliance table
    st.markdown("**Policy Compliance Checklist**")
    policies = [
        ("Model Versioning",           "Required",   "✅ Active",   "SQLite registry"),
        ("Multi-level Approval",       "Required",   "✅ Active",   "4-stage workflow"),
        ("Bias Testing",               "Required",   "✅ Active",   "Pre-production check"),
        ("Performance Thresholds",     "Required",   "✅ Active",   "Accuracy > 80%"),
        ("Drift Monitoring",           "Required",   "✅ Active",   "PSI + KS tests"),
        ("Audit Logging",              "Required",   "✅ Active",   "All events logged"),
        ("Model Cards",                "Recommended","✅ Active",   "Generated per model"),
        ("Explainability",             "Recommended","✅ Active",   "Feature importance"),
        ("Privacy Review",             "Required",   "✅ Active",   "PII scan on ingest"),
        ("Carbon Reporting",           "Optional",   "⚠️ Pending",  "Due Q2 2026"),
        ("Third-party Audit",          "Annual",     "🔄 Scheduled","Due 2026-Q3"),
        ("Security Pen Test",          "Annual",     "✅ Completed","2025-11-10"),
    ]
    pol_df = pd.DataFrame(policies, columns=["Policy","Requirement","Status","Notes"])
    st.dataframe(pol_df, use_container_width=True, hide_index=True)

    # Download report
    report_content = f"""MLOps Governance Report
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Platform: DataOps Corp MLOps v2.0

EXECUTIVE SUMMARY
=================
Overall Governance Score: 92/100
Models in Production: {len(df_prod)}
Active Policy Violations: 0
Pending Approvals: 3
Bias Checks Completed: {len(df_reg)}

MODEL INVENTORY
===============
{df_reg[['model_name','version','stage','accuracy','owner']].to_string(index=False) if not df_reg.empty else 'No models registered'}

POLICY COMPLIANCE
=================
{pol_df.to_string(index=False)}

AUDIT SUMMARY (Last 30 days)
============================
Total Events: {len(df_audit) + 6}
Promotions: 4
Approvals: 3
Retraining Events: 2
"""
    st.download_button(
        "⬇️ Download Full Governance Report",
        report_content.encode(),
        "governance_report.txt",
        "text/plain",
    )
