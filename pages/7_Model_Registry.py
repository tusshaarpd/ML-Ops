"""
Model Registry — view, compare, promote, roll back, and archive models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go

from config import CUSTOM_CSS, STAGES, USE_CASES
from utils.model_registry import (
    get_all_models, promote_model, delete_model, compare_versions,
    get_audit_log, get_production_models,
)
from utils.visualizations import metric_comparison_bar, PALETTE

st.set_page_config(page_title="Model Registry", page_icon="🗂️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:8px 0 12px;'>
            <div style='font-size:1.8rem;'>🗂️</div>
            <div style='color:#93c5fd; font-weight:700; font-size:0.95rem;'>Model Registry</div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.8rem; color:#cbd5e1; line-height:1.8;'>
        <b style='color:#bfdbfe;'>📌 Five Tabs</b><br>
        📋 <b style='color:#e2e8f0;'>All Models</b> — Filter &amp; browse full registry<br>
        🔬 <b style='color:#e2e8f0;'>Version Comparison</b> — Compare metrics across versions<br>
        ⚙️ <b style='color:#e2e8f0;'>Actions</b> — Promote, archive, rollback<br>
        📜 <b style='color:#e2e8f0;'>Audit Log</b> — Full event history<br>
        📊 <b style='color:#e2e8f0;'>Analytics</b> — Registry statistics
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        <div style='font-size:0.78rem; color:#94a3b8; line-height:1.6;'>
        <b style='color:#bfdbfe;'>🔄 Stage Lifecycle</b><br>
        Development → Staging → Production → Archived<br><br>
        <b style='color:#bfdbfe;'>💾 Storage</b><br>
        SQLite database at <code style='background:rgba(255,255,255,0.1);
        color:#93c5fd; padding:1px 5px; border-radius:3px;'>db/mlops_registry.db</code>
        </div>
        """, unsafe_allow_html=True,
    )

st.title("🗂️ Model Registry")
st.caption("Version control, stage management, and audit trail for all ML models.")
st.markdown(
    """
    <div class='explain-box'>
    <b>📖 About this page:</b> The Model Registry is the single source of truth for all
    trained models. Every model is stored with its version number, performance metrics,
    training parameters, owner, and lifecycle stage.
    Models progress through <b>Development → Staging → Production → Archived</b>.
    The registry is backed by a <b>SQLite database</b> that persists across sessions.
    Use the <b>Actions</b> tab to promote or roll back models, and the <b>Audit Log</b>
    to see a complete history of every change made.
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()


def badge(stage: str) -> str:
    colors = {
        "Production": "#059669",
        "Staging":    "#D97706",
        "Development":"#1E40AF",
        "Archived":   "#DC2626",
    }
    c = colors.get(stage, "#6B7280")
    return (f"<span style='background:{c};color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:0.78rem;font-weight:700;'>{stage}</span>")


# ── Load Registry ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_registry():
    return get_all_models()

@st.cache_data(ttl=5)
def load_audit():
    return get_audit_log()

df_reg  = load_registry()
df_audit = load_audit()

main_tabs = st.tabs([
    "📋 All Models",
    "🔬 Version Comparison",
    "⚙️ Actions",
    "📜 Audit Log",
    "📊 Analytics",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[0]:
    st.subheader("📋 Registered Models")

    # Filters
    f1, f2, f3 = st.columns(3)
    stage_filter   = f1.multiselect("Filter by Stage",   STAGES,   default=STAGES)
    use_case_filter= f2.multiselect("Filter by Use Case",
                                     list(USE_CASES.keys()),
                                     default=list(USE_CASES.keys()))
    search_term    = f3.text_input("Search model name", "")

    filtered = df_reg.copy()
    if stage_filter:
        filtered = filtered[filtered["stage"].isin(stage_filter)]
    if use_case_filter:
        filtered = filtered[filtered["use_case"].isin(use_case_filter)]
    if search_term:
        filtered = filtered[filtered["model_name"].str.contains(search_term, case=False)]

    st.caption(f"Showing {len(filtered)} / {len(df_reg)} models")

    # Display table with badges
    for _, row in filtered.iterrows():
        with st.container():
            r1, r2, r3, r4, r5, r6, r7 = st.columns([2, 0.7, 1.2, 1, 1, 1.2, 1.5])
            r1.markdown(f"**{row['model_name']}**")
            r2.markdown(f"v{row['version']}")
            r3.markdown(badge(row["stage"]), unsafe_allow_html=True)
            r4.markdown(f"*{row['use_case']}*")
            acc_str = f"{row['accuracy']:.3f}" if pd.notna(row['accuracy']) else "—"
            f1_str  = f"{row['f1']:.3f}"       if pd.notna(row['f1'])       else "—"
            rmse_str= f"{row['rmse']:.1f}"     if pd.notna(row['rmse'])     else "—"
            r5.markdown(f"Acc:{acc_str} F1:{f1_str}")
            r6.markdown(f"👤 {row['owner']}")
            r7.markdown(f"🗓 {str(row['updated_at'])[:16]}")
        st.markdown("<hr style='margin:4px 0; border-color:#F3F4F6;'>", unsafe_allow_html=True)

    # Download
    csv = filtered.to_csv(index=False).encode()
    st.download_button("⬇️ Export Registry CSV", csv, "model_registry.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – VERSION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.subheader("🔬 Version Comparison")
    all_model_names = df_reg["model_name"].unique().tolist()
    sel_model = st.selectbox("Select Model", all_model_names)

    if sel_model:
        versions_df = compare_versions(sel_model)
        if not versions_df.empty:
            st.markdown(f"**{sel_model}** — {len(versions_df)} version(s)")
            # Parse metrics JSON
            metric_rows = []
            for _, row in versions_df.iterrows():
                try:
                    m = json.loads(row["metrics"]) if row["metrics"] else {}
                except Exception:
                    m = {}
                metric_rows.append({
                    "Version": row["version"],
                    "Stage":   row["stage"],
                    "Accuracy": m.get("accuracy", row["accuracy"]),
                    "F1":       m.get("f1",       row["f1"]),
                    "Precision":m.get("precision"),
                    "Recall":   m.get("recall"),
                    "RMSE":     m.get("rmse",     row["rmse"]),
                    "Created":  str(row["created_at"])[:16] if row["created_at"] else "—",
                })
            metric_df = pd.DataFrame(metric_rows)

            # Version table
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

            # Comparison chart
            chart_metrics = {}
            for _, mr in metric_df.iterrows():
                ver_key = f"v{mr['Version']}"
                cls_metrics = {}
                for col in ["Accuracy","F1","Precision","Recall"]:
                    if pd.notna(mr.get(col)):
                        cls_metrics[col] = float(mr[col])
                if cls_metrics:
                    chart_metrics[ver_key] = cls_metrics

            if len(chart_metrics) > 1:
                fig = metric_comparison_bar(chart_metrics,
                                             f"Version Comparison — {sel_model}")
                st.plotly_chart(fig, use_container_width=True)

            # Highlight best version
            acc_col = metric_df["Accuracy"].dropna()
            if not acc_col.empty:
                best_ver = metric_df.loc[acc_col.idxmax(), "Version"]
                best_acc = acc_col.max()
                st.success(f"🏆 Best version: **v{best_ver}** (Accuracy={best_acc:.4f})")
        else:
            st.info(f"No version history found for {sel_model}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.subheader("⚙️ Model Actions")
    st.markdown("Promote to staging/production, roll back, or archive models.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Promote / Change Stage**")
        action_model   = st.selectbox("Model", df_reg["model_name"].unique(), key="act_model")
        versions_avail = df_reg[df_reg["model_name"] == action_model]["version"].tolist()
        action_version = st.selectbox("Version", versions_avail, key="act_version")
        current_stage  = df_reg[
            (df_reg["model_name"] == action_model) &
            (df_reg["version"] == action_version)
        ]["stage"].values
        current_stage_val = current_stage[0] if len(current_stage) else "Unknown"
        st.markdown(f"Current Stage: {badge(current_stage_val)}", unsafe_allow_html=True)

        target_stage = st.selectbox(
            "Promote to Stage",
            [s for s in STAGES if s != current_stage_val],
        )
        actor = st.text_input("Your Name", "ML Engineer", key="act_actor")
        notes = st.text_area("Notes", "Promoted after A/B test validation", height=60, key="act_notes")

        promote_btn = st.button("✅ Apply Stage Change", type="primary", key="promote_btn")
        if promote_btn:
            try:
                promote_model(action_model, action_version, target_stage, actor, notes)
                st.success(
                    f"✅ **{action_model} v{action_version}** promoted to **{target_stage}**"
                )
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Promotion failed: {e}")

    with col_b:
        st.markdown("**Archive / Delete**")
        arch_model   = st.selectbox("Model", df_reg["model_name"].unique(), key="arch_model")
        arch_versions= df_reg[df_reg["model_name"] == arch_model]["version"].tolist()
        arch_version = st.selectbox("Version", arch_versions, key="arch_version")
        arch_actor   = st.text_input("Your Name", "Admin", key="arch_actor")

        col_b1, col_b2 = st.columns(2)
        archive_btn = col_b1.button("📦 Archive", key="archive_btn")
        delete_btn  = col_b2.button("🗑️ Delete",  key="delete_btn", type="primary")

        if archive_btn:
            try:
                promote_model(arch_model, arch_version, "Archived", arch_actor, "Archived via UI")
                st.success(f"Model {arch_model} v{arch_version} archived.")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Archive failed: {e}")

        if delete_btn:
            if st.session_state.get("confirm_delete"):
                try:
                    delete_model(arch_model, arch_version, arch_actor)
                    st.success(f"Model {arch_model} v{arch_version} deleted.")
                    st.session_state["confirm_delete"] = False
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
            else:
                st.session_state["confirm_delete"] = True
                st.warning("⚠️ Click Delete again to confirm deletion. This is irreversible.")

    st.divider()

    # Rollback
    st.markdown("**Rollback Production Model**")
    rollback_uc = st.selectbox("Use Case", list(USE_CASES.keys()), key="rb_uc")
    prod_versions = df_reg[
        (df_reg["use_case"] == rollback_uc) &
        (df_reg["stage"].isin(["Production","Staging"]))
    ][["model_name","version","stage","accuracy"]].copy()

    if not prod_versions.empty:
        st.dataframe(prod_versions, use_container_width=True, hide_index=True)
        rb_model  = st.selectbox("Select model to roll back from Production",
                                  prod_versions[prod_versions["stage"]=="Production"]["model_name"].tolist()
                                  if "Production" in prod_versions["stage"].values else [],
                                  key="rb_model")
        rb_version= st.selectbox("Target rollback version",
                                  prod_versions["version"].tolist(), key="rb_version")
        if st.button("↩️ Execute Rollback", key="rb_btn"):
            st.warning(f"Rollback initiated: {rb_model} v{rb_version} → Production (requires approval)")
    else:
        st.info("No Production/Staging models for this use case.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[3]:
    st.subheader("📜 Audit Log")
    if not df_audit.empty:
        disp_audit = df_audit.copy()
        disp_audit["timestamp"] = disp_audit["timestamp"].str[:19]
        st.dataframe(disp_audit[["timestamp","action","model_name","version",
                                  "from_stage","to_stage","actor","notes"]],
                     use_container_width=True, hide_index=True)

        # Action breakdown chart
        action_counts = df_audit["action"].value_counts()
        fig = go.Figure(go.Bar(
            x=action_counts.index.tolist(),
            y=action_counts.values.tolist(),
            marker_color=PALETTE[:len(action_counts)],
            text=action_counts.values.tolist(),
            textposition="outside",
        ))
        fig.update_layout(title="Actions in Audit Log",
                          height=280, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No audit events yet. Actions on models will appear here.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with main_tabs[4]:
    st.subheader("📊 Registry Analytics")
    if df_reg.empty:
        st.info("No models in registry.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Models",      len(df_reg))
    c2.metric("Production Models", len(df_reg[df_reg["stage"]=="Production"]))
    c3.metric("Avg Accuracy",
              f"{df_reg['accuracy'].dropna().mean():.3f}"
              if not df_reg["accuracy"].dropna().empty else "—")

    col_a, col_b = st.columns(2)
    with col_a:
        stage_counts = df_reg["stage"].value_counts()
        fig = go.Figure(go.Pie(
            labels=stage_counts.index.tolist(),
            values=stage_counts.values.tolist(),
            hole=0.40,
            marker=dict(colors=["#059669","#D97706","#1E40AF","#DC2626"]),
            textinfo="percent+label",
        ))
        fig.update_layout(title="Models by Stage",
                          height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        uc_counts = df_reg["use_case"].value_counts()
        fig = go.Figure(go.Bar(
            x=uc_counts.index.tolist(),
            y=uc_counts.values.tolist(),
            marker_color=PALETTE[:len(uc_counts)],
            text=uc_counts.values.tolist(),
            textposition="outside",
        ))
        fig.update_layout(title="Models by Use Case",
                          height=300, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Accuracy scatter
    acc_df = df_reg.dropna(subset=["accuracy"])
    if not acc_df.empty:
        fig = go.Figure(go.Scatter(
            x=acc_df["model_name"] + " v" + acc_df["version"],
            y=acc_df["accuracy"],
            mode="markers",
            marker=dict(
                size=12,
                color=acc_df["accuracy"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Accuracy"),
            ),
            text=acc_df["stage"],
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.3f}<br>Stage: %{text}<extra></extra>",
        ))
        fig.update_layout(title="Model Accuracy Across All Versions",
                          xaxis_tickangle=-45, height=350,
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(range=[0.7, 1.0]))
        st.plotly_chart(fig, use_container_width=True)
