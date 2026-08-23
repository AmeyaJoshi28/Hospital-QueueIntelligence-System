"""
Step 7 -- Admin dashboard.

Per-run utilization and bottleneck status, SHAP explanations for
current predictions, and a what-if form that calls simulate.py and
shows a before/after comparison via Plotly.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from explain import WaitExplainer
from simulate import fit_distributions, monte_carlo, DATA_PATH

st.set_page_config(page_title="Admin Dashboard", page_icon="📊", layout="wide")
st.title("📊 Hospital Flow — Admin Dashboard")

tab_util, tab_explain, tab_whatif = st.tabs(["Utilization & Bottlenecks", "Prediction Explanations", "What-If Simulator"])


@st.cache_data
def get_utilization():
    return pd.read_csv("outputs/utilization_by_run.csv")


@st.cache_data
def get_raw():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def get_dists():
    return fit_distributions(get_raw())


@st.cache_resource
def get_explainers():
    return WaitExplainer("doc"), WaitExplainer("med")


# ---------------------------------------------------------------- tab 1 --
with tab_util:
    util = get_utilization()
    st.subheader("Utilization (ρ) by scenario")
    fig = px.box(util, x="Scenario", y=["rho_doctor", "rho_pharmacy"],
                 points="outliers", title="Doctor vs pharmacy utilization by scenario")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which stage is the binding bottleneck?")
    bottleneck_counts = util.groupby(["Scenario", "bottleneck"]).size().reset_index(name="n_runs")
    fig2 = px.bar(bottleneck_counts, x="Scenario", y="n_runs", color="bottleneck",
                  title="Bottleneck classification, runs per scenario")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Per-run detail")
    scenario_pick = st.selectbox("Filter by scenario", ["All"] + sorted(util["Scenario"].unique().tolist()))
    view = util if scenario_pick == "All" else util[util["Scenario"] == scenario_pick]
    st.dataframe(view[["RunKey", "Scenario", "NumActiveDoctors", "rho_doctor", "rho_pharmacy", "bottleneck"]],
                 use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- tab 2 --
with tab_explain:
    doc_exp, med_exp = get_explainers()
    doc_test = pd.read_parquet("outputs/test_predictions_doc_wait.parquet")

    st.subheader("Explain a held-out prediction")
    run_pick = st.selectbox("Run", sorted(doc_test["RunKey"].unique()))
    run_rows = doc_test[doc_test["RunKey"] == run_pick]
    patient_pick = st.selectbox("Patient", run_rows["PatientID"].tolist())
    row = run_rows[run_rows["PatientID"] == patient_pick].iloc[0]

    result = doc_exp.explain(row)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted DocWaitTime", f"{result['prediction']:.0f} min")
        st.metric("Actual DocWaitTime", f"{row['DocWaitTime']:.0f} min")
    with col2:
        st.metric("90% interval", f"{result['range'][0]:.0f}-{result['range'][1]:.0f} min")
    st.write(result["sentence"])


# ---------------------------------------------------------------- tab 3 --
with tab_whatif:
    st.subheader("Run a what-if scenario")
    dists = get_dists()

    col1, col2, col3 = st.columns(3)
    with col1:
        n_doc_before = st.number_input("Doctors — current", 1, 6, 2)
    with col2:
        n_doc_after = st.number_input("Doctors — what-if", 1, 6, 3)
    with col3:
        peak_pct = st.slider("Peak arrival rate change (%)", 0, 150, 0)

    n_reps = st.slider("Monte Carlo replications", 50, 300, 100, step=50)

    if st.button("Run simulation", type="primary"):
        with st.spinner(f"Running {n_reps} replications per scenario..."):
            before = monte_carlo(dists, num_doctors=n_doc_before, peak_limit=dists["peak_limit"],
                                  n_replications=n_reps)
            after = monte_carlo(dists, num_doctors=n_doc_after, peak_limit=dists["peak_limit"],
                                 arrival_rate_multiplier=1 + peak_pct / 100, n_replications=n_reps)

        fig = go.Figure()
        fig.add_trace(go.Box(y=before["mean_doc_wait"], name=f"Before (N={n_doc_before})"))
        fig.add_trace(go.Box(y=after["mean_doc_wait"], name=f"After (N={n_doc_after}, +{peak_pct}% peak)"))
        fig.update_layout(title="Mean DocWaitTime per run — before vs after (Monte Carlo distribution)",
                           yaxis_title="mean DocWaitTime (min)")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Before: median mean-DocWait", f"{before['mean_doc_wait'].median():.1f} min")
        c2.metric("After: median mean-DocWait", f"{after['mean_doc_wait'].median():.1f} min",
                   delta=f"{after['mean_doc_wait'].median() - before['mean_doc_wait'].median():.1f} min",
                   delta_color="inverse")
        st.caption("Simulation validated against real held-out scenario data — see README for error rates.")
