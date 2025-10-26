# app.py
import io
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from engine import (
    WorkforceTrainingSpec,
    TriangulationParams,
    run_workforce_training_sim,
    save_yaml_spec,
    load_yaml_spec,
)

st.set_page_config(page_title="Impact ROI – Workforce Training", page_icon="📈", layout="wide")

# ---------- Sidebar: App Meta ----------
with st.sidebar:
    st.header("⚙️ App")
    specs_dir = Path(st.text_input("Specs folder", value="specs"))
    specs_dir.mkdir(parents=True, exist_ok=True)
    st.caption("Specs are versioned YAML files saved to this folder.")

    st.divider()
    st.header("📂 Load Existing Spec")
    load_choice = st.selectbox(
        "Pick a spec to load",
        options=["(none)"] + [p.name for p in sorted(specs_dir.glob("*.yaml"))],
        index=0,
    )
    loaded_spec = None
    if load_choice and load_choice != "(none)":
        loaded_spec = load_yaml_spec(specs_dir / load_choice)

st.title("Direct Service • Workforce Training – ROI Model")

st.write(
    "This screen captures assumptions for a workforce training program and estimates per-person earnings gains, "
    "program ROI, and sensitivity via Monte Carlo. It also **triangulates** claimed wage uplift with "
    "**market benchmarks** and **literature priors**."
)

# ---------- Initialize Spec (from disk or defaults) ----------
if loaded_spec is None:
    spec = WorkforceTrainingSpec(
        model_name="Example IT Bootcamp 2025",
        country="USA",
        currency="USD",
        base_year=2025,
        horizon_years=10,
        discount_rate_real=0.03,
        # Scale
        participants=500,
        completion_rate=0.80,
        placement_rate=0.65,
        # Depth (baseline & claim)
        starting_wage_usd=22.5,
        baseline_earnings_cf=25000.0,  # counterfactual annual earnings
        employment_share_year1=0.90,
        annual_progression_pct=0.03,
        # Triangulation inputs (annual uplift in USD)
        uplift_claim_usd=12000.0,
        uplift_benchmark_usd=15000.0,
        uplift_literature_usd=10000.0,
        tri=TriangulationParams(
            sigma_claim=20000.0,
            sigma_benchmark=8000.0,
            sigma_literature=6000.0,
            enforce_feasibility=True,
        ),
        # Durability
        durability_type="geometric",  # fixed | geometric | exponential
        durability_years=6,
        annual_retention=0.85,  # for geometric
        half_life_years=None,   # for exponential
        # Program costs
        program_total_cost=2_000_000.0,
        # Simulation
        mc_draws=5000,
        seed=42,
    )
else:
    spec = loaded_spec

# ---------- UI: Basic Info ----------
st.subheader("1) Basic Info")
colA, colB, colC, colD = st.columns(4)
with colA:
    spec.model_name = st.text_input("Model name", value=spec.model_name)
with colB:
    spec.country = st.text_input("Country", value=spec.country)
with colC:
    spec.currency = st.text_input("Currency", value=spec.currency)
with colD:
    spec.base_year = st.number_input("Base year", min_value=1900, max_value=2100, value=spec.base_year, step=1)

colE, colF = st.columns(2)
with colE:
    spec.horizon_years = st.number_input("Horizon (years)", min_value=1, max_value=50, value=spec.horizon_years, step=1)
with colF:
    spec.discount_rate_real = st.number_input("Real discount rate", min_value=0.0, max_value=0.2, value=float(spec.discount_rate_real), step=0.005, format="%.3f")

st.divider()

# ---------- UI: Scale ----------
st.subheader("2) Scale")
col1, col2, col3 = st.columns(3)
with col1:
    spec.participants = st.number_input("Participants (intake)", min_value=0, value=int(spec.participants), step=10)
with col2:
    spec.completion_rate = st.number_input("Completion rate", min_value=0.0, max_value=1.0, value=float(spec.completion_rate), step=0.01)
with col3:
    spec.placement_rate = st.number_input("Placement rate (of completers)", min_value=0.0, max_value=1.0, value=float(spec.placement_rate), step=0.01)

st.divider()

# ---------- UI: Depth (Earnings) ----------
st.subheader("3) Depth – Earnings & Uplift")
col4, col5, col6 = st.columns(3)
with col4:
    spec.starting_wage_usd = st.number_input("Starting hourly wage (USD)", min_value=0.0, value=float(spec.starting_wage_usd), step=0.5)
with col5:
    spec.baseline_earnings_cf = st.number_input("Counterfactual annual earnings (USD)", min_value=0.0, value=float(spec.baseline_earnings_cf), step=500.0)
with col6:
    spec.employment_share_year1 = st.number_input("Employment share in year 1", min_value=0.0, max_value=1.0, value=float(spec.employment_share_year1), step=0.01)

col7, col8 = st.columns(2)
with col7:
    spec.annual_progression_pct = st.number_input("Wage progression per year", min_value=0.0, max_value=0.20, value=float(spec.annual_progression_pct), step=0.005, format="%.3f")
with col8:
    st.caption("Progression applies to the post-program wage path; counterfactual can be extended similarly inside the engine.")

st.markdown("**Triangulated annual uplift (USD)** — engine blends Claim vs. Benchmark vs. Literature by precision weights.")
col9, col10, col11 = st.columns(3)
with col9:
    spec.uplift_claim_usd = st.number_input("Claimed uplift (USD/yr)", min_value=0.0, value=float(spec.uplift_claim_usd), step=1000.0)
with col10:
    spec.uplift_benchmark_usd = st.number_input("Benchmark uplift (USD/yr)", min_value=0.0, value=float(spec.uplift_benchmark_usd), step=1000.0)
with col11:
    spec.uplift_literature_usd = st.number_input("Literature uplift (USD/yr)", min_value=0.0, value=float(spec.uplift_literature_usd), step=1000.0)

col12, col13, col14 = st.columns(3)
with col12:
    spec.tri.sigma_claim = st.number_input("σ Claim (higher = less trust)", min_value=1.0, value=float(spec.tri.sigma_claim), step=500.0)
with col13:
    spec.tri.sigma_benchmark = st.number_input("σ Benchmark", min_value=1.0, value=float(spec.tri.sigma_benchmark), step=250.0)
with col14:
    spec.tri.sigma_literature = st.number_input("σ Literature", min_value=1.0, value=float(spec.tri.sigma_literature), step=250.0)

spec.tri.enforce_feasibility = st.checkbox("Enforce feasibility (cap to non-negative and percentiles)", value=bool(spec.tri.enforce_feasibility))
st.caption("Tip: increase σ for noisier sources; the blend down-weights high-σ inputs automatically.")

st.divider()

# ---------- UI: Durability ----------
st.subheader("4) Durability")
col15, col16, col17 = st.columns(3)
with col15:
    spec.durability_type = st.selectbox("Type", options=["fixed", "geometric", "exponential"], index=["fixed","geometric","exponential"].index(spec.durability_type))
with col16:
    spec.durability_years = st.number_input("Years (if fixed)", min_value=1, max_value=50, value=int(spec.durability_years), step=1)
with col17:
    spec.annual_retention = st.number_input("Annual retention (if geometric)", min_value=0.0, max_value=1.0, value=float(spec.annual_retention), step=0.01)

col18, _ = st.columns(2)
with col18:
    spec.half_life_years = st.number_input("Half-life (if exponential)", min_value=0.1, max_value=50.0, value=float(spec.half_life_years or 3.0), step=0.1)

st.divider()

# ---------- UI: Costs & Simulation ----------
st.subheader("5) Costs & Simulation")
col19, col20, col21 = st.columns(3)
with col19:
    spec.program_total_cost = st.number_input("Program total cost (USD)", min_value=0.0, value=float(spec.program_total_cost), step=5000.0)
with col20:
    spec.mc_draws = st.number_input("Monte Carlo draws", min_value=500, max_value=20000, value=int(spec.mc_draws), step=500)
with col21:
    spec.seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=int(spec.seed), step=1)

st.divider()

# ---------- Actions: Save / Run ----------
col_btn1, col_btn2, col_btn3 = st.columns([1,1,2])
with col_btn1:
    if st.button("💾 Save Spec (YAML)", use_container_width=True):
        out_name = f"{spec.model_name.replace(' ', '_')}.yaml"
        save_yaml_spec(spec, specs_dir / out_name)
        st.success(f"Saved spec → {specs_dir / out_name}")

with col_btn2:
    run_now = st.button("▶️ Run Model", type="primary", use_container_width=True)
with col_btn3:
    st.caption("Saving creates a reproducible record. Running executes Monte Carlo with triangulation and durability.")

# ---------- Results ----------
if run_now:
    results = run_workforce_training_sim(spec)

    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Per-participant Δ Earnings (NPV, $)", f"{results['pp_delta_npv_p50']:,.0f}", f"P10 {results['pp_delta_npv_p10']:,.0f} / P90 {results['pp_delta_npv_p90']:,.0f}")
    m2.metric("ROI (Benefits ÷ Cost)", f"{results['roi_p50']:,.2f}", f"P10 {results['roi_p10']:,.2f} / P90 {results['roi_p90']:,.2f}")
    m3.metric("Triangulated Uplift (USD/yr)", f"{results['uplift_blended_mean']:,.0f}", f"σ {results['uplift_blended_sd']:,.0f}")
    m4.metric("Placed participants (mean)", f"{int(results['placed_mean']):,}", f"Comp. mean {int(results['completed_mean']):,}")

    st.markdown("**Sensitivity Snapshot (top drivers)**")
    st.dataframe(results["tornado_df"])

    st.markdown("**Distributional Outputs**")
    st.area_chart(results["dist_df"][["pp_delta_npv"]])

    st.markdown("**Assumption Snapshot (YAML)**")
    yaml_str = yaml.safe_dump(spec.model_dump(), sort_keys=False)
    st.code(yaml_str, language="yaml")

    # Export CSV
    csv = results["dist_df"].to_csv(index=False).encode("utf-8")
    st.download_button("Download simulation draws (CSV)", data=csv, file_name="simulation_draws.csv", mime="text/csv")
