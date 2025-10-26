# engine.py
from __future__ import annotations
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator

# ---------- Models ----------

class TriangulationParams(BaseModel):
    sigma_claim: float = Field(..., gt=0)
    sigma_benchmark: float = Field(..., gt=0)
    sigma_literature: float = Field(..., gt=0)
    enforce_feasibility: bool = True


class WorkforceTrainingSpec(BaseModel):
    # Meta
    model_name: str
    country: str = "USA"
    currency: str = "USD"
    base_year: int = 2025
    horizon_years: int = 10
    discount_rate_real: float = 0.03

    # Scale
    participants: int
    completion_rate: float
    placement_rate: float

    # Depth / baseline
    starting_wage_usd: float  # hourly starting wage for placed graduates
    baseline_earnings_cf: float  # annual counterfactual earnings ($)
    employment_share_year1: float  # share of year 1 employed (0-1)
    annual_progression_pct: float  # post wage progression (0-1)

    # Triangulation inputs (annual uplift $)
    uplift_claim_usd: float
    uplift_benchmark_usd: float
    uplift_literature_usd: float
    tri: TriangulationParams

    # Durability
    durability_type: str  # fixed | geometric | exponential
    durability_years: int
    annual_retention: float | None = None
    half_life_years: float | None = None

    # Costs
    program_total_cost: float

    # Simulation
    mc_draws: int = 5000
    seed: int = 42

    @field_validator("completion_rate", "placement_rate", "employment_share_year1", "annual_progression_pct")
    @classmethod
    def bounded_zero_one(cls, v):
        if not (0.0 <= v <= 1.0 or (v == v and v is not None and v >= 0)):  # allow >1 for progression? we cap later
            raise ValueError("Value must be between 0 and 1")
        return float(v)

    @field_validator("durability_type")
    @classmethod
    def durability_ok(cls, v):
        if v not in {"fixed", "geometric", "exponential"}:
            raise ValueError("durability_type must be one of fixed | geometric | exponential")
        return v


# ---------- Persistence ----------

def save_yaml_spec(spec: WorkforceTrainingSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(spec.model_dump(), f, sort_keys=False)


def load_yaml_spec(path: Path) -> WorkforceTrainingSpec:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return WorkforceTrainingSpec(**data)


# ---------- Math Helpers ----------

def blend_uplift_usd(claim: float, bench: float, lit: float, tri: TriangulationParams) -> Tuple[float, float]:
    """
    Precision-weighted average. Variance = sigma^2. Returns (mean, sd).
    Enforces basic feasibility if requested.
    """
    claim = max(0.0, float(claim))
    bench = max(0.0, float(bench))
    lit   = max(0.0, float(lit))

    w_claim = 1.0 / (tri.sigma_claim ** 2)
    w_bench = 1.0 / (tri.sigma_benchmark ** 2)
    w_lit   = 1.0 / (tri.sigma_literature ** 2)

    denom = w_claim + w_bench + w_lit
    if denom <= 0:
        return 0.0, 0.0
    mean = (claim * w_claim + bench * w_bench + lit * w_lit) / denom
    sd = math.sqrt(1.0 / denom)

    if tri.enforce_feasibility:
        mean = max(0.0, mean)
    return mean, sd


def durability_weights(spec: WorkforceTrainingSpec) -> np.ndarray:
    T = spec.horizon_years
    if spec.durability_type == "fixed":
        w = np.ones(T)
        w[int(spec.durability_years):] = 0.0
    elif spec.durability_type == "geometric":
        r = float(spec.annual_retention or 0.8)
        w = np.array([r ** t for t in range(T)])
    else:  # exponential half-life
        hl = float(spec.half_life_years or 3.0)
        lam = math.log(2) / hl
        w = np.array([math.exp(-lam * t) for t in range(T)])
    return w


def discount_factors(rate: float, T: int) -> np.ndarray:
    return np.array([(1.0 / ((1.0 + rate) ** t)) for t in range(1, T + 1)])


# ---------- Simulation ----------

def run_workforce_training_sim(spec: WorkforceTrainingSpec) -> Dict[str, any]:
    rng = np.random.default_rng(spec.seed)

    # People flow
    completed = spec.participants * spec.completion_rate
    placed = completed * spec.placement_rate

    # Triangulated uplift (annual $)
    uplift_mean, uplift_sd = blend_uplift_usd(
        spec.uplift_claim_usd,
        spec.uplift_benchmark_usd,
        spec.uplift_literature_usd,
        spec.tri,
    )

    # Draws: normal around triangulated uplift, truncated at 0
    uplifts = rng.normal(loc=uplift_mean, scale=uplift_sd, size=spec.mc_draws)
    uplifts = np.clip(uplifts, 0, None)

    # Year 1 employment share—can vary slightly
    emp_share_y1 = rng.normal(loc=spec.employment_share_year1, scale=0.03, size=spec.mc_draws)
    emp_share_y1 = np.clip(emp_share_y1, 0.0, 1.0)

    # Wage progression and durability
    prog = np.clip(rng.normal(loc=spec.annual_progression_pct, scale=0.01, size=spec.mc_draws), 0.0, 0.20)
    w_dur = durability_weights(spec)  # length T
    disc = discount_factors(spec.discount_rate_real, spec.horizon_years)  # length T

    # Build annual multipliers for each draw: uplift_t = uplift * (1+prog)^t * durability_t * employment_share (in year 1)
    years = np.arange(spec.horizon_years)
    growth_matrix = (1.0 + prog.reshape(-1, 1)) ** years.reshape(1, -1)  # [draws x T]
    dur_matrix = w_dur.reshape(1, -1)  # [1 x T]
    disc_matrix = disc.reshape(1, -1)  # [1 x T]

    # Employment: scale year1 by emp_share_y1, then relax toward 1 (or keep flat). Simple approach: min(1, y1 + 0.05*t)
    emp_by_year = np.minimum(1.0, emp_share_y1.reshape(-1, 1) + 0.05 * years.reshape(1, -1))

    # Per-participant NPV of earnings gains
    # First compute per placed-person NPV, then multiply by placed/participants.
    uplift_annual = uplifts.reshape(-1, 1) * growth_matrix  # [draws x T]
    delta_path = uplift_annual * dur_matrix * emp_by_year  # apply durability + employment
    npv_per_placed = (delta_path * disc_matrix).sum(axis=1)  # [draws]

    # Scale to all participants:
    share_placed = (placed / spec.participants) if spec.participants > 0 else 0.0
    pp_npv = npv_per_placed * share_placed  # per-intake-participant

    # ROI: total benefits / cost
    total_benefits = pp_npv * spec.participants  # aggregate benefits
    roi = total_benefits / max(spec.program_total_cost, 1e-8)

    # Summaries
    def p(x, q): return float(np.quantile(x, q))
    out = {
        "pp_delta_npv_p50": p(pp_npv, 0.5),
        "pp_delta_npv_p10": p(pp_npv, 0.1),
        "pp_delta_npv_p90": p(pp_npv, 0.9),
        "roi_p50": p(roi, 0.5),
        "roi_p10": p(roi, 0.1),
        "roi_p90": p(roi, 0.9),
        "uplift_blended_mean": float(uplift_mean),
        "uplift_blended_sd": float(uplift_sd),
        "completed_mean": float(completed),
        "placed_mean": float(placed),
    }

    # Simple “tornado” style sensitivities (one-way ±10% on select params)
    drivers = [
        ("Completion rate", "completion_rate", spec.completion_rate),
        ("Placement rate", "placement_rate", spec.placement_rate),
        ("Triangulated uplift", "uplift", uplift_mean),
        ("Durability (retention)", "annual_retention", spec.annual_retention or 0.85),
        ("Progression", "annual_progression_pct", spec.annual_progression_pct),
        ("Discount rate", "discount_rate_real", spec.discount_rate_real),
    ]
    rows = []
    base_pp = out["pp_delta_npv_p50"]
    for label, key, base in drivers:
        for direction, mult in [("Low", 0.9), ("High", 1.1)]:
            spec2 = spec.model_copy(deep=True)
            if key == "uplift":
                # scale all three anchors to simulate ±10% on uplift
                spec2.uplift_claim_usd *= mult
                spec2.uplift_benchmark_usd *= mult
                spec2.uplift_literature_usd *= mult
            elif key in {"completion_rate", "placement_rate", "annual_progression_pct", "annual_retention", "discount_rate_real"}:
                setattr(spec2, key, float(getattr(spec2, key) * mult))
            # run quick sim (fewer draws for speed)
            spec2.mc_draws = max(1000, int(spec.mc_draws / 2))
            res2 = run_workforce_training_sim_quick(spec2)
            rows.append({"Driver": label, "Case": direction, "P50 ΔNPV/pp": res2})

    tornado_df = pd.DataFrame(rows)
    tornado_df["Delta vs Base"] = tornado_df["P50 ΔNPV/pp"] - base_pp

    # Full distribution DataFrame
    dist_df = pd.DataFrame({
        "pp_delta_npv": pp_npv,
        "roi": roi,
        "npv_per_placed": npv_per_placed,
        "uplift_draw": uplifts,
    })

    out["tornado_df"] = tornado_df
    out["dist_df"] = dist_df
    return out


def run_workforce_training_sim_quick(spec: WorkforceTrainingSpec) -> float:
    """Fast variant returning only the median pp ΔNPV for tornado calc."""
    rng = np.random.default_rng(spec.seed + 7)

    # People flow
    completed = spec.participants * spec.completion_rate
    placed = completed * spec.placement_rate

    uplift_mean, uplift_sd = blend_uplift_usd(
        spec.uplift_claim_usd, spec.uplift_benchmark_usd, spec.uplift_literature_usd, spec.tri
    )
    uplifts = rng.normal(loc=uplift_mean, scale=uplift_sd, size=spec.mc_draws)
    uplifts = np.clip(uplifts, 0, None)

    emp_share_y1 = np.clip(rng.normal(loc=spec.employment_share_year1, scale=0.03, size=spec.mc_draws), 0.0, 1.0)
    prog = np.clip(rng.normal(loc=spec.annual_progression_pct, scale=0.01, size=spec.mc_draws), 0.0, 0.20)

    w_dur = durability_weights(spec)
    disc = discount_factors(spec.discount_rate_real, spec.horizon_years)
    years = np.arange(spec.horizon_years)

    growth_matrix = (1.0 + prog.reshape(-1, 1)) ** years.reshape(1, -1)
    dur_matrix = w_dur.reshape(1, -1)
    disc_matrix = disc.reshape(1, -1)
    emp_by_year = np.minimum(1.0, emp_share_y1.reshape(-1, 1) + 0.05 * years.reshape(1, -1))

    uplift_annual = uplifts.reshape(-1, 1) * growth_matrix
    delta_path = uplift_annual * dur_matrix * emp_by_year
    npv_per_placed = (delta_path * disc_matrix).sum(axis=1)

    share_placed = (placed / spec.participants) if spec.participants > 0 else 0.0
    pp_npv = npv_per_placed * share_placed
    return float(np.quantile(pp_npv, 0.5))
