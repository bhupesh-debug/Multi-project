"""feature_engineering.py — interpretable features for finance users.

Every feature is explainable in one sentence to a non-ML hiring manager.
No leakage: rNPV inputs that are algebraic components of the target
(peak sales, cost-to-complete) are kept ONLY as log-transformed drivers
and are flagged in the report — the classification model excludes them.
"""
import numpy as np
import pandas as pd

PHASE_ORDER = {"Preclinical": 0, "Phase I": 1, "Phase II": 2, "Phase III": 3}

REGRESSION_FEATURES = [
    "phase_ordinal", "indication_complexity_1_5", "unmet_need_1_5",
    "competitive_intensity_1_5", "enrollment_target", "sites",
    "duration_months", "monthly_burn_usd_m", "log_peak_sales",
    "log_cost_to_complete", "discount_rate", "pos_next_phase_illustrative",
    "pos_to_approval_illustrative", "cost_per_patient_usd_m",
    "burn_x_duration", "patients_per_site", "unmet_minus_competitive",
]
# Classification excludes the rNPV algebra inputs (peak sales / cost) to
# avoid leakage and to test whether operational features predict transition.
CLASSIFICATION_FEATURES = [
    "phase_ordinal", "indication_complexity_1_5", "unmet_need_1_5",
    "competitive_intensity_1_5", "enrollment_target", "sites",
    "duration_months", "monthly_burn_usd_m", "discount_rate",
    "pos_next_phase_illustrative", "cost_per_patient_usd_m",
    "burn_x_duration", "patients_per_site", "unmet_minus_competitive",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phase_ordinal"] = df["phase"].map(PHASE_ORDER)
    df["log_peak_sales"] = np.log1p(df["peak_sales_usd_m"])
    df["log_cost_to_complete"] = np.log1p(df["cost_to_complete_usd_m"])
    df["burn_x_duration"] = df["monthly_burn_usd_m"] * df["duration_months"]
    df["patients_per_site"] = df.apply(
        lambda r: r["enrollment_target"] / r["sites"] if r["sites"] else 0, axis=1)
    df["unmet_minus_competitive"] = df["unmet_need_1_5"] - df["competitive_intensity_1_5"]
    # One-hot therapeutic area + modality (kept separate; added at model time).
    dummies = pd.get_dummies(df[["therapeutic_area", "modality"]],
                             prefix=["area", "mod"], dtype=float)
    return pd.concat([df, dummies], axis=1)


def feature_matrix(df: pd.DataFrame, kind: str = "regression"):
    base = REGRESSION_FEATURES if kind == "regression" else CLASSIFICATION_FEATURES
    dummy_cols = [c for c in df.columns if c.startswith(("area_", "mod_"))]
    cols = base + dummy_cols
    return df[cols].copy(), cols
