"""Anomaly detection on monthly variance signals.

Two layers, both explainable (a management report must say WHY a month
was flagged, not just that a black box fired):
1. Rule flags — material variance thresholds from variance.py.
2. Isolation Forest on variance features (revenue var %, cost var %,
   AUM var %, margin change) — unsupervised, catches joint anomalies
   no single threshold sees. A month is an ANOMALY when the model and a
   firm rule agree, OR when any single cost line is material (a line-level
   spike can hide inside a small firm-total variance); model-only hits
   are reported as WATCH.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def build_features(firm_var: pd.DataFrame, firm: pd.DataFrame) -> pd.DataFrame:
    df = firm_var.copy()
    df["margin_change_pp"] = (df["operating_margin_actual"] - df["operating_margin_budget"]) * 100
    feats = df[["month", "revenue_variance_pct", "cost_variance_pct",
                "aum_actual_m", "margin_change_pp"]].copy()
    feats["aum_mom_pct"] = feats["aum_actual_m"].pct_change().fillna(0)
    return feats


def detect_anomalies(firm_var: pd.DataFrame, firm: pd.DataFrame,
                     contamination: float = 0.12, seed: int = 42) -> pd.DataFrame:
    feats = build_features(firm_var, firm)
    X = feats[["revenue_variance_pct", "cost_variance_pct",
               "aum_mom_pct", "margin_change_pp"]].fillna(0).values
    model = IsolationForest(contamination=contamination, random_state=seed)
    pred = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    score = model.decision_function(X)
    # Line-level materiality: a single cost line can spike (e.g. a vendor
    # bill) while total firm cost variance stays below the firm threshold.
    from src.data_generation import COST_LINES
    line_material = []
    for _, r in firm.iterrows():
        hit = False
        for line in COST_LINES:
            key = line.lower().replace(" ", "_")
            b, a = r[f"cost_{key}_budget"], r[f"cost_{key}_actual"]
            if b and abs(a - b) / b >= 0.10 and abs(a - b) >= 50_000:
                hit = True
        line_material.append(hit)
    out = feats[["month"]].copy()
    out["rule_flag"] = (firm_var["revenue_material"] | firm_var["cost_material"]).values
    out["line_material_flag"] = line_material
    out["model_anomaly"] = pred == -1
    out["anomaly_score"] = np.round(score, 3)
    out["revenue_variance_pct"] = firm_var["revenue_variance_pct"].values
    out["cost_variance_pct"] = firm_var["cost_variance_pct"].values
    # ANOMALY: model + firm rule agree, OR any single cost line is material
    # (a line-level spike is itself reportable to management).
    out["verdict"] = np.where((out["model_anomaly"] & out["rule_flag"]) | out["line_material_flag"],
                              "ANOMALY",
                              np.where(out["model_anomaly"], "WATCH", "Normal"))
    return out
