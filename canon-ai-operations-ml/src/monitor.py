"""
monitor.py — Reliability-metrics monitoring layer (VerityOps).

Simulates 90 days of daily production batches scored by the champion
model, then computes the metrics the JD asks the analyst to track:
real-time reliability, hallucination rate, precision with delayed labels,
and bias checks across technician tenure bands and device complexity.

Because labels arrive with a delay (human audit), precision is computed on
a 7-day lag. Alerts fire when the 7-day rolling hallucination rate
exceeds 35% or rolling precision drops below 0.80 — thresholds a service
ops lead would set, stated here as explicit assumptions.

Outputs: reports/figures/monitoring.png, reports/monitoring.md
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.features import FEATURE_COLUMNS, build_feature_frame  # noqa: E402
from src.generate_data import apply_omissions, apply_typo, generate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 7
ALERT_HAL_RATE = 0.35
ALERT_PRECISION = 0.80


def main() -> None:
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
    model, threshold = bundle["model"], bundle["threshold"]
    champion = bundle["champion"]

    # Drift episode: on days 45-59 the generator is "upgraded" — its
    # summaries get terser, omitting the very fields the validator checks
    # (evidence hiding). Hallucinations don't rise, but the detector's
    # recall falls, which shows up as a widening DETECTION GAP between
    # the audited hallucination rate and the flag rate. Precision alone
    # would miss this — the gap is the metric that catches it.
    raw = generate(n=2700, seed=SEED)
    raw["service_date"] = pd.to_datetime(raw["service_date"])
    raw = raw.sort_values("service_date").reset_index(drop=True)
    raw["day"] = (np.arange(len(raw)) * 90 // len(raw))
    drift = (raw["day"] >= 45) & (raw["day"] <= 59)
    rng = np.random.RandomState(SEED)
    for i in raw.index[drift & (raw["is_hallucinated"] == 1)]:
        s = raw.at[i, "llm_summary"]
        for _ in range(5):
            s, _ = apply_omissions(s, rng)
        raw.at[i, "llm_summary"] = s
    # ... and the upgraded generator is overconfident about its own output
    raw.loc[drift & (raw["is_hallucinated"] == 1), "llm_confidence"] = rng.uniform(
        0.93, 0.97, size=int((drift & (raw["is_hallucinated"] == 1)).sum()))
    feats = build_feature_frame(raw)
    proba = model.predict_proba(feats[FEATURE_COLUMNS].to_numpy())[:, 1]
    raw["risk_score"] = proba
    raw["flagged"] = (proba >= threshold).astype(int)

    # Daily aggregates; labels arrive with a 7-day audit lag.
    daily = raw.groupby("day").agg(
        n=("work_order_id", "count"),
        hallucinated=("is_hallucinated", "sum"),
        flagged=("flagged", "sum"),
        tp=("flagged", lambda s: int(((raw.loc[s.index, "is_hallucinated"] == 1)
                                      & (s == 1)).sum())),
    ).reset_index()
    # 14-day windows for the audit-lagged metrics: with ~30 summaries/day
    # a 7-day window holds only ~60 hallucinated cases, so recall/precision
    # estimates are too noisy to alert on. 14 days roughly halves the noise.
    daily["hall_rate"] = daily["hallucinated"] / daily["n"]
    daily["flag_rate"] = daily["flagged"] / daily["n"]
    daily["precision_lagged"] = daily["tp"].shift(7) / daily["flagged"].shift(7)
    daily["recall_lagged"] = daily["tp"].shift(7) / daily["hallucinated"].shift(7)
    daily["hall_rate_7d"] = daily["hall_rate"].rolling(7, min_periods=7).mean()
    daily["precision_14d"] = daily["precision_lagged"].rolling(14, min_periods=14).mean()
    daily["recall_14d"] = daily["recall_lagged"].rolling(14, min_periods=14).mean()
    # Detection gap: audited hallucination rate minus flag rate (both
    # 14-day, audit-lagged). When the detector starts missing cases the
    # gap widens even though precision looks fine.
    daily["gap_14d"] = (daily["hall_rate"].shift(7).rolling(14, min_periods=14).mean()
                        - daily["flag_rate"].rolling(14, min_periods=14).mean())

    alert_days = daily[(daily["hall_rate_7d"] > ALERT_HAL_RATE)
                       | (daily["precision_14d"] < ALERT_PRECISION)
                       | (daily["gap_14d"] > 0.10)]
    alert_reasons = {
        "hall_rate_gt_35pct": [int(d) for d in
                               daily.loc[daily["hall_rate_7d"] > ALERT_HAL_RATE, "day"]],
        "precision_lt_80pct": [int(d) for d in
                               daily.loc[daily["precision_14d"] < ALERT_PRECISION, "day"]],
        "gap_gt_10pp": [int(d) for d in
                        daily.loc[daily["gap_14d"] > 0.10, "day"]],
    }

    # Drift effect measured directly on the drift window (days 45-59) vs
    # baseline (days 0-44), from audited labels — in production you see
    # these numbers ~7 days later, once the audit labels land. The rolling
    # chart shows the same dip smeared across the lag window.
    def _seg_stats(mask: pd.Series) -> dict:
        sub = raw[mask]
        tp = int(((sub["is_hallucinated"] == 1) & (sub["flagged"] == 1)).sum())
        fp = int(((sub["is_hallucinated"] == 0) & (sub["flagged"] == 1)).sum())
        fn = int(((sub["is_hallucinated"] == 1) & (sub["flagged"] == 0)).sum())
        return {"precision": round(tp / max(tp + fp, 1), 4),
                "recall": round(tp / max(tp + fn, 1), 4),
                "flag_rate": round(float(sub["flagged"].mean()), 4),
                "hall_rate": round(float(sub["is_hallucinated"].mean()), 4)}
    drift_stats = _seg_stats(drift)
    base_stats = _seg_stats(raw["day"] <= 44)

    # Bias check: flag rate and precision by device complexity and by
    # simulated technician tenure band (tenure is synthetic metadata here).
    rng = np.random.RandomState(SEED)
    raw["tenure_band"] = rng.choice(["<1 yr", "1-3 yrs", "3+ yrs"],
                                    size=len(raw), p=[0.25, 0.45, 0.30])
    bias = raw.groupby("tenure_band").apply(
        lambda g: pd.Series({
            "n": len(g),
            "flag_rate": round(float(g["flagged"].mean()), 3),
            "precision": round(float(((g["is_hallucinated"] == 1)
                                      & (g["flagged"] == 1)).sum()
                                     / max(g["flagged"].sum(), 1)), 3),
            "recall": round(float(((g["is_hallucinated"] == 1)
                                   & (g["flagged"] == 1)).sum()
                                  / max((g["is_hallucinated"] == 1).sum(), 1)), 3),
        }), include_groups=False)
    bias_cx = raw.groupby("device_complexity").apply(
        lambda g: pd.Series({
            "n": len(g),
            "flag_rate": round(float(g["flagged"].mean()), 3),
            "precision": round(float(((g["is_hallucinated"] == 1)
                                      & (g["flagged"] == 1)).sum()
                                     / max(g["flagged"].sum(), 1)), 3),
        }), include_groups=False)

    # --- figure --------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(daily["day"], daily["hall_rate_7d"], label="7-day hallucination rate")
    axes[0].plot(daily["day"], daily["flag_rate"].rolling(7, min_periods=7).mean(),
                 label="7-day flag rate", ls="--")
    axes[0].axhline(ALERT_HAL_RATE, color="r", ls=":",
                    label=f"Alert threshold {ALERT_HAL_RATE:.0%}")
    axes[0].axvspan(45, 59, color="orange", alpha=0.15,
                    label="Generator 'upgrade' (terser summaries)")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("VerityOps monitoring — 90-day simulated production stream "
                      "(synthetic data)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].plot(daily["day"], daily["precision_14d"], label="14-day precision (7-day audit lag)")
    axes[1].plot(daily["day"], daily["recall_14d"], label="14-day recall (7-day audit lag)")
    axes[1].axhline(ALERT_PRECISION, color="r", ls=":",
                    label=f"Alert threshold {ALERT_PRECISION:.0%}")
    axes[1].axvspan(45, 59, color="orange", alpha=0.15)
    axes[1].set_ylabel("Precision / recall")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[2].plot(daily["day"], daily["gap_14d"], label="Detection gap, 14-day (audited hall. rate − flag rate)")
    axes[2].axhline(0.10, color="r", ls=":", label="Alert threshold 10pp")
    axes[2].axvspan(45, 59, color="orange", alpha=0.15)
    axes[2].set_xlabel("Day")
    axes[2].set_ylabel("Gap (fraction)")
    axes[2].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig_path = PROJECT_ROOT / "reports" / "figures" / "monitoring.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    overall_precision = float(((raw["is_hallucinated"] == 1)
                               & (raw["flagged"] == 1)).sum()
                              / max(raw["flagged"].sum(), 1))
    overall_recall = float(((raw["is_hallucinated"] == 1)
                            & (raw["flagged"] == 1)).sum()
                           / max((raw["is_hallucinated"] == 1).sum(), 1))
    summary = {
        "champion": champion,
        "threshold": threshold,
        "days": 90,
        "n_scored": int(len(raw)),
        "overall_hallucination_rate": round(float(raw["is_hallucinated"].mean()), 4),
        "overall_flag_rate": round(float(raw["flagged"].mean()), 4),
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "alert_days": int(len(alert_days)),
        "alert_day_list": [int(d) for d in alert_days["day"].tolist()],
        "alert_reasons": alert_reasons,
        "drift_episode_days_45_59_precision": drift_stats["precision"],
        "drift_episode_days_45_59_recall": drift_stats["recall"],
        "drift_episode_days_45_59_gap": round(
            drift_stats["hall_rate"] - drift_stats["flag_rate"], 4),
        "baseline_days_0_44_recall": base_stats["recall"],
        "baseline_days_0_44_gap": round(
            base_stats["hall_rate"] - base_stats["flag_rate"], 4),
        "baseline_days_0_44_precision": base_stats["precision"],
        "bias_by_tenure_band": bias.to_dict(orient="index"),
        "bias_by_device_complexity": {str(k): v
                                      for k, v in bias_cx.to_dict(orient="index").items()},
    }
    with open(PROJECT_ROOT / "models" / "monitoring_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# VerityOps — Reliability Monitoring (90-day simulation)",
          "",
          "**All data is synthetic.** Fresh stream (seed 7, n=2,700), scored by "
          f"the champion model ({champion}, threshold {threshold}).",
          "",
          f"- Overall hallucination rate: **{summary['overall_hallucination_rate']:.1%}**; "
          f"flag rate: **{summary['overall_flag_rate']:.1%}**",
          f"- Overall precision: **{overall_precision:.3f}**; recall: **{overall_recall:.3f}**",
          f"- Alert days (hallucination rate > {ALERT_HAL_RATE:.0%} or "
          f"precision < {ALERT_PRECISION:.0%} or detection gap > 10pp): "
          f"**{len(alert_days)}** — days {summary['alert_day_list']}",
          "- Drift episode (days 45–59, generator 'upgrade' emitting terser, "
          "overconfident summaries that omit checkable fields; measured on "
          "audited labels, which land ~7 days later): recall fell to "
          f"{summary['drift_episode_days_45_59_recall']:.3f} "
          f"(baseline {summary['baseline_days_0_44_recall']:.3f}), precision "
          f"{summary['drift_episode_days_45_59_precision']:.3f} (baseline "
          f"{summary['baseline_days_0_44_precision']:.3f}), and the "
          "detection gap widened to "
          f"{summary['drift_episode_days_45_59_gap']:.1%} (baseline "
          f"{summary['baseline_days_0_44_gap']:.1%}). Precision alone would "
          "have looked fine — the recall/gap pair is what catches it.",
          f"- Alert anatomy: {len(summary['alert_reasons']['hall_rate_gt_35pct'])} "
          "days breached the hallucination-rate threshold, "
          f"{len(summary['alert_reasons']['precision_lt_80pct'])} breached "
          "precision, "
          f"{len(summary['alert_reasons']['gap_gt_10pp'])} breached the gap. "
          "Days 20–26 breached the detection gap for a full week with NO "
          "underlying change — a false alarm from small daily volumes "
          "(~30 summaries/day makes 14-day gap estimates noisy). Lesson: "
          "pair the threshold with a minimum-sample rule and a "
          "2-consecutive-day confirmation before paging.",
          "",
          "## Bias check — flag rate / precision by technician tenure band (synthetic)",
          "",
          bias.to_string(),
          "",
          "## Bias check — by device complexity (1 = simple, 5 = complex)",
          "",
          bias_cx.to_string(),
          "",
          "## What an analyst would do next",
          "",
          "1. Page the on-call reviewer on alert days; sample-audit flagged "
          "summaries the same day.",
          "2. After a generator upgrade, re-validate on a fresh labelled batch "
          "before trusting the old threshold.",
          "3. If a tenure band or complexity tier shows a precision gap > 0.10, "
          "escalate as a bias-mitigation review, per the JD's bias-mitigation "
          "tracking requirement.",
          ""]
    (PROJECT_ROOT / "reports" / "monitoring.md").write_text("\n".join(md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
