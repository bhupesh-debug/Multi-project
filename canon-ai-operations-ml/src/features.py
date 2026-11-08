"""
features.py — Feature engineering for the VerityOps hallucination/risk
classifier.

Design principle: every feature is computable in production from (a) the
structured source work order and (b) the LLM-generated summary text, plus
the generator's self-reported confidence. NOTHING here uses the label,
the corruption-type tags, or any information that would not exist at
scoring time. The corruption_types column exists only for analysis.

Feature families
----------------
1. Grounding checks — does the summary agree with the source record?
   (part code, date, technician, serial, device, SLA, cost, hours)
2. Text-shape signals — length, hedge/certainty language, entity overlap.
3. Generator metadata — self-reported confidence, model version, source
   completeness, device complexity.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

HEDGE_WORDS = ["approximately", "around", "roughly", "likely", "possibly",
               "maybe", "presumably", "seems", "appears"]
CERTAINTY_WORDS = ["confirmed", "verified", "completed", "definitely",
                   "exactly"]

MONEY_RE = re.compile(r"\$([\d,]+\.\d{2})")
HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hours")
PART_RE = re.compile(r"FM\d-[A-Z]\d{3}")
SERIAL_RE = re.compile(r"CN\d{6}[A-Z]")
DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December) (\d{1,2}), (\d{4})")

FEATURE_COLUMNS = [
    "num_match_rate", "num_mismatch_count", "part_code_match", "date_match",
    "tech_name_match", "serial_match", "device_match", "sla_match",
    "part_stated", "date_stated", "serial_stated", "sla_stated",
    "fields_stated_count",
    "cost_stated", "cost_delta_pct", "hours_delta", "hours_stated",
    "summary_len_words", "len_ratio", "hedge_count", "certainty_count",
    "entity_overlap", "invented_name_count", "next_step_match",
    "sentence_count", "llm_confidence", "source_completeness",
    "device_complexity", "is_v24",
]

TECH_SURNAMES = ["Okafor", "Rivera", "Patel", "Kim", "Novak", "Brooks",
                 "Chen", "Alvarez", "Tanaka", "Doyle", "Haddad", "Fitzgerald"]


def _numbers_in(text: str) -> set[float]:
    nums = set()
    for m in MONEY_RE.findall(text):
        nums.add(round(float(m.replace(",", "")), 2))
    for m in HOURS_RE.findall(text):
        nums.add(round(float(m), 1))
    return nums


def extract_features(row: pd.Series) -> dict:
    summary = str(row["llm_summary"])
    words = summary.split()
    sentences = [s for s in re.split(r"[.!?]\s+", summary) if s.strip()]

    # --- grounding: numbers -------------------------------------------------
    summary_nums = _numbers_in(summary)
    true_nums = {round(float(row["total_cost"]), 2),
                 round(float(row["parts_cost"]), 2),
                 round(float(row["labor_hours"]), 1)}
    matched = summary_nums & true_nums
    num_match_rate = len(matched) / max(len(summary_nums), 1)
    num_mismatch_count = len(summary_nums - true_nums)

    # --- grounding: entities ------------------------------------------------
    # "stated" vs "match" matters: a field the summary never mentions is
    # missing evidence, not evidence of error. The model must learn that
    # distinction instead of treating every absent field as a failure
    # (which is exactly why the naive rules baseline over-flags).
    part_stated = int(bool(PART_RE.search(summary)))
    date_stated = int(bool(DATE_RE.search(summary)))
    serial_stated = int(bool(SERIAL_RE.search(summary)))
    sla_stated = int("SLA status:" in summary)
    fields_stated_count = (part_stated + date_stated + serial_stated
                           + sla_stated + int(bool(HOURS_RE.search(summary)))
                           + int(bool(MONEY_RE.search(summary))))
    part_code_match = int(str(row["part_code"]) in summary)
    svc_date = pd.to_datetime(str(row["service_date"]))
    date_str_long = svc_date.strftime("%B %d, %Y")
    date_str_nopad = svc_date.strftime("%B ") + str(svc_date.day) + svc_date.strftime(", %Y")
    date_match = int(date_str_long in summary or date_str_nopad in summary)
    tech_surname = str(row["technician"]).split(". ")[-1]
    tech_name_match = int(tech_surname in summary)
    serial_match = int(str(row["serial_no"]) in summary)
    device_match = int(str(row["device_model"]) in summary)
    sla_match = int(f"SLA status: {row['sla_status']}" in summary)

    # --- grounding: cost / hours -------------------------------------------
    money = [float(m.replace(",", "")) for m in MONEY_RE.findall(summary)]
    cost_stated = int(len(money) > 0)
    if money:
        # the largest money figure is the stated total
        stated_total = max(money)
        cost_delta_pct = abs(stated_total - float(row["total_cost"])) / float(row["total_cost"])
    else:
        cost_delta_pct = 0.0
    hours_found = [float(h) for h in HOURS_RE.findall(summary)]
    hours_stated = int(len(hours_found) > 0)
    hours_delta = abs(hours_found[0] - float(row["labor_hours"])) if hours_found else 0.0

    # --- text shape ---------------------------------------------------------
    summary_len_words = len(words)
    source_words = 60  # approx. words in a structured work-order record
    len_ratio = summary_len_words / source_words
    low = summary.lower()
    hedge_count = sum(low.count(w) for w in HEDGE_WORDS)
    certainty_count = sum(low.count(w) for w in CERTAINTY_WORDS)
    caps = set(re.findall(r"\b[A-Z][a-z]+\b", summary))
    source_caps = set(re.findall(r"\b[A-Z][a-z]+\b",
                                 f"{row['customer']} {row['device_model']} {row['technician']}"))
    entity_overlap = len(caps & source_caps) / max(len(caps | source_caps), 1)
    invented_name_count = sum(1 for s in TECH_SURNAMES
                              if s != tech_surname and s in summary)
    next_step_match = int(any(
        kw in summary for kw in
        ["No further action", "Monitor print quality", "follow-up visit",
         "spare feed roller", "firmware update", "network settings"]))
    sentence_count = len(sentences)

    return {
        "num_match_rate": round(num_match_rate, 3),
        "num_mismatch_count": num_mismatch_count,
        "part_code_match": part_code_match,
        "date_match": date_match,
        "tech_name_match": tech_name_match,
        "serial_match": serial_match,
        "device_match": device_match,
        "sla_match": sla_match,
        "part_stated": part_stated,
        "date_stated": date_stated,
        "serial_stated": serial_stated,
        "sla_stated": sla_stated,
        "fields_stated_count": fields_stated_count,
        "cost_stated": cost_stated,
        "cost_delta_pct": round(cost_delta_pct, 4),
        "hours_delta": round(hours_delta, 2),
        "hours_stated": hours_stated,
        "summary_len_words": summary_len_words,
        "len_ratio": round(len_ratio, 3),
        "hedge_count": hedge_count,
        "certainty_count": certainty_count,
        "entity_overlap": round(entity_overlap, 3),
        "invented_name_count": invented_name_count,
        "next_step_match": next_step_match,
        "sentence_count": sentence_count,
        "llm_confidence": float(row["llm_confidence"]),
        "source_completeness": float(row["source_completeness"]),
        "device_complexity": int(row["device_complexity"]),
        "is_v24": int(str(row["llm_model_version"]).endswith("v2.4")),
    }


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.apply(extract_features, axis=1, result_type="expand")
    keep = ["work_order_id", "service_date", "is_hallucinated",
            "corruption_types", "n_corruptions", "omitted_fields", "has_typo"]
    keep = [c for c in keep if c in df.columns]
    return pd.concat([df[keep].reset_index(drop=True), feats], axis=1)


if __name__ == "__main__":
    from pathlib import Path
    raw = pd.read_csv(Path(__file__).resolve().parents[1]
                      / "data" / "synthetic" / "field_service_summaries.csv")
    X = build_feature_frame(raw)
    print(X.shape)
    print(X[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
