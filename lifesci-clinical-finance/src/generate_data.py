"""
generate_data.py — SYNTHETIC data generator.

ALL DATA IS SYNTHETIC. No real clinical-program, pharma, or biotech data
is used anywhere in this project. Values are drawn from illustrative,
invented distributions chosen to exercise the forecasting / valuation
pipeline — they are NOT estimates of any real program, company, or drug.
Phase success probabilities below are ILLUSTRATIVE ASSUMPTIONS for the
simulation only, not published industry rates and not real-world claims.

Outputs:
  data/synthetic/clinical_programs.csv   (240 synthetic programs)
  data/synthetic/monthly_burn.csv        (synthetic monthly spend, 60 programs x 24 months)
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
N_PROGRAMS = 240

# Illustrative phase success assumptions (SYNTHETIC — for simulation only).
PHASE_POS = {"Preclinical": 0.35, "Phase I": 0.60, "Phase II": 0.35, "Phase III": 0.60}
PHASE_TO_APPROVAL = {"Preclinical": 0.08, "Phase I": 0.12, "Phase II": 0.22, "Phase III": 0.55}

THERAPEUTIC_AREAS = ["Oncology", "Neurology", "Immunology", "Cardiology",
                     "Rare Disease", "Infectious Disease"]
MODALITIES = ["Small Molecule", "Monoclonal Antibody", "Cell Therapy",
              "Gene Therapy", "RNA / Oligonucleotide", "Vaccine"]
PHASES = ["Preclinical", "Phase I", "Phase II", "Phase III"]


def generate_programs(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_PROGRAMS):
        area = rng.choice(THERAPEUTIC_AREAS)
        modality = rng.choice(MODALITIES)
        phase = rng.choice(PHASES, p=[0.20, 0.25, 0.30, 0.25])
        complexity = int(rng.integers(1, 6))
        unmet_need = int(rng.integers(1, 6))
        competitive = int(rng.integers(1, 6))
        # Enrollment / sites scale with phase (synthetic ranges).
        base_enroll = {"Preclinical": 0, "Phase I": 40, "Phase II": 220,
                       "Phase III": 900}[phase]
        enrollment = int(max(0, rng.normal(base_enroll, base_enroll * 0.35))) if base_enroll else 0
        sites = int(max(1, enrollment / rng.uniform(8, 20))) if enrollment else int(rng.integers(1, 4))
        duration = int(rng.normal({"Preclinical": 18, "Phase I": 20,
                                   "Phase II": 30, "Phase III": 42}[phase], 6))
        duration = max(6, duration)
        monthly_burn = float(np.round(rng.lognormal(mean=np.log(1.2), sigma=0.55), 3))  # $M / month
        peak_sales = float(np.round(rng.lognormal(mean=np.log(420), sigma=0.9), 1))     # $M / yr
        cost_to_complete = float(np.round(monthly_burn * duration * rng.uniform(0.85, 1.25), 2))
        discount = float(rng.choice([0.08, 0.10, 0.12]))
        # Risk adjustment (illustrative formula, fully disclosed in README).
        pos_next = PHASE_POS[phase] * (0.9 + 0.05 * unmet_need - 0.03 * competitive
                                       + 0.02 * (complexity - 3))
        pos_next = float(np.clip(pos_next, 0.05, 0.95))
        pos_approval = PHASE_TO_APPROVAL[phase]
        # rNPV ($M): risk-adjusted value of peak-sales stream (illustrative
        # 0.35 margin proxy, 8-year horizon, discounted) minus discounted costs.
        pv_sales = peak_sales * 0.35 * sum(1 / (1 + discount) ** t for t in range(1, 9))
        pv_costs = cost_to_complete / (1 + discount) ** (duration / 12 / 2)
        rnpv = float(np.round(pos_approval * pv_sales - pv_costs, 2))
        # Next-phase transition outcome, drawn from pos_next (synthetic label).
        transition_success = int(rng.random() < pos_next)
        rows.append({
            "program_id": f"SYN-P{ i + 1:03d}",
            "therapeutic_area": area,
            "modality": modality,
            "phase": phase,
            "indication_complexity_1_5": complexity,
            "unmet_need_1_5": unmet_need,
            "competitive_intensity_1_5": competitive,
            "enrollment_target": enrollment,
            "sites": sites,
            "duration_months": duration,
            "monthly_burn_usd_m": monthly_burn,
            "peak_sales_usd_m": peak_sales,
            "cost_to_complete_usd_m": cost_to_complete,
            "discount_rate": discount,
            "pos_next_phase_illustrative": round(pos_next, 3),
            "pos_to_approval_illustrative": pos_approval,
            "rnpv_usd_m": rnpv,
            "transition_success": transition_success,
            "is_synthetic": True,
        })
    return pd.DataFrame(rows)


def generate_monthly_burn(programs: pd.DataFrame, seed: int = RNG_SEED + 1) -> pd.DataFrame:
    """Synthetic monthly planned vs actual spend for a 60-program subset."""
    rng = np.random.default_rng(seed)
    subset = programs.sample(60, random_state=seed).reset_index(drop=True)
    rows = []
    for _, p in subset.iterrows():
        base = p["monthly_burn_usd_m"]
        for month in range(1, 25):
            # Planned: smooth ramp; actual: planned x lognormal noise + occasional spike.
            planned = base * (0.7 + 0.3 * month / 24)
            actual = planned * float(rng.lognormal(mean=0.0, sigma=0.18))
            if rng.random() < 0.05:
                actual *= float(rng.uniform(1.4, 1.8))  # synthetic cost spike
            rows.append({
                "program_id": p["program_id"],
                "phase": p["phase"],
                "therapeutic_area": p["therapeutic_area"],
                "month": month,
                "planned_spend_usd_m": round(float(planned), 3),
                "actual_spend_usd_m": round(float(actual), 3),
                "is_synthetic": True,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    programs = generate_programs()
    burn = generate_monthly_burn(programs)
    programs.to_csv(out / "clinical_programs.csv", index=False)
    burn.to_csv(out / "monthly_burn.csv", index=False)
    print(f"programs: {programs.shape}, burn rows: {burn.shape}")
    print(programs.head(3).to_string())
