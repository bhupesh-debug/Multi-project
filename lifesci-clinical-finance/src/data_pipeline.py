"""data_pipeline.py — load, validate, and clean the SYNTHETIC datasets."""
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_programs(path: str | pathlib.Path | None = None) -> pd.DataFrame:
    path = path or ROOT / "data" / "synthetic" / "clinical_programs.csv"
    df = pd.read_csv(path)
    assert df["is_synthetic"].all(), "Dataset must be flagged synthetic"
    assert df["program_id"].is_unique, "program_id must be unique"
    assert df.isna().sum().sum() == 0, "No missing values expected in synthetic data"
    assert (df["rnpv_usd_m"].notna()).all()
    return df


def load_burn(path: str | pathlib.Path | None = None) -> pd.DataFrame:
    path = path or ROOT / "data" / "synthetic" / "monthly_burn.csv"
    df = pd.read_csv(path)
    assert df["is_synthetic"].all(), "Dataset must be flagged synthetic"
    return df


def clean_programs(df: pd.DataFrame) -> pd.DataFrame:
    """Type fixes + derived accounting fields (no rows dropped — synthetic, complete)."""
    df = df.copy()
    df["cost_per_patient_usd_m"] = df.apply(
        lambda r: r["cost_to_complete_usd_m"] / r["enrollment_target"]
        if r["enrollment_target"] > 0 else 0.0, axis=1)
    df["total_planned_cost_usd_m"] = df["monthly_burn_usd_m"] * df["duration_months"]
    return df


if __name__ == "__main__":
    df = clean_programs(load_programs())
    burn = load_burn()
    print(df.shape, burn.shape)
    print(df.describe(include="all").T[["count", "mean", "std", "min", "max"]].to_string())
