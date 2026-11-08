"""
generate_data.py — Synthetic data generator for the VerityOps hallucination/risk
classifier (Canon AI Operations & Data Analyst ML project).

ALL DATA IS SYNTHETIC. No real Canon data, customers, technicians, serial
numbers, or service records are used. Every name, company, device serial,
part code, cost, and date below is invented by this script with a fixed
random seed so the dataset is fully reproducible.

What is generated
-----------------
One row per field-service work order:
  * Ground-truth structured fields (device, part, technician, date, hours,
    costs, SLA status, next step) — the "source of truth" the validator
    checks the LLM summary against.
  * llm_summary — a synthetic LLM-generated summary of the visit.
  * is_hallucinated — 1 if the summary contains one or more factual
    corruptions vs. the ground truth, else 0.
  * corruption_types — which corruption(s) were injected (analysis only;
    NEVER used as a model feature).
  * llm_confidence — simulated self-reported confidence of the generator.

Usage:
    python -m src.generate_data            # writes data/synthetic/*.csv
    python -m src.generate_data --n 2400 --seed 42
"""

from __future__ import annotations

import argparse
import random
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Synthetic vocabularies (all invented)
# ---------------------------------------------------------------------------

DEVICE_MODELS = [
    "imageRUNNER ADVANCE DX C5870i",
    "imageRUNNER ADVANCE DX C5860i",
    "imageRUNNER ADVANCE DX 6780i",
    "imageRUNNER ADVANCE DX C259iF",
    "imagePRESS C910",
    "imagePRESS Lite C170",
    "imageCLASS MF465dw",
    "imageFORMULA DR-C240",
]

PARTS = [
    ("FM1-A101", "fuser unit"),
    ("FM1-B204", "drum unit"),
    ("FM1-C317", "toner cartridge (black)"),
    ("FM1-D428", "transfer belt"),
    ("FM1-E539", "feed roller kit"),
    ("FM1-F640", "ADF pickup roller"),
    ("FM1-G751", "main controller board"),
    ("FM1-H862", "fixing film assembly"),
    ("FM1-J973", "waste toner container"),
    ("FM1-K084", "developer unit"),
]

TECHNICIANS = [
    "M. Okafor", "J. Rivera", "S. Patel", "D. Kim", "A. Novak",
    "T. Brooks", "L. Chen", "R. Alvarez", "K. Tanaka", "P. Doyle",
    "H. Haddad", "E. Fitzgerald",
]

CUSTOMERS = [
    "Riverside Dental Group", "Harborview Legal LLP", "Northgate Realty",
    "Beacon Hill School District", "Cedar Park Clinic", "Summit Logistics",
    "Maplewood Accounting", "Bayview Imaging Center", "Ironworks Fabrication",
    "Willow Creek College", "Pinnacle Insurance Group", "Foxglove Marketing",
]

ISSUE_TYPES = [
    "paper jam in the ADF", "faded print output", "error code E0007",
    "toner streaking on copies", "slow print queue", "scanner not responding",
    "frequent misfeeds from tray 2", "network connectivity drop",
    "unusual grinding noise", "colour calibration drift",
]

NEXT_STEPS = [
    "No further action required; unit returned to service.",
    "Monitor print quality over the next 50 jobs and report any recurrence.",
    "Schedule a follow-up visit in 30 days to inspect the replacement part.",
    "Order a spare feed roller kit to keep on site.",
    "Recommend a firmware update at the next scheduled maintenance.",
    "Customer to confirm network settings with their IT team.",
]

FAKE_NEXT_STEPS = [
    "A full replacement unit has been ordered and will arrive within 48 hours.",
    "The customer has been credited $1,500 for the downtime.",
    "A senior engineer will visit tomorrow to replace the main board.",
    "The service contract has been upgraded to same-day response at no charge.",
]

# ---------------------------------------------------------------------------
# Summary templates (faithful versions; corruptions are applied afterwards)
# ---------------------------------------------------------------------------

TEMPLATES = [
    ("Service visit completed on {date_str} by technician {tech} for {customer}. "
     "The {device} (serial {serial}) reported {issue}. The technician replaced "
     "the {part_name} (part {part_code}). Labour: {hours} hours. "
     "Parts cost ${parts_cost:.2f}; total cost ${total_cost:.2f}. "
     "SLA status: {sla}. {next_step}"),
    ("Technician {tech} serviced the {device} at {customer} on {date_str}. "
     "Issue: {issue}. Resolution: replaced the {part_name} (part {part_code}). "
     "The visit took {hours} hours. Total charge ${total_cost:.2f} "
     "(parts ${parts_cost:.2f}). SLA status: {sla}. {next_step}"),
    ("On {date_str}, {tech} completed a field service visit for {customer}. "
     "Device: {device}, serial {serial}. The reported problem was {issue}; "
     "the {part_name} (part {part_code}) was replaced. Labour {hours} hours, "
     "total cost ${total_cost:.2f}. SLA status: {sla}. {next_step}"),
]

HEDGE_PHRASES = ["approximately ", "around ", "roughly ", "likely "]
CERTAINTY_PHRASES = ["confirmed ", "verified "]


MONEY_RE = re.compile(r"\$([\d,]+\.\d{2})")


def fmt_date(d: date) -> str:
    return d.strftime("%B %d, %Y")  # e.g. "October 14, 2026"


def apply_omissions(summary: str, rng: random.Random) -> tuple[str, list[str]]:
    """Realistic LLM behaviour: a summary often OMITS a field entirely.
    An omitted field cannot be checked — so a hallucination hiding in an
    omitted field is genuinely undetectable from the summary, and a
    faithful summary with omissions looks suspicious. This is what makes
    the classification problem non-trivial (no single check is decisive).
    Each omission fires independently with p=0.10, max two per summary."""
    candidates = [
        ("part_code", re.compile(r" \(part FM\d-[A-Z]\d{3}\)")),
        ("serial", re.compile(r" \(serial CN\d{6}[A-Z]\)")),
        ("sla", re.compile(r"SLA status: (met|breached)\. ")),
        ("cost", re.compile(r"Parts cost \$[\d,]+\.\d{2}; total cost \$[\d,]+\.\d{2}\. ")),
        ("cost2", re.compile(r"Total charge \$[\d,]+\.\d{2} \(parts \$[\d,]+\.\d{2}\)\. ")),
        ("cost3", re.compile(r"total cost \$[\d,]+\.\d{2}\. ")),
    ]
    omitted: list[str] = []
    out = summary
    rng.shuffle(candidates)
    for name, pattern in candidates:
        if len(omitted) >= 2:
            break
        if rng.random() < 0.15 and pattern.search(out):
            out = pattern.sub(" ", out, count=1)
            omitted.append(name.rstrip("23"))
    return re.sub(r"\s+", " ", out).strip(), omitted


def apply_typo(summary: str, rng: random.Random) -> tuple[str, bool]:
    """Transposing two adjacent digits in one money figure — OCR/typing
    noise in an otherwise faithful summary. Looks like a mismatch but is
    NOT a hallucination; the model must weigh evidence, not trust any
    single check."""
    m = list(MONEY_RE.finditer(summary))
    if not m:
        return summary, False
    match = rng.choice(m)
    digits = list(match.group(1).replace(",", ""))
    if len(digits) < 4:
        return summary, False
    i = rng.randint(0, len(digits) - 2)
    if digits[i] == digits[i + 1]:
        return summary, False
    digits[i], digits[i + 1] = digits[i + 1], digits[i]
    new_val = "".join(digits)
    new_text = summary[:match.start(1)] + f"{float(new_val):,.2f}" + summary[match.end(1):]
    return new_text, True


def make_serial(rng: random.Random) -> str:
    return f"CN{rng.randint(100000, 999999)}{rng.choice('ABCDEFGH')}"


def corrupt_summary(text: str, truth: dict, rng: random.Random,
                    n_corrupt: int, serial_in_text: bool,
                    pool: list[str] | None = None) -> tuple[str, list[str]]:
    """Inject n factual corruptions into a faithful summary. Returns the
    corrupted text and the list of corruption types applied."""
    corruptions = rng.sample(
        [c for c in (pool or ["wrong_part_code", "wrong_date", "invented_cost",
                              "wrong_technician", "fabricated_next_step",
                              "inflated_hours", "wrong_sla", "wrong_device",
                              "invented_serial", "subtle_issue_swap",
                              "wrong_part_name"])
         # invented_serial is only a real corruption when the summary
         # template actually prints the serial (templates 1 & 3); in
         # template 2 the serial never appears, so swapping it would
         # create an undetectable label — a labelling error, not a
         # model failure. Exclude it there.
         if not (c == "invented_serial" and not serial_in_text)],
        k=min(n_corrupt, 11),
    )
    applied: list[str] = []
    out = text
    for c in corruptions:
        if c == "wrong_part_code":
            other = rng.choice([p for p in PARTS if p[0] != truth["part_code"]])
            out = out.replace(truth["part_code"], other[0])
        elif c == "wrong_date":
            shifted = truth["service_date"] + timedelta(days=rng.choice([-9, -4, 3, 7, 12]))
            out = out.replace(fmt_date(truth["service_date"]), fmt_date(shifted))
        elif c == "invented_cost":
            fake_total = round(truth["total_cost"] * rng.uniform(1.6, 3.2), 2)
            out = re.sub(r"total cost \$[\d,]+\.\d{2}",
                         f"total cost ${fake_total:,.2f}", out, count=1)
            out = re.sub(r"Total charge \$[\d,]+\.\d{2}",
                         f"Total charge ${fake_total:,.2f}", out, count=1)
        elif c == "wrong_technician":
            other = rng.choice([t for t in TECHNICIANS if t != truth["technician"]])
            out = out.replace(truth["technician"], other)
        elif c == "fabricated_next_step":
            out = re.sub(r"(SLA status: (met|breached)\. ).*?$",
                         r"\1" + rng.choice(FAKE_NEXT_STEPS), out)
        elif c == "inflated_hours":
            fake_hours = round(truth["labor_hours"] * rng.uniform(1.8, 2.6), 1)
            out = re.sub(r"(\d+(?:\.\d+)?) hours", f"{fake_hours} hours", out, count=1)
            out = re.sub(r"Labour (\d+(?:\.\d+)?) hours",
                         f"Labour {fake_hours} hours", out, count=1)
        elif c == "wrong_sla":
            flipped = "breached" if truth["sla_status"] == "met" else "met"
            out = out.replace(f"SLA status: {truth['sla_status']}",
                              f"SLA status: {flipped}")
        elif c == "wrong_device":
            other = rng.choice([d for d in DEVICE_MODELS if d != truth["device_model"]])
            out = out.replace(truth["device_model"], other)
        elif c == "invented_serial":
            out = out.replace(truth["serial_no"], make_serial(rng))
        elif c == "subtle_issue_swap":
            # The reported problem is restated as a different, plausible
            # problem. No structured field contradicts it — this is the
            # genuinely hard case: only a human (or the source record's
            # free-text notes) could catch it.
            other = rng.choice([i for i in ISSUE_TYPES if i != truth["issue_type"]])
            out = out.replace(truth["issue_type"], other)
        elif c == "wrong_part_name":
            # Right part code, wrong part name — the code check passes.
            other = rng.choice([p for p in PARTS if p[0] != truth["part_code"]])
            out = out.replace(truth["part_name"], other[1])
        applied.append(c)
    return out, applied


def generate(n: int = 2400, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    start = date(2025, 1, 6)
    for i in range(n):
        service_date = start + timedelta(days=rng.randint(0, 660))
        part_code, part_name = rng.choice(PARTS)
        parts_cost = round(rng.uniform(45, 620), 2)
        labor_hours = round(rng.uniform(0.5, 4.5), 1)
        labor_rate = 145.0
        total_cost = round(parts_cost + labor_hours * labor_rate, 2)
        tech = rng.choice(TECHNICIANS)
        truth = {
            "work_order_id": f"WO-{100000 + i}",
            "service_date": service_date,
            "device_model": rng.choice(DEVICE_MODELS),
            "serial_no": make_serial(rng),
            "technician": tech,
            "customer": rng.choice(CUSTOMERS),
            "issue_type": rng.choice(ISSUE_TYPES),
            "part_code": part_code,
            "part_name": part_name,
            "labor_hours": labor_hours,
            "parts_cost": parts_cost,
            "total_cost": total_cost,
            "sla_status": rng.choices(["met", "breached"], weights=[0.86, 0.14])[0],
            "next_step": rng.choice(NEXT_STEPS),
            "source_completeness": round(rng.uniform(0.85, 1.0), 2),
            "device_complexity": rng.randint(1, 5),
        }
        template = rng.choice(TEMPLATES)
        template_idx = TEMPLATES.index(template)
        serial_in_text = template_idx in (0, 2)  # templates 1 & 3 print {serial}
        summary = template.format(
            date_str=fmt_date(service_date), tech=tech,
            customer=truth["customer"], device=truth["device_model"],
            serial=truth["serial_no"], issue=truth["issue_type"],
            part_name=part_name, part_code=part_code, hours=labor_hours,
            parts_cost=parts_cost, total_cost=total_cost,
            sla=truth["sla_status"], next_step=truth["next_step"],
        )
        # ~8% of summaries get a hedge phrase (more common in hallucinated ones)
        is_hallucinated = rng.random() < 0.30
        corruption_types: list[str] = []
        if is_hallucinated:
            if rng.random() < 0.30:
                # Subtle-only hallucination: nothing in the structured
                # record contradicts the summary. These are the model's
                # irreducible blind spot (~30% of hallucinations) and the
                # reason human audit can never be fully removed.
                summary, corruption_types = corrupt_summary(
                    summary, truth, rng, 1, serial_in_text,
                    pool=["subtle_issue_swap", "wrong_part_name"])
            else:
                n_corrupt = rng.choices([1, 2, 3], weights=[0.55, 0.35, 0.10])[0]
                summary, corruption_types = corrupt_summary(
                    summary, truth, rng, n_corrupt, serial_in_text)
            if rng.random() < 0.35:
                summary = summary.replace("The technician replaced",
                                          "The technician " + rng.choice(HEDGE_PHRASES) + "replaced", 1)
        else:
            if rng.random() < 0.12:
                summary = summary.replace("The technician replaced",
                                          "The technician " + rng.choice(CERTAINTY_PHRASES) + "replaced", 1)
        # --- realistic noise: omissions + typos (applied AFTER corruption,
        # so a corrupted field can be hidden by omission) ------------------
        omitted: list[str] = []
        summary, omitted = apply_omissions(summary, rng)
        has_typo = False
        if not is_hallucinated and rng.random() < 0.06:
            summary, has_typo = apply_typo(summary, rng)
        # Simulated generator self-confidence: hallucinated outputs skew
        # only slightly lower and overlap heavily (realistic — overconfident
        # hallucinations are common; confidence alone must never decide).
        if is_hallucinated:
            confidence = min(0.99, max(0.50, rng.gauss(0.81, 0.10)))
        else:
            confidence = min(0.99, max(0.55, rng.gauss(0.86, 0.09)))
        rows.append({
            **truth,
            "service_date": service_date.isoformat(),
            "llm_summary": summary,
            "is_hallucinated": int(is_hallucinated),
            "corruption_types": ";".join(corruption_types),
            "n_corruptions": len(corruption_types),
            "omitted_fields": ";".join(omitted),
            "has_typo": int(has_typo),
            "llm_confidence": round(confidence, 3),
            "llm_model_version": rng.choice(["verity-llm v2.3", "verity-llm v2.4"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate(n=args.n, seed=args.seed)
    out = OUT_DIR / "field_service_summaries.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
    print(f"Hallucination prevalence: {df['is_hallucinated'].mean():.1%}")
    print(f"Date range: {df['service_date'].min()} to {df['service_date'].max()}")
    print("Corruption mix (hallucinated rows only):")
    print(df.loc[df["is_hallucinated"] == 1, "corruption_types"]
          .str.split(";").explode().value_counts().to_string())


if __name__ == "__main__":
    main()
