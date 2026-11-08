"""run_all.py — Reproduce the whole VerityOps project end to end.

    python run_all.py

Steps: (1) generate synthetic data, (2) train + compare models,
(3) run the 90-day reliability-monitoring simulation.
All outputs land in data/, models/, and reports/.
"""

from __future__ import annotations

import subprocess
import sys


def run(*args: str) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run([sys.executable, "-m", *args], check=True)


def main() -> None:
    run("src.generate_data")
    run("src.features")
    run("src.train")
    run("src.monitor")
    print("\nDone. See reports/model_report.md, reports/monitoring.md, "
          "models/metrics.json, reports/figures/.")


if __name__ == "__main__":
    main()
