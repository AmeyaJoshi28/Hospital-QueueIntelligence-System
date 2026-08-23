"""
Step 4 -- Resource utilization & bottleneck detection.

rho per RunKey and per stage = (total server-busy time) / (run duration x server count).
Doctor stage: c = NumActiveDoctors, busy time = sum(DocServiceTime).
Pharmacy stage: c = 1 (single server), busy time = sum(MedServiceTime).

Bottleneck flag combines two signals, both rule-based (no model):
  - which stage's rho is higher / closer to saturation (>=0.85 is
    treated as "saturated")
  - whether that stage's queue length is trending upward over the
    course of the run (a growing queue is the clearest sign a stage
    can't keep up with arrivals, independent of the average rho)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("outputs/features.parquet")
OUT_DIR = Path("outputs")

SATURATION_THRESHOLD = 0.85


def run_duration(g: pd.DataFrame) -> float:
    start = g["ArrivalTime"].min()
    end = max(g["doc_end"].max(), g["pharmacy_end"].max())
    return float(end - start)


def queue_trend(g: pd.DataFrame, queue_col: str) -> float:
    """Spearman-style trend: correlation between arrival order and queue length.
    Positive => queue tends to grow over the course of the run."""
    g = g.sort_values("ArrivalTime")
    order = np.arange(len(g))
    q = g[queue_col].values
    if q.std() == 0:
        return 0.0
    return float(np.corrcoef(order, q)[0, 1])


def analyze_run(g: pd.DataFrame) -> dict:
    T = run_duration(g)
    c_doc = g["NumActiveDoctors_recovered"].iloc[0]
    doc_busy = g["DocServiceTime"].sum()
    med_busy = g["MedServiceTime"].sum()

    rho_doc = doc_busy / (T * c_doc) if T > 0 else np.nan
    rho_med = med_busy / (T * 1) if T > 0 else np.nan

    doc_trend = queue_trend(g, "queue_for_doctor_at_arrival")
    med_trend = queue_trend(g, "pharmacy_queue_at_doc_end")

    doc_saturated = rho_doc >= SATURATION_THRESHOLD
    med_saturated = rho_med >= SATURATION_THRESHOLD

    if doc_saturated and not med_saturated:
        bottleneck = "doctor"
    elif med_saturated and not doc_saturated:
        bottleneck = "pharmacy"
    elif doc_saturated and med_saturated:
        bottleneck = "doctor" if rho_doc >= rho_med else "pharmacy"
    else:
        # neither formally saturated: use whichever queue is trending up more
        bottleneck = "doctor" if doc_trend >= med_trend else "pharmacy"
        if doc_trend < 0.05 and med_trend < 0.05:
            bottleneck = "none (system stable, no binding constraint)"

    return {
        "RunKey": g["RunKey"].iloc[0],
        "Scenario": g["Scenario"].iloc[0],
        "NumActiveDoctors": int(c_doc),
        "run_duration": T,
        "rho_doctor": float(rho_doc),
        "rho_pharmacy": float(rho_med),
        "doctor_queue_trend": doc_trend,
        "pharmacy_queue_trend": med_trend,
        "bottleneck": bottleneck,
    }


def main():
    df = pd.read_parquet(IN_PATH)
    rows = [analyze_run(g) for _, g in df.groupby("RunKey", sort=False)]
    util = pd.DataFrame(rows)
    util.to_csv(OUT_DIR / "utilization_by_run.csv", index=False)

    print("=== Utilization summary by Scenario ===")
    print(util.groupby("Scenario")[["rho_doctor", "rho_pharmacy"]].mean().round(3))
    print()
    print("=== Bottleneck distribution by Scenario ===")
    print(pd.crosstab(util["Scenario"], util["bottleneck"]))

    summary = {
        "mean_rho_doctor_by_scenario": util.groupby("Scenario")["rho_doctor"].mean().round(3).to_dict(),
        "mean_rho_pharmacy_by_scenario": util.groupby("Scenario")["rho_pharmacy"].mean().round(3).to_dict(),
        "bottleneck_counts_by_scenario": pd.crosstab(util["Scenario"], util["bottleneck"]).to_dict(),
    }
    with open(OUT_DIR / "utilization_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
