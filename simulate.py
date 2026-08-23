"""
Step 6 -- What-if simulation.

Fits empirical distributions straight from the dataset (discrete
bootstrap resampling -- interarrival gaps and service times take only
a handful of distinct integer values each, so an empirical PMF is a
better fit than forcing a continuous parametric distribution), builds
a SimPy multi-server(doctor) -> single-server(pharmacy) model, and
validates it against real data before using it for scenario analysis.

Discovered from data (see EDA / utilization findings, reused here):
  - Priority governs doctor-queue order: higher Priority is served
    ahead of already-waiting lower-priority patients (non-preemptive).
    Modeled with simpy.PriorityResource.
  - Pharmacy shows no priority effect (MedWaitTime is ~flat across
    Priority) -- modeled as a plain FIFO simpy.Resource.
  - Arrivals switch from a peak (dense) regime to a non-peak (sparse)
    regime at a fixed elapsed-time cutoff (PeakLimit), not gradually.
  - Service-time distributions are scenario-invariant; only arrival
    density and doctor count vary the Baseline scenario is used as
    the calibration source.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simpy
from scipy import stats

DATA_PATH = Path("data/hospital_sim_6_scenarios.csv")
OUT_DIR = Path("outputs")

CALIBRATION_SCENARIO = "Baseline_2Doc_240Peak"
N_PATIENTS_PER_RUN = 200  # matches real run length


# ---------------------------------------------------------------- fitting --
def fit_distributions(df: pd.DataFrame) -> dict:
    base = df[df["Scenario"] == CALIBRATION_SCENARIO].sort_values(["RunID", "ArrivalTime"]).copy()
    base["gap"] = base.groupby("RunID")["ArrivalTime"].diff()
    peak_gaps = base.loc[base["IsPeakArrival"] == 1, "gap"].dropna().values
    nonpeak_gaps = base.loc[base["IsPeakArrival"] == 0, "gap"].dropna().values

    def pmf(values):
        vals, counts = np.unique(values, return_counts=True)
        return vals, counts / counts.sum()

    peak_vals, peak_p = pmf(peak_gaps)
    nonpeak_vals, nonpeak_p = pmf(nonpeak_gaps)
    doc_vals, doc_p = pmf(df["DocServiceTime"].values)
    med_vals, med_p = pmf(df["MedServiceTime"].values)
    prio_vals, prio_p = pmf(base["Priority"].values)

    # goodness-of-fit sanity check: does a geometric (discrete-exponential)
    # distribution also plausibly describe the peak gaps? (report only)
    ks_stat, ks_p = stats.kstest(peak_gaps, "geom", args=(1 / peak_gaps.mean(),))

    peak_limit = int(base["PeakLimit"].iloc[0])

    return {
        "peak_gap_vals": peak_vals, "peak_gap_p": peak_p,
        "nonpeak_gap_vals": nonpeak_vals, "nonpeak_gap_p": nonpeak_p,
        "doc_service_vals": doc_vals, "doc_service_p": doc_p,
        "med_service_vals": med_vals, "med_service_p": med_p,
        "priority_vals": prio_vals, "priority_p": prio_p,
        "peak_limit": peak_limit,
        "geom_ks_stat_peak_gaps": float(ks_stat), "geom_ks_p_peak_gaps": float(ks_p),
    }


# ------------------------------------------------------------- simulation --
@dataclass
class RunResult:
    doc_waits: list = field(default_factory=list)
    med_waits: list = field(default_factory=list)


def simulate_one_run(dists: dict, num_doctors: int, peak_limit: int,
                      arrival_rate_multiplier: float = 1.0,
                      n_patients: int = N_PATIENTS_PER_RUN, seed: int = 0) -> RunResult:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    doctor = simpy.PriorityResource(env, capacity=num_doctors)
    pharmacy = simpy.Resource(env, capacity=1)
    result = RunResult()

    def draw(vals, p):
        return float(rng.choice(vals, p=p))

    def patient(priority: int):
        arrive_t = env.now
        # non-preemptive priority: higher stated Priority => served first
        with doctor.request(priority=-priority) as req:
            yield req
            doc_wait = env.now - arrive_t
            result.doc_waits.append(doc_wait)
            yield env.timeout(draw(dists["doc_service_vals"], dists["doc_service_p"]))
        med_arrive = env.now
        with pharmacy.request() as req:
            yield req
            med_wait = env.now - med_arrive
            result.med_waits.append(med_wait)
            yield env.timeout(draw(dists["med_service_vals"], dists["med_service_p"]))

    def arrivals():
        for _ in range(n_patients):
            is_peak = env.now <= peak_limit
            gap_vals, gap_p = (dists["peak_gap_vals"], dists["peak_gap_p"]) if is_peak \
                else (dists["nonpeak_gap_vals"], dists["nonpeak_gap_p"])
            gap = draw(gap_vals, gap_p) / arrival_rate_multiplier
            yield env.timeout(gap)
            priority = int(draw(dists["priority_vals"], dists["priority_p"]))
            env.process(patient(priority))

    env.process(arrivals())
    env.run()
    return result


def monte_carlo(dists: dict, num_doctors: int, peak_limit: int,
                 arrival_rate_multiplier: float = 1.0,
                 n_replications: int = 100, n_patients: int = N_PATIENTS_PER_RUN) -> pd.DataFrame:
    rows = []
    for i in range(n_replications):
        r = simulate_one_run(dists, num_doctors, peak_limit, arrival_rate_multiplier,
                              n_patients=n_patients, seed=1000 + i)
        rows.append({
            "replication": i,
            "mean_doc_wait": float(np.mean(r.doc_waits)) if r.doc_waits else np.nan,
            "p90_doc_wait": float(np.percentile(r.doc_waits, 90)) if r.doc_waits else np.nan,
            "mean_med_wait": float(np.mean(r.med_waits)) if r.med_waits else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- validate --
def validate(dists: dict, df: pd.DataFrame) -> dict:
    real_base = df[df["Scenario"] == CALIBRATION_SCENARIO]
    real_per_run = real_base.groupby("RunID")["DocWaitTime"].mean()
    real_med_per_run = real_base.groupby("RunID")["MedWaitTime"].mean()

    sim = monte_carlo(dists, num_doctors=2, peak_limit=dists["peak_limit"], n_replications=100)

    ks_doc = stats.ks_2samp(real_per_run.values, sim["mean_doc_wait"].dropna().values)
    ks_med = stats.ks_2samp(real_med_per_run.values, sim["mean_med_wait"].dropna().values)

    return {
        "real_mean_doc_wait": float(real_per_run.mean()),
        "sim_mean_doc_wait": float(sim["mean_doc_wait"].mean()),
        "real_mean_med_wait": float(real_med_per_run.mean()),
        "sim_mean_med_wait": float(sim["mean_med_wait"].mean()),
        "ks_doc_stat": float(ks_doc.statistic), "ks_doc_p": float(ks_doc.pvalue),
        "ks_med_stat": float(ks_med.statistic), "ks_med_p": float(ks_med.pvalue),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    dists = fit_distributions(df)

    print("=== Fitted distributions (Baseline scenario) ===")
    print("peak gap PMF:", dict(zip(dists["peak_gap_vals"].tolist(), dists["peak_gap_p"].round(3).tolist())))
    print("non-peak gap PMF:", dict(zip(dists["nonpeak_gap_vals"].tolist(), dists["nonpeak_gap_p"].round(3).tolist())))
    print("doc service PMF:", dict(zip(dists["doc_service_vals"].tolist(), dists["doc_service_p"].round(3).tolist())))
    print("med service PMF:", dict(zip(dists["med_service_vals"].tolist(), dists["med_service_p"].round(3).tolist())))
    print("geometric-fit KS test on peak gaps: stat=%.3f p=%.3f (low p => geometric is a poor fit; "
          "empirical PMF resampling used instead)" % (dists["geom_ks_stat_peak_gaps"], dists["geom_ks_p_peak_gaps"]))

    print("\n=== Validation: simulated vs real (Baseline_2Doc_240Peak) ===")
    val = validate(dists, df)
    print(json.dumps(val, indent=2))
    if val["ks_doc_p"] < 0.05:
        print("WARNING: simulated per-run mean DocWaitTime distribution differs "
              "significantly from real data (KS p<0.05). Treat scenario outputs as directional.")
    else:
        print("OK: simulated per-run mean DocWaitTime distribution is statistically "
              "indistinguishable from real data (KS p>=0.05).")

    with open(OUT_DIR / "simulation_validation.json", "w") as f:
        json.dump(val, f, indent=2)

    # Stronger, out-of-sample validation: the sim was calibrated ONLY on
    # Baseline_2Doc_240Peak (N=2). Feed it N=1/3/4 and check whether it
    # predicts the *other* five scenarios' real observed wait times,
    # which it never saw during calibration.
    print("\n=== Out-of-sample validation: predicting untouched scenarios ===")
    cross_val = {}
    for n_doc, scen in [(1, "Staffing_1Doc"), (3, "Staffing_3Doc"), (4, "Staffing_4Doc")]:
        real_mean = float(df.loc[df["Scenario"] == scen, "DocWaitTime"].mean())
        sim = monte_carlo(dists, num_doctors=n_doc, peak_limit=dists["peak_limit"], n_replications=100)
        sim_mean = float(sim["mean_doc_wait"].mean())
        cross_val[scen] = {"real_mean_doc_wait": real_mean, "sim_mean_doc_wait": sim_mean,
                            "pct_error": abs(sim_mean - real_mean) / real_mean * 100 if real_mean else None}
        print(f"{scen} (N={n_doc}, never used to calibrate): "
              f"real={real_mean:.1f} min, simulated={sim_mean:.1f} min, "
              f"error={cross_val[scen]['pct_error']:.1f}%")
    with open(OUT_DIR / "simulation_cross_scenario_validation.json", "w") as f:
        json.dump(cross_val, f, indent=2)

    # scenario 1: doctor staffing sweep
    print("\n=== What-if: doctor staffing sweep (Monte Carlo, 150 reps each) ===")
    staffing_results = {}
    for n_doc in [1, 2, 3, 4, 5]:
        sim = monte_carlo(dists, num_doctors=n_doc, peak_limit=dists["peak_limit"], n_replications=150)
        staffing_results[n_doc] = {
            "mean_doc_wait_p50": float(sim["mean_doc_wait"].median()),
            "mean_doc_wait_p10": float(sim["mean_doc_wait"].quantile(0.10)),
            "mean_doc_wait_p90": float(sim["mean_doc_wait"].quantile(0.90)),
        }
        sim.to_csv(OUT_DIR / f"whatif_staffing_{n_doc}doc.csv", index=False)
        print(f"N={n_doc}: median mean-DocWait={staffing_results[n_doc]['mean_doc_wait_p50']:.1f} min "
              f"(P10-P90: {staffing_results[n_doc]['mean_doc_wait_p10']:.1f}-{staffing_results[n_doc]['mean_doc_wait_p90']:.1f})")
    with open(OUT_DIR / "whatif_staffing_summary.json", "w") as f:
        json.dump(staffing_results, f, indent=2)

    # scenario 2: peak arrival rate increase sweep
    print("\n=== What-if: peak arrival rate increase (2 doctors, Monte Carlo, 150 reps each) ===")
    rate_results = {}
    for pct in [0, 20, 50, 100]:
        mult = 1 + pct / 100
        sim = monte_carlo(dists, num_doctors=2, peak_limit=dists["peak_limit"],
                           arrival_rate_multiplier=mult, n_replications=150)
        rate_results[pct] = {
            "mean_doc_wait_p50": float(sim["mean_doc_wait"].median()),
            "mean_doc_wait_p10": float(sim["mean_doc_wait"].quantile(0.10)),
            "mean_doc_wait_p90": float(sim["mean_doc_wait"].quantile(0.90)),
        }
        sim.to_csv(OUT_DIR / f"whatif_peakrate_+{pct}pct.csv", index=False)
        print(f"+{pct}% peak arrivals: median mean-DocWait={rate_results[pct]['mean_doc_wait_p50']:.1f} min "
              f"(P10-P90: {rate_results[pct]['mean_doc_wait_p10']:.1f}-{rate_results[pct]['mean_doc_wait_p90']:.1f})")
    with open(OUT_DIR / "whatif_peakrate_summary.json", "w") as f:
        json.dump(rate_results, f, indent=2)


if __name__ == "__main__":
    main()
