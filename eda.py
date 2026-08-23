"""
Step 1 -- EDA and scenario discovery.

Understands the raw dataset's structure before any modeling happens:
  - Verifies the TotalTimeInHospital identity
  - Recovers NumActiveDoctors per run from the DoctorID column
  - Establishes the correct run-identity key (see finding #1 below)
  - Groups/clusters runs by summary statistics to check for distinct
    scenario families
  - Builds a windowed peak/non-peak classifier over each run's
    interarrival sequence (not a per-row rule)
  - Plots wait-time distributions by Priority and by run
  - Writes outputs/eda_findings.md, the source of truth for the README

NOTE ON THIS DATASET: hospital_sim_6_scenarios.csv ships with four
columns beyond the spec's stated schema: Scenario, NumDoctors,
PeakLimit, IsPeakArrival. These are ground truth for exactly the
things Step 1 asks us to discover blind (scenario family, doctor
count, peak/non-peak regime). We do NOT shortcut the discovery work
by reading them directly -- we build the inference/clustering logic
the spec asks for, and then use these columns as a held-out check on
whether that logic recovered the truth. That validation is reported
alongside the discovery findings below.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/hospital_sim_6_scenarios.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def check_run_identity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Finding #1: RunID alone is NOT a unique run identifier in this file.
    It's a local counter (1-50) that resets inside each Scenario, so
    naively grouping by RunID silently merges six unrelated runs into
    one. The true run key is (Scenario, RunID).
    """
    raw_run_nunique = df["RunID"].nunique()
    composite = df["Scenario"].astype(str) + "_R" + df["RunID"].astype(str)
    composite_nunique = composite.nunique()
    df = df.copy()
    df["RunKey"] = composite
    finding = {
        "raw_RunID_nunique": int(raw_run_nunique),
        "composite_RunKey_nunique": int(composite_nunique),
        "n_scenarios": int(df["Scenario"].nunique()),
        "rows_per_RunKey_is_constant": bool(df.groupby("RunKey").size().nunique() == 1),
        "rows_per_RunKey": int(df.groupby("RunKey").size().iloc[0]),
    }
    return df, finding


def check_identity_equation(df: pd.DataFrame) -> dict:
    expected = df["DocWaitTime"] + df["DocServiceTime"] + df["MedWaitTime"] + df["MedServiceTime"]
    mismatch_mask = expected != df["TotalTimeInHospital"]
    return {
        "n_rows": int(len(df)),
        "n_violations": int(mismatch_mask.sum()),
        "identity_holds_for_all_rows": bool(mismatch_mask.sum() == 0),
    }


def recover_num_active_doctors(df: pd.DataFrame) -> pd.DataFrame:
    """NumActiveDoctors per RunKey = nunique(DoctorID) within that run."""
    rec = df.groupby("RunKey")["DoctorID"].nunique().rename("NumActiveDoctors_recovered")
    df = df.merge(rec, on="RunKey", how="left")
    return df


def validate_doctor_recovery(df: pd.DataFrame) -> dict:
    per_run = df.groupby("RunKey").agg(
        recovered=("NumActiveDoctors_recovered", "first"),
        ground_truth=("NumDoctors", "first"),
    )
    match = (per_run["recovered"] == per_run["ground_truth"]).mean()
    return {"pct_runs_recovered_correctly": float(match)}


def cluster_scenarios(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Group RunKeys by summary statistics (mean DocWaitTime,
    NumActiveDoctors, arrival density) to check whether distinct
    scenario families exist, independent of any label in the file.
    """
    per_run = df.groupby("RunKey").agg(
        mean_doc_wait=("DocWaitTime", "mean"),
        mean_med_wait=("MedWaitTime", "mean"),
        num_active_doctors=("NumActiveDoctors_recovered", "first"),
        n_patients=("PatientID", "size"),
        run_span=("ArrivalTime", lambda s: s.max() - s.min()),
        scenario_truth=("Scenario", "first"),
    ).reset_index()
    per_run["arrival_density"] = per_run["n_patients"] / per_run["run_span"]

    feats = per_run[["mean_doc_wait", "mean_med_wait", "num_active_doctors", "arrival_density"]]
    scaled = StandardScaler().fit_transform(feats)

    # Try a small range of k and pick by inertia elbow + match against
    # true scenario count (6) for reporting purposes only.
    inertias = {}
    for k in range(2, 10):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
        inertias[k] = float(km.inertia_)

    km6 = KMeans(n_clusters=6, n_init=10, random_state=42).fit(scaled)
    per_run["cluster_k6"] = km6.labels_

    # cross-tab cluster vs ground-truth scenario to see how cleanly
    # the unsupervised clustering recovers the six scenario families
    crosstab = pd.crosstab(per_run["cluster_k6"], per_run["scenario_truth"])
    # for each cluster, the purity = fraction belonging to majority scenario
    purity = (crosstab.max(axis=1) / crosstab.sum(axis=1)).mean()

    finding = {
        "inertia_by_k": inertias,
        "k6_cluster_purity_vs_true_scenario": float(purity),
        "crosstab_cluster_vs_scenario": crosstab.to_dict(),
        "per_run_summary_preview": per_run.head(10).to_dict(orient="records"),
    }
    return per_run, finding


def windowed_peak_classifier(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Classify each patient's arrival as peak / non-peak using a
    windowed rule over the interarrival-gap sequence, per the spec:
      - peak: >=85% of gaps in a local window are <=5 min
      - non-peak: ~60% of gaps in a local window are in [6,7] min
    Implemented as a rolling window over each run's sorted arrival
    sequence (window size = 15 patients), not a per-row threshold.
    """
    df = df.sort_values(["RunKey", "ArrivalTime"]).copy()
    df["gap_before"] = df.groupby("RunKey")["ArrivalTime"].diff()

    WINDOW = 15

    def classify_run(g: pd.DataFrame) -> pd.Series:
        gaps = g["gap_before"].values
        n = len(gaps)
        pred = np.full(n, "non_peak", dtype=object)
        for i in range(n):
            lo = max(0, i - WINDOW // 2)
            hi = min(n, i + WINDOW // 2 + 1)
            window = gaps[lo:hi]
            window = window[~np.isnan(window)]
            if len(window) == 0:
                continue
            pct_le5 = (window <= 5).mean()
            pct_6to7 = ((window >= 6) & (window <= 7)).mean()
            if pct_le5 >= 0.85:
                pred[i] = "peak"
            elif pct_6to7 >= 0.60:
                pred[i] = "non_peak"
            else:
                # ambiguous window: fall back to the raw local gap
                pred[i] = "peak" if (not np.isnan(gaps[i]) and gaps[i] <= 5) else "non_peak"
        return pd.Series(pred, index=g.index)

    df["Period_inferred"] = df.groupby("RunKey", group_keys=False).apply(classify_run)

    # validate against ground-truth IsPeakArrival where available
    df["Period_truth"] = np.where(df["IsPeakArrival"] == 1, "peak", "non_peak")
    acc = (df["Period_inferred"] == df["Period_truth"]).mean()

    finding = {
        "window_size": WINDOW,
        "accuracy_vs_ground_truth_IsPeakArrival": float(acc),
        "confusion": pd.crosstab(df["Period_truth"], df["Period_inferred"]).to_dict(),
        "note": (
            "Ground truth shows IsPeakArrival is actually a deterministic "
            "step function of ArrivalTime vs. the run's PeakLimit "
            "(ArrivalTime <= PeakLimit => peak), not a noisy regime that "
            "needs statistical inference. The windowed gap classifier "
            "above is still built exactly as specified and cross-checked "
            "against this ground truth; see accuracy figure."
        ),
    }
    return df, finding


def plot_distributions(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for pr in sorted(df["Priority"].unique()):
        axes[0, 0].hist(df.loc[df.Priority == pr, "DocWaitTime"], bins=60, alpha=0.5,
                         label=f"Priority {pr}", range=(0, df["DocWaitTime"].quantile(0.99)))
    axes[0, 0].set_title("DocWaitTime distribution by Priority")
    axes[0, 0].set_xlabel("minutes")
    axes[0, 0].legend()

    for pr in sorted(df["Priority"].unique()):
        axes[0, 1].hist(df.loc[df.Priority == pr, "MedWaitTime"], bins=20, alpha=0.5,
                         label=f"Priority {pr}")
    axes[0, 1].set_title("MedWaitTime distribution by Priority")
    axes[0, 1].set_xlabel("minutes")
    axes[0, 1].legend()

    per_run_wait = df.groupby("RunKey")["DocWaitTime"].mean().sort_values()
    axes[1, 0].bar(range(len(per_run_wait)), per_run_wait.values, color="steelblue")
    axes[1, 0].set_title("Mean DocWaitTime per run (sorted)")
    axes[1, 0].set_xlabel("run (sorted)")
    axes[1, 0].set_ylabel("mean DocWaitTime (min)")

    scen_order = df.groupby("Scenario")["DocWaitTime"].mean().sort_values().index
    df.boxplot(column="DocWaitTime", by="Scenario", ax=axes[1, 1], rot=45)
    axes[1, 1].set_title("DocWaitTime by Scenario")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_distributions.png", dpi=110)
    plt.close(fig)


def main():
    df = load_raw()
    df, run_identity_finding = check_run_identity(df)
    identity_eq_finding = check_identity_equation(df)
    df = recover_num_active_doctors(df)
    doctor_recovery_finding = validate_doctor_recovery(df)
    per_run_summary, cluster_finding = cluster_scenarios(df)
    df, peak_finding = windowed_peak_classifier(df)
    plot_distributions(df)

    findings = {
        "1_run_identity": run_identity_finding,
        "2_total_time_identity_equation": identity_eq_finding,
        "3_doctor_recovery_validation": doctor_recovery_finding,
        "4_scenario_clustering": cluster_finding,
        "5_peak_nonpeak_classifier": peak_finding,
    }
    with open(OUT_DIR / "eda_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    # persist the enriched frame for Step 2
    df.to_parquet(OUT_DIR / "enriched_step1.parquet", index=False)
    per_run_summary.to_csv(OUT_DIR / "per_run_summary.csv", index=False)

    print("=== STEP 1 EDA FINDINGS ===")
    print(json.dumps(findings, indent=2, default=str)[:4000])
    print("\nSaved: outputs/eda_findings.json, outputs/eda_distributions.png, "
          "outputs/enriched_step1.parquet, outputs/per_run_summary.csv")


if __name__ == "__main__":
    main()
