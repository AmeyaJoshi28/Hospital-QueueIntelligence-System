"""
Step 2 -- Feature engineering: inferring system state from static event logs.

The raw log gives each patient's final wait/service times but not what
the system looked like at the moment they arrived. We reconstruct it:

  1. Rebuild each patient's timeline: doctor-consult start/end and
     pharmacy start/end, from ArrivalTime + the four wait/service columns.
  2. From that timeline, derive a state-at-arrival snapshot for every
     patient -- how many patients were already queued for the doctor,
     how many doctors were busy, how many patients were queued for
     pharmacy -- at the exact instant this patient walked in. This uses
     only OTHER patients' already-realized intervals; it never uses the
     patient's own wait/service times to describe their own arrival
     state (no leakage).
  3. As a documented extension (see README), we additionally snapshot
     state at the moment each patient reaches the pharmacy stage
     (their own doc_end), since that is what actually drives their
     MedWaitTime and the spec calls out pharmacy-stage features
     specifically for that model. This uses the patient's own
     ArrivalTime/DocWaitTime/DocServiceTime -- which are given, realized
     facts by the time they reach pharmacy, not the target being
     predicted -- so it is not leakage for the MedWaitTime target.
  4. NumActiveDoctors and inferred Period (peak/non-peak) from Step 1
     are attached as features too.

This reconstruction is the core technical differentiator of this
project: turning a flat outcome log into the queueing state that
produced those outcomes.
"""
import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("outputs/enriched_step1.parquet")
OUT_PATH = Path("outputs/features.parquet")


def reconstruct_timeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["doc_start"] = df["ArrivalTime"] + df["DocWaitTime"]
    df["doc_end"] = df["doc_start"] + df["DocServiceTime"]
    df["pharmacy_start"] = df["doc_end"] + df["MedWaitTime"]
    df["pharmacy_end"] = df["pharmacy_start"] + df["MedServiceTime"]
    return df


def state_snapshot_at_times(g: pd.DataFrame, query_times: np.ndarray) -> pd.DataFrame:
    """
    For a single run's patients g (already timeline-reconstructed),
    compute, for each query time t in query_times (aligned to g's row
    order), the count of OTHER patients whose interval covers t for
    three queueing states: doctor queue, doctor-busy, pharmacy queue.
    Vectorized via broadcasting (run size is small, ~200 patients).
    """
    arr = g["ArrivalTime"].values[:, None]      # (n,1)
    ds = g["doc_start"].values[:, None]
    de = g["doc_end"].values[:, None]
    ps = g["pharmacy_start"].values[:, None]

    t = query_times[None, :]                    # (1,n)

    in_doc_queue = (arr <= t) & (t < ds)         # arrived, not yet started doctor
    in_doc_busy = (ds <= t) & (t < de)           # currently with a doctor
    in_pharm_queue = (de <= t) & (t < ps)         # finished doctor, waiting for pharmacy

    # zero out self-comparison (a patient never counts toward their own queue)
    n = g.shape[0]
    self_mask = np.eye(n, dtype=bool)
    in_doc_queue = in_doc_queue & ~self_mask
    in_doc_busy = in_doc_busy & ~self_mask
    in_pharm_queue = in_pharm_queue & ~self_mask

    return pd.DataFrame({
        "queue_for_doctor_at_arrival": in_doc_queue.sum(axis=0),
        "doctors_busy_at_arrival": in_doc_busy.sum(axis=0),
        "queue_for_pharmacy_at_arrival": in_pharm_queue.sum(axis=0),
    }, index=g.index)


def state_snapshot_at_pharmacy_entry(g: pd.DataFrame) -> pd.DataFrame:
    """
    Extension: pharmacy-stage snapshot at each patient's own doc_end
    (the moment they join the pharmacy queue). Single-server pharmacy,
    so 'busy' is 0/1.
    """
    query_times = g["doc_end"].values
    ps = g["pharmacy_start"].values[:, None]
    pe = g["pharmacy_end"].values[:, None]
    t = query_times[None, :]

    in_pharm_queue = (ps > t) & (g["doc_end"].values[:, None] <= t)  # arrived at pharmacy (doc_end<=t) but not started
    # more direct: someone is "ahead in pharmacy queue" if their doc_end <= t < their pharmacy_start
    de = g["doc_end"].values[:, None]
    in_pharm_queue = (de <= t) & (t < ps)
    pharm_busy = (ps <= t) & (t < pe)

    n = g.shape[0]
    self_mask = np.eye(n, dtype=bool)
    in_pharm_queue = in_pharm_queue & ~self_mask
    pharm_busy = pharm_busy & ~self_mask

    return pd.DataFrame({
        "pharmacy_queue_at_doc_end": in_pharm_queue.sum(axis=0),
        "pharmacy_busy_at_doc_end": pharm_busy.sum(axis=0).clip(0, 1),
    }, index=g.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = reconstruct_timeline(df)

    doc_state_parts = []
    pharm_state_parts = []
    for run_key, g in df.groupby("RunKey", sort=False):
        g = g.sort_values("ArrivalTime")
        doc_state_parts.append(state_snapshot_at_times(g, g["ArrivalTime"].values))
        pharm_state_parts.append(state_snapshot_at_pharmacy_entry(g))

    doc_state = pd.concat(doc_state_parts)
    pharm_state = pd.concat(pharm_state_parts)

    df = df.join(doc_state).join(pharm_state)

    # sanity: doctors_busy_at_arrival should never exceed NumActiveDoctors
    over = (df["doctors_busy_at_arrival"] > df["NumActiveDoctors_recovered"]).sum()
    assert over == 0, f"{over} rows have more doctors busy than exist -- reconstruction bug"

    return df


def main():
    df = pd.read_parquet(IN_PATH)
    df = build_features(df)

    feature_cols = [
        "Priority", "Period_inferred", "NumActiveDoctors_recovered",
        "queue_for_doctor_at_arrival", "doctors_busy_at_arrival",
        "queue_for_pharmacy_at_arrival", "pharmacy_queue_at_doc_end",
        "pharmacy_busy_at_doc_end",
    ]
    print("=== Feature summary ===")
    print(df[feature_cols].describe(include="all"))
    print()
    print("=== Correlation of reconstructed state with actual wait times ===")
    corr_doc = df[["queue_for_doctor_at_arrival", "doctors_busy_at_arrival", "DocWaitTime"]].corr()["DocWaitTime"]
    corr_med = df[["pharmacy_queue_at_doc_end", "pharmacy_busy_at_doc_end", "MedWaitTime"]].corr()["MedWaitTime"]
    print("DocWaitTime correlations:\n", corr_doc)
    print("MedWaitTime correlations:\n", corr_med)

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows x {df.shape[1]} cols to {OUT_PATH}")


if __name__ == "__main__":
    main()
