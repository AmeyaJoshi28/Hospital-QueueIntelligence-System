"""
Shared utilities for Step 3 models.

Train/test split is by RunKey (the run), not by row: patients within
a run share the same doctor count, arrival regime, and congestion
history, so a row-level split would leak run-level information across
the boundary. We hold out whole runs, stratified by Scenario so every
scenario family is represented in both splits.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

FEATURES_PATH = Path("outputs/features.parquet")

DOC_FEATURES = [
    "Priority", "NumActiveDoctors_recovered", "Period_peak_flag",
    "queue_for_doctor_at_arrival", "doctors_busy_at_arrival",
]
MED_FEATURES = [
    "Priority", "Period_peak_flag",
    "pharmacy_queue_at_doc_end", "pharmacy_busy_at_doc_end",
]
TOTAL_FEATURES = list(dict.fromkeys(DOC_FEATURES + MED_FEATURES))


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df["Period_peak_flag"] = (df["Period_inferred"] == "peak").astype(int)
    return df


def run_level_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split whole RunKeys into train/test, stratified by Scenario."""
    runs = df[["RunKey", "Scenario"]].drop_duplicates()
    train_runs, test_runs = train_test_split(
        runs, test_size=test_size, random_state=seed, stratify=runs["Scenario"]
    )
    train_df = df[df["RunKey"].isin(train_runs["RunKey"])].copy()
    test_df = df[df["RunKey"].isin(test_runs["RunKey"])].copy()
    return train_df, test_df


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def interval_coverage(y_true, lo, hi) -> float:
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
