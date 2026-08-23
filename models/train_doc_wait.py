"""
Step 3a -- DocWaitTime regression: baseline linear regression vs
XGBoost, wrapped with mapie split-conformal intervals for calibrated
prediction ranges (not just point estimates).

Features: Priority, Period, NumActiveDoctors, reconstructed
doctor-queue-state-at-arrival.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import DOC_FEATURES, load_features, run_level_split, metrics, interval_coverage

TARGET = "DocWaitTime"
OUT_DIR = Path("outputs")


def main():
    df = load_features()
    train_df, test_df = run_level_split(df)

    # further split train into fit/calibration sets by RunKey for mapie
    train_runs = train_df["RunKey"].to_numpy(dtype=object)
    train_runs = np.unique(train_runs)
    fit_runs, calib_runs = train_test_split(train_runs, test_size=0.25, random_state=42)
    fit_df = train_df[train_df["RunKey"].isin(fit_runs)]
    calib_df = train_df[train_df["RunKey"].isin(calib_runs)]

    X_fit, y_fit = fit_df[DOC_FEATURES], fit_df[TARGET].values
    X_calib, y_calib = calib_df[DOC_FEATURES], calib_df[TARGET].values
    X_test, y_test = test_df[DOC_FEATURES], test_df[TARGET].values

    results = {}

    # --- Baseline: Linear Regression ---
    lr = LinearRegression().fit(X_fit, y_fit)
    lr_pred = lr.predict(X_test)
    lr_pred = np.clip(lr_pred, 0, None)
    results["linear_regression"] = metrics(y_test, lr_pred)

    # --- Advanced: XGBoost ---
    xgb = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ).fit(X_fit, y_fit)
    xgb_pred = xgb.predict(X_test)
    xgb_pred = np.clip(xgb_pred, 0, None)
    results["xgboost"] = metrics(y_test, xgb_pred)

    # --- XGBoost + mapie conformal intervals ---
    mapie_reg = SplitConformalRegressor(estimator=xgb, confidence_level=0.9, prefit=True)
    mapie_reg.conformalize(X_calib, y_calib)
    xgb_pred_c, xgb_interval = mapie_reg.predict_interval(X_test)
    lo = np.clip(xgb_interval[:, 0, 0], 0, None)
    hi = np.clip(xgb_interval[:, 1, 0], 0, None)
    results["xgboost_conformal"] = {
        **metrics(y_test, np.clip(xgb_pred_c, 0, None)),
        "interval_coverage_90pct_target": interval_coverage(y_test, lo, hi),
        "mean_interval_width": float(np.mean(hi - lo)),
    }

    print("=== DocWaitTime: baseline vs advanced ===")
    print(json.dumps(results, indent=2))

    best = min(["linear_regression", "xgboost"], key=lambda k: results[k]["MAE"])
    print(f"\nBest point-estimate model: {best} "
          f"(MAE {results[best]['MAE']:.2f} vs "
          f"{results['linear_regression' if best=='xgboost' else 'xgboost']['MAE']:.2f})")

    joblib.dump(xgb, OUT_DIR / "model_doc_wait_xgb.joblib")
    joblib.dump(lr, OUT_DIR / "model_doc_wait_lr.joblib")
    joblib.dump(mapie_reg, OUT_DIR / "model_doc_wait_mapie.joblib")
    with open(OUT_DIR / "results_doc_wait.json", "w") as f:
        json.dump(results, f, indent=2)

    # cache test predictions for the explainability step
    cols = list(dict.fromkeys(["RunKey", "PatientID", TARGET] + DOC_FEATURES))
    out = test_df[cols].copy()
    out["pred_xgb"] = xgb_pred
    out["pred_lo90"] = lo
    out["pred_hi90"] = hi
    out.to_parquet(OUT_DIR / "test_predictions_doc_wait.parquet", index=False)


if __name__ == "__main__":
    main()
