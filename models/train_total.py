"""
Step 3c -- TotalTimeInHospital: model directly, and compare against
summing the (already-trained) DocWaitTime + MedWaitTime models plus
the known service times. Reports which is more accurate and why.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from mapie.regression import SplitConformalRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import TOTAL_FEATURES, DOC_FEATURES, MED_FEATURES, load_features, run_level_split, metrics, \
    interval_coverage

TARGET = "TotalTimeInHospital"
OUT_DIR = Path("outputs")


def main():
    df = load_features()
    train_df, test_df = run_level_split(df)

    train_runs = np.unique(train_df["RunKey"].to_numpy(dtype=object))
    fit_runs, calib_runs = train_test_split(train_runs, test_size=0.25, random_state=42)
    fit_df = train_df[train_df["RunKey"].isin(fit_runs)]
    calib_df = train_df[train_df["RunKey"].isin(calib_runs)]

    results = {}

    # --- Direct model ---
    X_fit, y_fit = fit_df[TOTAL_FEATURES], fit_df[TARGET].values
    X_calib, y_calib = calib_df[TOTAL_FEATURES], calib_df[TARGET].values
    X_test, y_test = test_df[TOTAL_FEATURES], test_df[TARGET].values

    lr = LinearRegression().fit(X_fit, y_fit)
    results["linear_regression_direct"] = metrics(y_test, np.clip(lr.predict(X_test), 0, None))

    xgb_direct = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ).fit(X_fit, y_fit)
    direct_pred = np.clip(xgb_direct.predict(X_test), 0, None)
    results["xgboost_direct"] = metrics(y_test, direct_pred)

    mapie_reg = SplitConformalRegressor(estimator=xgb_direct, confidence_level=0.9, prefit=True)
    mapie_reg.conformalize(X_calib, y_calib)
    direct_pred_c, direct_interval = mapie_reg.predict_interval(X_test)
    lo = np.clip(direct_interval[:, 0, 0], 0, None)
    hi = np.clip(direct_interval[:, 1, 0], 0, None)
    results["xgboost_direct_conformal"] = {
        **metrics(y_test, np.clip(direct_pred_c, 0, None)),
        "interval_coverage_90pct_target": interval_coverage(y_test, lo, hi),
        "mean_interval_width": float(np.mean(hi - lo)),
    }

    # --- Sum-of-parts: reuse the already-trained doc/med XGB models ---
    doc_xgb = joblib.load(OUT_DIR / "model_doc_wait_xgb.joblib")
    med_xgb = joblib.load(OUT_DIR / "model_med_wait_xgb.joblib")

    doc_pred_test = np.clip(doc_xgb.predict(test_df[DOC_FEATURES]), 0, None)
    med_pred_test = np.clip(med_xgb.predict(test_df[MED_FEATURES]), 0, None)
    # TotalTime = DocWait + DocService + MedWait + MedService; service times are
    # known/given (not modeled -- they're not queue-dependent), so the
    # sum-of-parts estimate adds the *known* service times to the two
    # *predicted* wait times, exactly mirroring the dataset's own identity.
    sum_pred = doc_pred_test + test_df["DocServiceTime"].values + med_pred_test + test_df["MedServiceTime"].values
    results["sum_of_parts_xgb"] = metrics(y_test, sum_pred)

    print("=== TotalTimeInHospital: direct model vs sum-of-parts ===")
    print(json.dumps(results, indent=2))

    direct_mae = results["xgboost_direct"]["MAE"]
    sum_mae = results["sum_of_parts_xgb"]["MAE"]
    winner = "direct model" if direct_mae < sum_mae else "sum-of-parts"
    print(f"\n{winner} is more accurate "
          f"(direct MAE={direct_mae:.2f}, sum-of-parts MAE={sum_mae:.2f})")

    joblib.dump(xgb_direct, OUT_DIR / "model_total_direct_xgb.joblib")
    with open(OUT_DIR / "results_total.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
