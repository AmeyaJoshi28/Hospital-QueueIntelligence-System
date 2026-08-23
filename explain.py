"""
Step 5 -- Explainability.

Turns DocWaitTime / MedWaitTime predictions into natural-language
explanations backed by SHAP, e.g.:

  "Expected doctor wait: 22 min (range 16-29 min). Main factors:
   peak-hour arrival (+9 min), only 2 active doctors this run
   (+7 min), medium priority (+2 min)."

A bare number is never returned on its own -- every explanation
carries a range and at least one contributing factor.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from models.common import DOC_FEATURES, MED_FEATURES

OUT_DIR = Path("outputs")

FRIENDLY_NAMES = {
    "Priority": {
        1: "low priority",
        2: "medium priority",
        3: "high priority",
    },
    "Period_peak_flag": {
        1: "peak-hour arrival",
        0: "non-peak arrival",
    },
}


def _friendly_factor(feature: str, value, shap_value: float) -> str:
    sign = "+" if shap_value >= 0 else "-"
    minutes = f"{sign}{abs(shap_value):.0f} min"

    if feature in FRIENDLY_NAMES:
        label = FRIENDLY_NAMES[feature].get(int(value), f"{feature}={value}")
        return f"{label} ({minutes})"
    if feature == "NumActiveDoctors_recovered":
        return f"only {int(value)} active doctor{'s' if value != 1 else ''} this run ({minutes})"
    if feature == "queue_for_doctor_at_arrival":
        return f"{int(value)} patient{'s' if value != 1 else ''} already queued for the doctor ({minutes})"
    if feature == "doctors_busy_at_arrival":
        return f"{int(value)} doctor{'s' if value != 1 else ''} already busy on arrival ({minutes})"
    if feature == "pharmacy_queue_at_doc_end":
        return f"{int(value)} patient{'s' if value != 1 else ''} ahead in the pharmacy queue ({minutes})"
    if feature == "pharmacy_busy_at_doc_end":
        state = "busy" if value >= 1 else "free"
        return f"pharmacy counter {state} on arrival ({minutes})"
    return f"{feature}={value} ({minutes})"


class WaitExplainer:
    def __init__(self, stage: str):
        assert stage in ("doc", "med")
        self.stage = stage
        self.model = joblib.load(OUT_DIR / f"model_{'doc_wait' if stage=='doc' else 'med_wait'}_xgb.joblib")
        self.mapie = joblib.load(OUT_DIR / f"model_{'doc_wait' if stage=='doc' else 'med_wait'}_mapie.joblib")
        self.features = DOC_FEATURES if stage == "doc" else MED_FEATURES
        self.explainer = shap.TreeExplainer(self.model)
        self.label = "doctor" if stage == "doc" else "pharmacy"

    def explain(self, row: pd.Series, top_k: int = 3) -> dict:
        X = row[self.features].to_frame().T.astype(float)
        pred, interval = self.mapie.predict_interval(X)
        pred = max(0.0, float(pred[0]))
        lo = max(0.0, float(interval[0, 0, 0]))
        hi = max(0.0, float(interval[0, 1, 0]))

        shap_values = self.explainer.shap_values(X)[0]
        order = np.argsort(-np.abs(shap_values))[:top_k]
        factors = [
            _friendly_factor(self.features[i], X.iloc[0, i], shap_values[i])
            for i in order
        ]

        sentence = (
            f"Expected {self.label} wait: {pred:.0f} min "
            f"(range {lo:.0f}-{hi:.0f} min). "
            f"Main factors: {', '.join(factors)}."
        )
        return {
            "prediction": pred,
            "range": (lo, hi),
            "factors": factors,
            "sentence": sentence,
        }


def main():
    doc_test = pd.read_parquet(OUT_DIR / "test_predictions_doc_wait.parquet")
    med_test = pd.read_parquet(OUT_DIR / "test_predictions_med_wait.parquet")

    doc_exp = WaitExplainer("doc")
    med_exp = WaitExplainer("med")

    print("=== Sample DocWaitTime explanations ===")
    for _, row in doc_test.sample(3, random_state=1).iterrows():
        print("-", doc_exp.explain(row)["sentence"], f"  [actual: {row['DocWaitTime']:.0f} min]")

    print()
    print("=== Sample MedWaitTime explanations ===")
    for _, row in med_test.sample(3, random_state=1).iterrows():
        print("-", med_exp.explain(row)["sentence"], f"  [actual: {row['MedWaitTime']:.0f} min]")


if __name__ == "__main__":
    main()
