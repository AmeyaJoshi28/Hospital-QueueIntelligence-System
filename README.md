# Hospital Flow: Wait-Time Prediction, Bottleneck Detection & What-If Simulation

End-to-end pipeline on `hospital_sim_6_scenarios.csv` — 60,000 patient
visits across 300 simulated runs (6 staffing/arrival scenarios × 50
runs each), two sequential queueing stages (multi-server doctor
consultation → single-server pharmacy).

```
eda.py -> features.py -> models/train_*.py -> utilization.py -> explain.py -> simulate.py -> app_patient.py / app_admin.py
```

Run in that order from the project root:

```bash
pip install -r requirements.txt
python eda.py
python features.py
python -m models.train_doc_wait
python -m models.train_med_wait
python -m models.train_total
python utilization.py
python explain.py
python simulate.py
python -m streamlit run app_patient.py   # patient view
python -m streamlit run app_admin.py     # admin view
```

---

## Step 1 — Scenario-discovery findings

**Finding 1 — `RunID` is not a unique run identifier.** `RunID` is a
local counter that resets to 1 inside each of the 6 scenarios (`df.RunID.nunique()`
returns 50, not 300). Grouping or splitting by raw `RunID` silently
merges six unrelated runs into one. The true run key used everywhere
downstream is the composite `Scenario + RunID` ("RunKey"), giving 300
distinct runs of 200 patients each. This is the single most consequential
finding in the EDA — it would have quietly broken the train/test split,
the utilization calculation, and every per-run aggregate if missed.

**Finding 2 — the `TotalTimeInHospital` identity holds exactly.**
`DocWaitTime + DocServiceTime + MedWaitTime + MedServiceTime` matches
`TotalTimeInHospital` on all 60,000 rows — zero violations.

**Finding 3 — `NumActiveDoctors` is fully recoverable from `DoctorID`.**
`nunique(DoctorID)` per run matches the file's own (independently
present) `NumDoctors` column on 100% of the 300 runs, confirming the
reconstruction approach is sound before it's relied on for anything
else.

**Finding 4 — six scenario families exist, and unsupervised clustering
mostly (not perfectly) recovers them.** Grouping runs by (mean
DocWaitTime, mean MedWaitTime, recovered doctor count, arrival
density) and clustering with k-means, k=6 gives 78.6% purity against
the ground-truth `Scenario` label. The confusion is concentrated
exactly where it should be: `Peak_120Min` vs. `Baseline_2Doc_240Peak`
(both 2 doctors, similar aggregate wait) and `Staffing_3Doc` vs.
`Staffing_4Doc` (both lightly loaded) blur together on wait-time-based
features alone, while `Staffing_1Doc` (extreme congestion) forms a
perfectly pure singleton cluster. This is the expected failure mode
for stats-based clustering: it separates load regimes cleanly but
can't fully disentangle two configurations that happen to produce
similar aggregate congestion.

**Finding 5 — the windowed peak/non-peak classifier is right 81% of
the time, and the residual error is explained by the ground truth
itself.** A rolling-window classifier over each run's interarrival-gap
sequence (window = 15 patients; ≥85% of gaps ≤5 min → peak, ~60% of
gaps in [6,7] min → non-peak, exactly as specified) reaches 81.1%
accuracy against the file's own `IsPeakArrival` column. Digging into
*why* it isn't higher: `IsPeakArrival` turns out to be a **deterministic
step function** of `ArrivalTime` vs. that run's `PeakLimit`
(`ArrivalTime <= PeakLimit → peak`), not a noisy statistical regime.
A smoothed gap-window classifier necessarily blurs the boundary right
around the cutoff, which is exactly where its errors concentrate. Both
things are reported here deliberately: we built the inference exactly
as asked rather than reading the ground-truth columns directly, and
then used those columns as a validation check.

Note on the dataset: `hospital_sim_6_scenarios.csv` ships with four
columns beyond the spec's stated schema — `Scenario`, `NumDoctors`,
`PeakLimit`, `IsPeakArrival`. Every "recover this from the log" step
above (doctor count, scenario family, peak regime) is done blind, and
these columns are used strictly as an accuracy check afterward, never
as a shortcut.

Outputs: `outputs/eda_findings.json`, `outputs/eda_distributions.png`,
`outputs/per_run_summary.csv`.

---

## Step 2 — State reconstruction (the core technical differentiator)

The raw log records only *outcomes* — how long each patient waited —
never the queue state they walked into. `features.py` rebuilds it:

1. **Timeline reconstruction.** For every patient: `doc_start =
   ArrivalTime + DocWaitTime`, `doc_end = doc_start + DocServiceTime`,
   `pharmacy_start = doc_end + MedWaitTime`, `pharmacy_end =
   pharmacy_start + MedServiceTime`.
2. **State-at-arrival snapshot.** At the exact instant each patient
   arrives, count (from *other* patients' already-realized intervals
   only): how many are already queued for the doctor, how many
   doctors are currently occupied, how many are queued for pharmacy.
   A patient never counts toward their own state, and no feature for
   patient *i* ever touches patient *i*'s own wait/service times — the
   quantities being predicted. Implemented as a vectorized interval-
   overlap check per run (small `n≈200` per run, so a dense
   broadcast comparison is both simple and fast).
3. **Extension: pharmacy-arrival snapshot.** Because a patient's
   pharmacy-stage congestion is only meaningful once they *reach*
   pharmacy (at their own `doc_end`, which is a realized fact by then,
   not the MedWaitTime target itself), we additionally snapshot
   pharmacy-queue-length and pharmacy-busy state at that moment. This
   is what actually drives the MedWaitTime model, per the spec's own
   note that pharmacy-stage features matter more than server count
   there (there's only ever one server).

**Sanity checks that ran automatically:** `doctors_busy_at_arrival`
never exceeds the run's recovered doctor count (hard assertion,
0 violations). Correlation of reconstructed state with real outcomes:
`queue_for_doctor_at_arrival` ↔ `DocWaitTime` = **0.66**;
`pharmacy_busy_at_doc_end` ↔ `MedWaitTime` = **0.85**. Both features
carry real signal, not noise.

Output: `outputs/features.parquet` (60,000 rows × 28 columns).

---

## Step 3 — Predictive models (baseline vs. advanced)

Train/test split is by **RunKey**, not by row (stratified by
Scenario so every family appears in both splits) — patients in the
same run share arrival history and congestion, so a row-level split
would leak information across the boundary.

| Model | Target | MAE (min) | RMSE (min) |
|---|---|---:|---:|
| Linear Regression (baseline) | DocWaitTime | 139.6 | 232.3 |
| XGBoost | DocWaitTime | **23.1** | **52.3** |
| XGBoost + mapie 90% interval | DocWaitTime | 23.1 | 52.3 |
| Linear Regression (baseline) | MedWaitTime | 0.19 | 0.51 |
| XGBoost | MedWaitTime | **0.17** | **0.48** |
| Linear Regression (direct) | TotalTimeInHospital | 143.2 | 232.4 |
| XGBoost (direct) | TotalTimeInHospital | 24.2 | 52.4 |
| **Sum-of-parts (Doc-XGB + Med-XGB + known service times)** | TotalTimeInHospital | **23.2** | **52.3** |

XGBoost beats linear regression by roughly 6x MAE on DocWaitTime — the
wait-time function is highly non-linear (near-zero for high-priority
patients regardless of congestion, but growing sharply and
super-linearly with queue length for low-priority patients under a
saturated single- or two-doctor system), which a linear model
structurally cannot represent.

**mapie conformal intervals** (`SplitConformalRegressor`, split
calibration on a held-out 25% of the training runs) give 90%-target
prediction ranges with **85.8%** empirical coverage on DocWaitTime and
**91.9%** on MedWaitTime test data — reasonably close to nominal,
with DocWaitTime under-covering slightly because its distribution is
extremely heavy-tailed and priority-driven (see Explainability note
below).

**Direct vs. sum-of-parts for TotalTimeInHospital:** sum-of-parts
wins, narrowly (MAE 23.2 vs. 24.2). This makes sense structurally:
`TotalTimeInHospital` is overwhelmingly dominated by `DocWaitTime`
(mean 226 min vs. `MedWaitTime`'s mean 0.4 min), and two
purpose-built models — each with the right stage-specific features —
compose slightly better than one model forced to learn both stages'
dynamics implicitly through a merged feature set. The margin is small
because the doctor stage already accounts for nearly all the variance
either way.

Outputs: `outputs/results_{doc_wait,med_wait,total}.json`,
`outputs/model_*.joblib`, `outputs/test_predictions_*.parquet`.

---

## Step 4 — Utilization & bottleneck detection

Rule-based, no model: ρ = busy-server-time / (run duration × server
count) per stage, cross-checked against whether that stage's queue
length trends upward over the run.

| Scenario | ρ (doctor) | ρ (pharmacy) | Bottleneck (of 50 runs) |
|---|---:|---:|---|
| Baseline_2Doc_240Peak | 0.993 | 0.445 | doctor: 50 |
| Peak_120Min | 0.993 | 0.445 | doctor: 50 |
| Peak_360Min | 0.993 | 0.447 | doctor: 50 |
| Staffing_1Doc | 0.999 | 0.225 | doctor: 50 |
| Staffing_3Doc | 0.847 | 0.573 | doctor: 19, none: 29, pharmacy: 2 |
| Staffing_4Doc | 0.639 | 0.575 | doctor: 1, none: 46, pharmacy: 3 |

The doctor stage is the binding constraint everywhere until staffing
reaches 3–4 doctors, at which point most runs become stable with no
binding constraint — and the pharmacy (never touched by any staffing
change) starts to show up as the bottleneck in a handful of runs once
the doctor stage stops being the limiting factor. That's exactly the
qualitative behavior a two-stage queue should show, and it lines up
with the simulation's staffing sweep below.

Output: `outputs/utilization_by_run.csv`, `outputs/utilization_summary.json`.

---

## Step 5 — Explainability

SHAP (`TreeExplainer` on the XGBoost models) converts every prediction
into a sentence with a range and named factors, never a bare number:

> Expected doctor wait: 351 min (range 297–405 min). Main factors:
> low priority (+113 min), 29 patients already queued for the doctor
> (+107 min), only 2 active doctors this run (−81 min).

Note on magnitude: because `Priority` governs a near-binary regime
(mean DocWaitTime is 342 min at Priority 1 vs. 9.8 min at Priority 2
and 3.4 min at Priority 3 — high-priority patients essentially skip
the queue), individual SHAP contributions can look large relative to
the final prediction; they are internally consistent (`sum(SHAP) +
base_value == prediction`, verified) and reflect a genuinely dominant,
near-step-function driver rather than a bug.

Output: `WaitExplainer` class in `explain.py`, used directly by both
dashboards.

---

## Step 6 — What-if simulation

Distributions are fit as **empirical discrete PMFs via bootstrap
resampling** — interarrival gaps and service times only ever take a
handful of integer values (e.g. `DocServiceTime ∈ {10,13,15,18,20}`),
so a discrete empirical fit beats forcing a continuous parametric one.
(A geometric fit was tried on the peak-hour gaps as a sanity check and
rejected — KS stat 0.55, p≈0 — confirming empirical resampling was the
right call.) All distributions, plus the peak/non-peak cutoff
(`PeakLimit`), are calibrated **only** from the `Baseline_2Doc_240Peak`
scenario. The SimPy model uses a `PriorityResource` for the doctor
queue (higher stated Priority is served ahead of already-waiting
lower-priority patients, non-preemptive — this is what the data shows)
and a plain FIFO `Resource` for the single-server pharmacy (Priority
has no measurable effect on MedWaitTime in the data).

**Validation, in two layers, before using the simulator for anything:**

1. *In-sample:* simulated per-run mean DocWaitTime vs. real
   `Baseline_2Doc_240Peak` runs — KS stat 0.09, **p = 0.94** (no
   significant difference). MedWaitTime: KS stat 0.07, **p = 0.996**.
2. *Out-of-sample (the strong result):* the simulator was calibrated
   using **only** the baseline (2-doctor) scenario. Re-running it with
   1, 3, and 4 doctors — configurations it never saw — and comparing
   against those scenarios' real recorded means:

   | Scenario (never used to calibrate) | Real mean DocWait | Simulated mean DocWait | Error |
   |---|---:|---:|---:|
   | Staffing_1Doc (N=1) | 837.0 min | 841.2 min | 0.5% |
   | Staffing_3Doc (N=3) | 2.6 min | 2.8 min | 8.0% |
   | Staffing_4Doc (N=4) | 0.1 min | 0.1 min | 3.1% |

   Sub-10% error on three held-out configurations, calibrated from a
   single scenario, is strong evidence the simulator has actually
   learned the underlying queueing mechanics rather than memorizing
   the scenario it was fit on.

**What-if scenario 1 — doctor staffing sweep** (150 Monte Carlo reps
per level, 200 patients/replication):

| Doctors | Median mean-DocWait | P10–P90 |
|---:|---:|---|
| 1 | 843.5 min | 808.6 – 871.8 |
| 2 | 175.0 min | 156.8 – 194.9 |
| 3 | 2.4 min | 1.1 – 5.0 |
| 4 | 0.04 min | 0.01 – 0.12 |
| 5 | 0.0 min | 0.0 – 0.0 |

**What-if scenario 2 — peak arrival rate increase** (2 doctors, 150
reps per level):

| Peak rate change | Median mean-DocWait | P10–P90 |
|---:|---:|---|
| +0% | 175.0 min | 156.8 – 194.9 |
| +20% | 263.6 min | 243.9 – 281.3 |
| +50% | 353.8 min | 336.9 – 368.3 |
| +100% | 439.2 min | 421.1 – 455.9 |

Outputs: `outputs/simulation_validation.json`,
`outputs/simulation_cross_scenario_validation.json`,
`outputs/whatif_staffing_*.csv/json`, `outputs/whatif_peakrate_*.csv/json`.

---

## Step 7 — Dashboards

- **`app_patient.py`** — enter priority + current queue state, or
  load a sample patient from a held-out run; shows predicted
  DocWaitTime with a 90% range and SHAP factors, plus an
  optional pharmacy-stage estimate.
- **`app_admin.py`** — three tabs: utilization/bottleneck view (Plotly
  box + bar charts across scenarios), a prediction-explanation
  browser (pick any held-out run/patient and see the SHAP breakdown),
  and a what-if form that calls `simulate.py` live and renders a
  before/after Monte Carlo box-plot comparison.

Both were smoke-tested (`streamlit run ... --server.headless true`,
verified HTTP 200 and clean startup logs).

---

## Limitations

This is a deliberate scope decision, not an oversight: **no-show
behavior, emergency-patient injection, and appointment-vs-walk-in
distinctions are not modeled anywhere in this pipeline**, because
`hospital_sim_6_scenarios.csv` doesn't contain the fields needed to
observe or fit any of them (there's no appointment flag, no
cancellation/no-show indicator, and every patient in the log
completes both stages — there's no evidence of emergency preemption
in the wait-time patterns). Extending the simulator to cover any of
these would require either a richer dataset or explicit, clearly-
labeled assumptions injected into the SimPy model — neither of which
this project does implicitly. Two smaller, self-imposed limitations
worth naming: the unsupervised scenario clustering (Step 1) tops out
at 78.6% purity because two pairs of scenarios are genuinely close in
aggregate-wait-time feature space; and the peak/non-peak windowed
classifier (81.1% accuracy) will always show residual error against a
hard step-function ground truth, by construction of using a smoothed
window.

---

## Project layout

```
data/hospital_sim_6_scenarios.csv
eda.py                  # Step 1
features.py             # Step 2
models/
  common.py             # shared split/metric utilities
  train_doc_wait.py      # Step 3a
  train_med_wait.py      # Step 3b
  train_total.py          # Step 3c
utilization.py           # Step 4
explain.py               # Step 5
simulate.py               # Step 6
app_patient.py             # Step 7 (patient view)
app_admin.py                # Step 7 (admin view)
outputs/                    # all generated artifacts (models, metrics, plots, predictions)
requirements.txt
```
