"""
Step 7 -- Patient dashboard.

Enter a priority + current queue state (or pick a sample patient from
the data) and see a predicted wait, its range, and a plain-language
explanation of what's driving it.
"""
import pandas as pd
import streamlit as st

from explain import WaitExplainer

st.set_page_config(page_title="Patient Wait Estimator", page_icon="🏥", layout="centered")

st.title("🏥 Estimated Wait Time")
st.caption("Enter your details, or load a sample patient from a real simulated run.")


@st.cache_resource
def get_explainers():
    return WaitExplainer("doc"), WaitExplainer("med")


@st.cache_data
def get_sample_patients():
    doc = pd.read_parquet("outputs/test_predictions_doc_wait.parquet")
    med = pd.read_parquet("outputs/test_predictions_med_wait.parquet")
    return doc, med


doc_exp, med_exp = get_explainers()
doc_samples, med_samples = get_sample_patients()

mode = st.radio("How would you like to start?", ["Enter my details", "Load a sample patient"], horizontal=True)

if mode == "Load a sample patient":
    idx = st.selectbox("Pick a sample patient (from held-out simulated runs)",
                        doc_samples.index, format_func=lambda i: f"Patient {doc_samples.loc[i, 'PatientID']} "
                                                                   f"(run {doc_samples.loc[i, 'RunKey']})")
    row = doc_samples.loc[idx]
    priority = int(row["Priority"])
    num_doctors = int(row["NumActiveDoctors_recovered"])
    is_peak = bool(row["Period_peak_flag"])
    queue_for_doctor = int(row["queue_for_doctor_at_arrival"])
    doctors_busy = int(row["doctors_busy_at_arrival"])
    st.info(f"Actual recorded wait for this patient was {row['DocWaitTime']:.0f} min.")
else:
    priority = st.select_slider("Priority", options=[1, 2, 3],
                                 format_func=lambda p: {1: "Low", 2: "Medium", 3: "High"}[p])
    num_doctors = st.slider("Active doctors right now", 1, 4, 2)
    is_peak = st.checkbox("Currently peak hours?", value=True)
    queue_for_doctor = st.number_input("Patients already waiting for a doctor", min_value=0, max_value=50, value=3)
    doctors_busy = st.number_input("Doctors currently busy", min_value=0, max_value=num_doctors, value=min(num_doctors, 2))

doc_row = pd.Series({
    "Priority": priority,
    "NumActiveDoctors_recovered": num_doctors,
    "Period_peak_flag": int(is_peak),
    "queue_for_doctor_at_arrival": queue_for_doctor,
    "doctors_busy_at_arrival": doctors_busy,
})

result = doc_exp.explain(doc_row)

st.metric("Expected doctor wait", f"{result['prediction']:.0f} min",
          help=f"90% prediction range: {result['range'][0]:.0f}-{result['range'][1]:.0f} min")
st.write(f"**Range:** {result['range'][0]:.0f}\u2013{result['range'][1]:.0f} min (90% confidence)")
st.write("**Main factors:**")
for factor in result["factors"]:
    st.write(f"- {factor}")

with st.expander("What about the pharmacy stage afterward?"):
    med_row = pd.Series({
        "Priority": priority,
        "Period_peak_flag": int(is_peak),
        "pharmacy_queue_at_doc_end": 0,
        "pharmacy_busy_at_doc_end": 0,
    })
    med_result = med_exp.explain(med_row)
    st.write(med_result["sentence"])
    st.caption("Pharmacy wait assumes you arrive there with no one currently ahead of you; "
               "actual pharmacy congestion depends on how many patients finish their doctor visit "
               "around the same time as you.")

st.divider()
st.caption("Predictions come from an XGBoost model with conformal (mapie) prediction intervals, "
           "trained on simulated hospital visit data. This is a demo, not medical advice.")
