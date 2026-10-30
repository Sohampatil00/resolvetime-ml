"""A compact, single-ticket interface for ResolveTime."""

import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

from resolvetime.data import UNKNOWN
from resolvetime.modeling import predict_tickets

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/resolution_model.joblib"

st.set_page_config(page_title="ResolveTime", page_icon="⏱", layout="centered")
st.title("ResolveTime")
st.write("Predict how many hours an IT support ticket may take to resolve.")


@st.cache_resource
def load_artifact(path, modified_ns):
    return joblib.load(path)


def ordered_levels(values):
    """Show category numbers in natural order and missing values last."""
    return sorted(values, key=lambda value: (
        value == UNKNOWN,
        [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)],
    ))


def display_level(value):
    return "Unknown / not provided" if value == UNKNOWN else value


if not MODEL_PATH.exists():
    st.info("Train the model once to start making predictions.")
    commands = "python train.py"
    if not (ROOT / "data/raw/incident_event_log.csv").exists():
        commands = "python download_data.py\n" + commands
    st.code(commands, language="bash")
    st.caption("Run from the resolve_time folder, then refresh this page.")
    st.stop()

try:
    model_modified = MODEL_PATH.stat().st_mtime_ns
    artifact = load_artifact(str(MODEL_PATH), model_modified)
    evaluation = artifact["evaluation"]
    if evaluation["versions"]["scikit_learn"] != sklearn.__version__:
        st.error("Use the environment that trained this model, or retrain it with the current environment.")
        st.stop()
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

levels = artifact["category_levels"]
if "rt_default_opening" not in st.session_state:
    st.session_state["rt_default_opening"] = datetime.now().replace(second=0, microsecond=0)
default_opening = st.session_state["rt_default_opening"]

with st.container(border=True):
    left, right = st.columns(2)
    with left:
        category = st.selectbox(
            "Category", ordered_levels(levels["category"]), format_func=display_level,
            help="The source hides real problem names. These are the original anonymous category codes.",
        )
    with right:
        subcategory = st.selectbox(
            "Subcategory", ordered_levels(artifact["subcategory_by_category"][category]),
            format_func=display_level,
        )
    left, right = st.columns(2)
    with left:
        impact = st.selectbox("Impact", ordered_levels(levels["impact"]), format_func=display_level,
                              help="How seriously the incident affects the service or organization.")
    with right:
        urgency = st.selectbox("Urgency", ordered_levels(levels["urgency"]), format_func=display_level,
                               help="How quickly the incident needs attention.")
    with st.expander("Additional ticket details"):
        contacts = ordered_levels(levels["contact_type"])
        contact = st.selectbox("Reported through", contacts,
                               index=contacts.index("Phone") if "Phone" in contacts else 0,
                               format_func=display_level)
        left, right = st.columns(2)
        with left:
            opened_date = st.date_input("Reported date", value=default_opening.date())
        with right:
            opened_time = st.time_input("Reported time", value=default_opening.time())

    opened = datetime.combine(opened_date, opened_time)
    ticket = {"opened_at": opened.isoformat(sep=" "), "category": category,
              "subcategory": subcategory, "contact_type": contact,
              "impact": impact, "urgency": urgency}
    input_key = (model_modified, *ticket.values())
    # Preserve the result across harmless reruns, but never show it for changed
    # fields or a newly trained model. Additional details are genuine inputs.
    if st.session_state.get("rt_prediction_key") != input_key:
        st.session_state.pop("rt_prediction", None)
    if st.button("Predict resolution time", type="primary", use_container_width=True):
        try:
            prediction = predict_tickets(artifact, pd.DataFrame([ticket])).iloc[0]
            st.session_state["rt_prediction"] = {
                "hours": float(prediction.predicted_resolution_hours), "note": prediction.input_note,
            }
            st.session_state["rt_prediction_key"] = input_key
        except ValueError as error:
            st.error(str(error))

result = st.session_state.get("rt_prediction")
if result:
    with st.container(border=True):
        st.metric("Estimated resolution time", f"{result['hours']:,.1f} hours")
        try:
            expected = pd.Timestamp(opened) + pd.to_timedelta(result["hours"], unit="h")
            st.write(f"Around **{expected:%d %b %Y, %H:%M}**, in your input timezone.")
        except (ValueError, OverflowError):
            pass
        st.caption("An estimate in clock hours, including nights and weekends. Completion time is not guaranteed.")
        if result["note"]:
            st.warning(result["note"])

with st.expander("Model results"):
    st.write(f"Selected model: **{artifact['model_name']}**")
    test = evaluation["test"]
    baseline = evaluation["test_median_baseline"]
    a, b, c = st.columns(3)
    a.metric("Test MAE", f"{test['mae_hours']:.2f} h")
    b.metric("Test RMSE", f"{test['rmse_hours']:.2f} h")
    c.metric("Test R²", f"{test['r2']:.3f}")
    st.caption(f"Measured on {test['tickets']:,} later, held-out resolved tickets. MAE and RMSE: lower is better. R²: higher is better.")
    st.write(f"Median baseline test MAE: **{baseline['mae_hours']:.2f} hours**.")
    if test["mae_hours"] >= baseline["mae_hours"]:
        st.info("The selected model did not beat the simple baseline on test MAE. This version needs improvement.")
    rows = pd.DataFrame(evaluation["validation_comparison"])
    rows = rows.sort_values("mae_hours")[["model", "mae_hours"]].rename(
        columns={"model": "Model", "mae_hours": "Validation MAE (hours)"})
    rows["Validation MAE (hours)"] = rows["Validation MAE (hours)"].map(lambda value: f"{value:,.2f}")
    st.table(rows.set_index("Model"))
    st.caption("Validation chooses the model. The later test set measures its performance.")

with st.expander("How the ML works"):
    st.write("**Task:** Supervised regression — learn from past tickets to predict a number of hours.")
    st.write("**Features (X):** Category, subcategory, impact, urgency, reporting method, and time features.")
    st.write("**Target (y):** Hours between opening and final observed resolution.")
    st.write("**Preprocessing:** Handle unknown values, one-hot encode categories, and standardize numerical features.")
    st.write("**Feature engineering:** Hour, weekday, month, weekend, and the impact–urgency combination.")
    st.write("**Models:** Multiple Linear Regression, k-NN regression, and Gradient Boosting; compare with a median baseline.")
    st.write(f"**Tuning:** Grid search with {evaluation['cv_folds']} expanding time-based cross-validation folds.")
    split = evaluation["split"]
    st.write(f"**Evaluation:** {split['train']:,} initial training tickets, {split['validation']:,} validation tickets, and {split['test']:,} test tickets. Final refit uses {split['final_training']:,} eligible pre-test tickets.")
    st.caption("Time boundaries are based on opening days; these are not equal-sized ticket groups. Future resolution details are excluded from inputs.")
    audit = evaluation["dataset_audit"]
    st.markdown(f"**Data:** {audit['unique_incidents']:,} incidents; {audit['usable_resolved_incidents']:,} usable resolved initial tickets. [UCI dataset](https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log), Amaral, Fantinato and Peres, CC BY 4.0.")
    st.caption("Category names are anonymized. Unresolved cases are excluded. Results from this historical company require validation before use elsewhere.")
