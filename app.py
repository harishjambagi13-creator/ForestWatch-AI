from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
RESULTS_PATH = ROOT / "outputs" / "model_comparison.csv"
OUT = ROOT / "outputs"

st.set_page_config(page_title="ForestWatch AI", page_icon="🌲", layout="wide")
st.title("🌲 ForestWatch AI")
st.subheader("Forest Fire Risk Prediction Using Machine Learning")
st.write("Estimate a demonstration risk category from weather, dryness and terrain features.")

if not MODEL_PATH.exists():
    st.error("Trained model is missing. In the project terminal, run: `python src/train_models.py`")
    st.stop()

with st.sidebar:
    st.header("🌦️ Conditions")
    temperature_c = st.slider("Temperature (°C)", 0.0, 50.0, 34.0)
    relative_humidity_pct = st.slider("Relative humidity (%)", 5.0, 100.0, 28.0)
    wind_speed_kmh = st.slider("Wind speed (km/h)", 0.0, 60.0, 24.0)
    rainfall_7d_mm = st.slider("Rainfall in last 7 days (mm)", 0.0, 250.0, 2.0)
    drought_days = st.slider("Days since meaningful rainfall", 0, 90, 22)
    vegetation_dryness_pct = st.slider("Vegetation dryness (%)", 0.0, 100.0, 78.0)
    slope_deg = st.slider("Slope (degrees)", 0.0, 60.0, 15.0)
    forest_fuel_load_pct = st.slider("Forest fuel load (%)", 0.0, 100.0, 65.0)
    month = st.selectbox("Month", list(range(1, 13)), index=2)

st.markdown("### 1. Predict fire-risk category")
if st.button("🔥 Predict Risk", type="primary", use_container_width=True):
    row = {
        "temperature_c": temperature_c,
        "relative_humidity_pct": relative_humidity_pct,
        "wind_speed_kmh": wind_speed_kmh,
        "rainfall_7d_mm": rainfall_7d_mm,
        "drought_days": drought_days,
        "vegetation_dryness_pct": vegetation_dryness_pct,
        "slope_deg": slope_deg,
        "forest_fuel_load_pct": forest_fuel_load_pct,
        "month": month
    }
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([row])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    if prediction == "High":
        st.error(f"## HIGH DEMONSTRATION RISK")
        st.warning("Review local official fire-weather alerts, avoid ignition sources, and follow forestry/emergency guidance.")
    elif prediction == "Moderate":
        st.warning("## MODERATE DEMONSTRATION RISK")
        st.write("Conditions may support fire spread. Continue monitoring official local guidance.")
    else:
        st.success("## LOW DEMONSTRATION RISK")
        st.write("The model estimates comparatively lower risk under the entered conditions; risk can still change.")
    c1, c2 = st.columns(2)
    c1.metric("Predicted category", prediction)
    c2.metric("Highest class probability", f"{confidence:.1%}")
    st.markdown("**Class probability estimates**")
    st.dataframe(pd.DataFrame({"Risk level": model.classes_, "Probability": probabilities})
                  .sort_values("Probability", ascending=False)
                  .assign(Probability=lambda d: d["Probability"].map(lambda x: f"{x:.1%}")),
                  use_container_width=True, hide_index=True)

st.divider()
st.markdown("### 2. Compare ML algorithms")
if RESULTS_PATH.exists():
    result = pd.read_csv(RESULTS_PATH)
    st.dataframe(result.style.format({c: "{:.3f}" for c in result.columns if c != "Model"}),
                 use_container_width=True, hide_index=True)
    st.image(str(OUT / "model_comparison.png"), caption="Performance comparison")
    if (OUT / "roc_curves_best_model.png").exists():
        st.image(str(OUT / "roc_curves_best_model.png"), caption="ROC curves for the best model")
    if (OUT / "feature_importance.png").exists():
        st.image(str(OUT / "feature_importance.png"), caption="Feature importance")
else:
    st.info("Train models with `python src/train_models.py` to generate comparison results and plots.")

with st.expander("About the project and limitations"):
    st.write("""
    **Models:** Logistic Regression baseline, Random Forest, Extra Trees and Hist Gradient Boosting.

    **Metrics:** Accuracy, weighted precision, weighted recall, weighted F1 and one-vs-rest ROC-AUC.
    The project also saves confusion matrices and classification reports.

    **Important limitation:** The included dataset is synthetic and intended only for an academic demonstration.
    It is not trained on verified historical wildfire records, cannot provide real geographic forecasts, and must not be used as an official warning or emergency decision tool.
    """)
