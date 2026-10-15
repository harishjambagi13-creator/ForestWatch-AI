from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "random_forest.joblib"

def ask(label, default, cast=float):
    raw = input(f"{label} [{default}]: ").strip()
    return cast(raw) if raw else default

def main():
    if not MODEL.exists():
        print("Model file not found. Run: python src/train_models.py")
        return
    print("\nFORESTWATCH AI — FIRE RISK DEMO")
    print("Enter weather and forest conditions. Press Enter to use sample values.\n")
    row = {
        "temperature_c": ask("Temperature (°C)", 34.0),
        "relative_humidity_pct": ask("Relative humidity (%)", 28.0),
        "wind_speed_kmh": ask("Wind speed (km/h)", 24.0),
        "rainfall_7d_mm": ask("Rainfall in last 7 days (mm)", 2.0),
        "drought_days": ask("Days since meaningful rainfall", 22, int),
        "vegetation_dryness_pct": ask("Vegetation dryness (%)", 78.0),
        "slope_deg": ask("Slope (degrees)", 15.0),
        "forest_fuel_load_pct": ask("Forest fuel load (%)", 65.0),
        "month": ask("Month (1-12)", 3, int)
    }
    model = joblib.load(MODEL)
    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    print(f"\nPredicted risk level: {pred.upper()}")
    print("Model probability estimates:")
    for cls, p in sorted(zip(model.classes_, probs), key=lambda item: item[1], reverse=True):
        print(f"  {cls:<10} {p:.1%}")
    print("\nDemo only: not an official alert or operational fire forecast.")

if __name__ == "__main__":
    main()
