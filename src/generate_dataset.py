from pathlib import Path
import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)
N = 5000

# Synthetic, educational dataset. Labels are generated from a noisy fire-weather score.
temperature = np.clip(rng.normal(29, 7.5, N), 5, 48)
relative_humidity = np.clip(rng.beta(2.2, 2.0, N) * 100, 8, 100)
wind_speed = np.clip(rng.gamma(2.0, 7.0, N), 0, 55)
rain_7d = np.clip(rng.exponential(18, N), 0, 220)
drought_days = np.clip(rng.gamma(2.1, 8.0, N), 0, 90)
vegetation_dryness = np.clip(rng.beta(2.5, 2.0, N) * 100, 0, 100)
slope = np.clip(rng.gamma(2.0, 9.0, N), 0, 55)
forest_fuel_load = np.clip(rng.normal(55, 20, N), 5, 100)
month = rng.integers(1, 13, N)

# Seasonal dryness proxy for a generic region, not a real geographic fire model.
seasonal_factor = np.select(
    [np.isin(month, [1, 2, 3, 4]), np.isin(month, [5, 6, 7, 8]), np.isin(month, [9, 10, 11, 12])],
    [12, -5, 3], default=0
)
score = (
    0.085 * temperature
    + 0.075 * (100 - relative_humidity)
    + 0.105 * wind_speed
    + 0.105 * vegetation_dryness
    + 0.075 * drought_days
    + 0.035 * forest_fuel_load
    + 0.025 * slope
    - 0.065 * np.sqrt(rain_7d + 1)
    + seasonal_factor
    + rng.normal(0, 4.5, N)
)
low_cut, high_cut = np.quantile(score, [0.42, 0.78])
risk = np.where(score < low_cut, "Low", np.where(score < high_cut, "Moderate", "High"))

df = pd.DataFrame({
    "temperature_c": temperature.round(2),
    "relative_humidity_pct": relative_humidity.round(2),
    "wind_speed_kmh": wind_speed.round(2),
    "rainfall_7d_mm": rain_7d.round(2),
    "drought_days": drought_days.round(0).astype(int),
    "vegetation_dryness_pct": vegetation_dryness.round(2),
    "slope_deg": slope.round(2),
    "forest_fuel_load_pct": forest_fuel_load.round(2),
    "month": month,
    "risk_level": risk
})
out = Path(__file__).resolve().parents[1] / "data" / "forest_fire_risk.csv"
df.to_csv(out, index=False)
print(f"Generated {len(df)} synthetic rows: {out}")
print(df["risk_level"].value_counts().to_string())
