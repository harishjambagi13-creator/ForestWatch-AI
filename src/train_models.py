from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "forest_fire_risk.csv"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Dataset missing. Run: python src/generate_dataset.py")

df = pd.read_csv(DATA)
features = [
    "temperature_c", "relative_humidity_pct", "wind_speed_kmh",
    "rainfall_7d_mm", "drought_days", "vegetation_dryness_pct",
    "slope_deg", "forest_fuel_load_pct", "month"
]
X, y = df[features], df["risk_level"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=220, max_depth=14, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ]),
    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=220, max_depth=14, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ]),
    "Hist Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=20, random_state=42
        ))
    ])
}

results = []
fitted = {}
class_names = ["Low", "Moderate", "High"]
for name, pipe in models.items():
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    classes = list(pipe.classes_)
    # Ensure multiclass AUC uses the class order associated with predict_proba.
    y_bin = label_binarize(y_test, classes=classes)
    try:
        auc_score = roc_auc_score(y_bin, proba, average="weighted", multi_class="ovr")
    except ValueError:
        auc_score = float("nan")
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "ROC_AUC_ovr_weighted": auc_score
    }
    results.append(row)
    fitted[name] = pipe
    safe = name.lower().replace(" ", "_")
    joblib.dump(pipe, MODELS / f"{safe}.joblib")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=class_names, display_labels=class_names,
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT / f"{safe}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    (OUT / f"{safe}_classification_report.txt").write_text(
        classification_report(y_test, pred, labels=class_names, zero_division=0),
        encoding="utf-8"
    )

results_df = pd.DataFrame(results).sort_values("F1_weighted", ascending=False)
results_df.to_csv(OUT / "model_comparison.csv", index=False)

# Metrics comparison plot
fig, ax = plt.subplots(figsize=(11, 6))
plot_df = results_df.set_index("Model")[["Accuracy", "Precision_weighted", "Recall_weighted", "F1_weighted"]]
plot_df.columns = ["Accuracy", "Precision", "Recall", "F1"]
plot_df.plot(kind="bar", ax=ax, ylim=(0, 1.05))
ax.set_ylabel("Score")
ax.set_title("ForestWatch AI — Model Performance Comparison")
ax.tick_params(axis="x", rotation=15)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "model_comparison.png", dpi=160)
plt.close(fig)

# One-vs-rest ROC curves for best model (by weighted F1).
best_name = results_df.iloc[0]["Model"]
best_pipe = fitted[best_name]
proba = best_pipe.predict_proba(X_test)
classes = list(best_pipe.classes_)
y_bin = label_binarize(y_test, classes=classes)
fig, ax = plt.subplots(figsize=(8, 6))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
    ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"One-vs-Rest ROC Curves — {best_name}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "roc_curves_best_model.png", dpi=160)
plt.close(fig)

# Feature importance for tree-based best model, when available.
model_obj = best_pipe.named_steps["model"]
if hasattr(model_obj, "feature_importances_"):
    imp = pd.Series(model_obj.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot(kind="barh", ax=ax)
    ax.set_title(f"Feature Importance — {best_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=160)
    plt.close(fig)

summary = {
    "best_model_by_weighted_f1": str(best_name),
    "best_weighted_f1": float(results_df.iloc[0]["F1_weighted"]),
    "best_accuracy": float(results_df.iloc[0]["Accuracy"]),
    "dataset_rows": int(len(df)),
    "features": features,
    "classes": class_names,
    "note": "Results are from synthetic educational data and are not a real-world fire forecast."
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nModel comparison:")
print(results_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nBest by weighted F1: {best_name}")
print("Saved models to models/ and plots/reports to outputs/.")
