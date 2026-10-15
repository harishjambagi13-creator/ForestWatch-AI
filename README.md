# 🌲 ForestWatch AI — Forest Fire Risk Prediction

An advanced college mini-project demonstrating multiclass supervised learning for a **synthetic forest-fire risk dataset**. It is distinct from crop recommendation: inputs describe weather, recent rainfall, drought, vegetation dryness, terrain and fuel load; output is Low / Moderate / High demonstration risk.

> **Academic safety note:** The included data is synthetic. This is not a validated fire forecasting model, does not represent a specific forest or region, and must not be used for operational warnings or emergency decisions.

## Included
- 5,000-row reproducible synthetic dataset
- Dataset-generation script
- Four ML algorithms: Logistic Regression, Random Forest, Extra Trees, Hist Gradient Boosting
- Stratified train/test split
- Accuracy, weighted precision/recall/F1, multiclass one-vs-rest ROC-AUC
- Confusion matrices and classification reports for each model
- Performance comparison graph
- ROC curves for best model by weighted F1
- Feature-importance plot when supported by the selected best model
- Saved models (`.joblib`)
- Streamlit prediction dashboard
- Terminal prediction script
- README, requirements and Windows launcher

## Folder structure
```text
ForestWatch_AI/
├── app.py
├── requirements.txt
├── RUN_ON_WINDOWS.bat
├── README.md
├── data/
│   └── forest_fire_risk.csv
├── src/
│   ├── generate_dataset.py
│   ├── train_models.py
│   └── predict.py
├── models/
│   └── (generated .joblib models)
└── outputs/
    └── (metrics, plots, reports generated during training)
```

## Run on Windows with VS Code

1. Extract the ZIP.
2. Open the extracted `ForestWatch_AI` folder in VS Code.
3. Choose **Terminal → New Terminal**.
4. Create and activate an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install requirements:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Generate the dataset again if needed:

```powershell
python src\generate_dataset.py
```

7. Train and evaluate models:

```powershell
python src\train_models.py
```

Training creates model files in `models/` and plots/reports in `outputs/`. Wait for the comparison table and the final success message before launching the app.

8. Launch the dashboard:

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

### Terminal predictor
After training, run:

```powershell
python src\predict.py
```

### One-click Windows launcher
Double-click `RUN_ON_WINDOWS.bat`. It installs packages, trains the models, and starts the Streamlit app. If PowerShell execution policy blocks environment activation, use the manual commands above.

## Features used
- Temperature (°C)
- Relative humidity (%)
- Wind speed (km/h)
- Rainfall over the last 7 days (mm)
- Days since meaningful rainfall
- Vegetation dryness (%)
- Slope (degrees)
- Forest fuel load (%)
- Month

## Algorithms
- **Logistic Regression:** interpretable linear baseline.
- **Random Forest:** many decision trees combined using bagging.
- **Extra Trees:** randomized tree ensemble.
- **Hist Gradient Boosting:** sequential boosting for nonlinear patterns.

## Methodology
1. Load the generated CSV.
2. Split into training and test data with stratification.
3. Fit four classifiers.
4. Calculate test metrics and reports.
5. Save confusion matrices, ROC curves and model-comparison plots.
6. Select the dashboard's demonstration predictor (Random Forest).
7. Predict a risk category from user-entered conditions.

## Viva questions
**Why is it a classification problem?**  
The target has three categories: Low, Moderate and High.

**Why compare several algorithms?**  
Different models learn patterns differently; test metrics allow a consistent comparison.

**Why not rely only on accuracy?**  
Precision, recall and F1 provide additional information about false alarms and missed risk classes.

**What is a confusion matrix?**  
A table showing true versus predicted classes.

**What does ROC-AUC mean here?**  
It summarizes one-vs-rest class-ranking performance across thresholds.

**What is the key limitation?**  
The labels are synthetic, not validated against real wildfire events. High demo accuracy does not imply real-world fire prediction quality.

## Future work
- Replace synthetic data with documented historical fire and weather records.
- Validate on a separate region and later time period.
- Add location-aware official weather and fire-danger indices.
- Calibrate probabilities and conduct domain-expert validation.
- Add explainability (SHAP) and monitoring for data drift.
