
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR / "data" / "processed"
    / "customer_churn_scoring_data.csv"
)

MODEL_FILE = BASE_DIR / "models" / "customer_churn_model.joblib"

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "customer_churn_predictions_2_years.csv"
)


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}. "
        "Run train_churn_model.py first."
    )

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Scoring data not found: {DATA_FILE}. "
        "Run prepare_churn_data.py first."
    )

artifact = joblib.load(MODEL_FILE)
model = artifact["model"]
features = artifact["feature_columns"]

df = pd.read_csv(DATA_FILE)

if df.empty:
    raise ValueError("The customer scoring dataset is empty.")

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing model features: {missing}")

X = df[features].apply(pd.to_numeric, errors="coerce")


# ============================================================
# 3. GENERATE PREDICTIONS
# ============================================================

df["Churn_Probability"] = model.predict_proba(X)[:, 1]
df["Predicted_Churn_24_Months"] = (
    df["Churn_Probability"] >= 0.50
).astype(int)

df["Risk_Level"] = pd.cut(
    df["Churn_Probability"],
    bins=[-0.001, 0.40, 0.70, 1.001],
    labels=["Low", "Medium", "High"],
).astype("string")

df["Model_Name"] = artifact["model_name"]
df["Forecast_Horizon_Months"] = 24

df["Prediction_Date"] = pd.to_datetime(
    df["Snapshot_Date"], errors="coerce"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("CUSTOMER CHURN PREDICTIONS COMPLETED")
print("=" * 60)
print(f"Customers scored: {len(df):,}")
print("\nRisk distribution:")
print(df["Risk_Level"].value_counts(dropna=False))
print(f"\nOutput saved to:\n{OUTPUT_FILE}")

print(
    "\nImportant: these are model-generated probabilities. "
    "Their reliability depends on the quality and representativeness "
    "of the historical churn labels."
)
