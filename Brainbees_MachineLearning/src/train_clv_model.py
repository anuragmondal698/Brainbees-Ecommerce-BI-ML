
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "customer_clv_training_data.csv"

MODEL_FILE = MODEL_DIR / "customer_clv_model.joblib"
FEATURES_FILE = MODEL_DIR / "customer_clv_feature_columns.joblib"
EVALUATION_FILE = DATA_DIR / "customer_clv_model_evaluation.csv"
TEST_PREDICTIONS_FILE = DATA_DIR / "customer_clv_test_predictions.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Training data not found: {DATA_FILE}\n"
        "Run prepare_clv_data.py first."
    )

df = pd.read_csv(DATA_FILE)
df["Snapshot_Date"] = pd.to_datetime(
    df["Snapshot_Date"], errors="coerce"
)

target = "Future_Revenue_12M"

features = [
    "Total_Orders_12M",
    "Revenue_12M",
    "Profit_12M",
    "Units_12M",
    "Average_Order_Value_12M",
    "Recency_Days",
    "Active_Months_12M",
]

required = [
    "Customer_ID",
    "Snapshot_Date",
    target,
] + features

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[target] = pd.to_numeric(df[target], errors="coerce")

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["Customer_ID", "Snapshot_Date", target]
).copy()

if len(df) < 20:
    raise ValueError(
        f"Only {len(df)} usable training rows were found. "
        "Collect more suitable historical snapshots before training."
    )

if df[target].nunique() < 2:
    raise ValueError(
        "Future revenue has fewer than two unique values. "
        "A regression model cannot learn meaningful variation."
    )

print("=" * 65)
print("BRAINBEES CUSTOMER VALUE MODEL")
print("=" * 65)
print(f"Training rows: {len(df):,}")
print(f"Customers: {df['Customer_ID'].nunique():,}")
print(f"Snapshots: {df['Snapshot_Date'].nunique():,}")
print("\nTarget summary:")
print(df[target].describe().to_string())


# ============================================================
# 3. TIME-BASED TRAIN/TEST SPLIT
# ============================================================

# Reserve the latest 20% of distinct snapshots for testing.
snapshot_dates = sorted(df["Snapshot_Date"].unique())

if len(snapshot_dates) < 2:
    raise ValueError(
        "Only one historical snapshot exists. "
        "A time-based holdout cannot be created."
    )

split_position = max(1, int(len(snapshot_dates) * 0.8))

if split_position >= len(snapshot_dates):
    split_position = len(snapshot_dates) - 1

test_start_date = pd.Timestamp(snapshot_dates[split_position])

train_df = df[df["Snapshot_Date"] < test_start_date].copy()
test_df = df[df["Snapshot_Date"] >= test_start_date].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "The time-based split produced an empty train or test set."
    )

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print(f"\nTraining period ends before: {test_start_date.date()}")
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")


# ============================================================
# 4. DEFINE BASELINE AND RANDOM FOREST
# ============================================================

models = {
    "Mean_Baseline": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyRegressor(strategy="mean")),
    ]),
    "Random_Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}


# ============================================================
# 5. TRAIN AND EVALUATE
# ============================================================

results = []
trained_models = {}
test_predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Revenue cannot be negative; clip predictions at zero.
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else np.nan

    results.append({
        "Model_Name": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Train_Rows": len(train_df),
        "Test_Rows": len(test_df),
        "Test_Start_Date": test_start_date.date(),
    })

    trained_models[name] = model
    test_predictions[name] = predictions

    print(f"\n--- {name} ---")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}" if not np.isnan(r2) else "R²: unavailable")


evaluation = pd.DataFrame(results)
evaluation.to_csv(EVALUATION_FILE, index=False)

print("\nModel comparison:")
print(evaluation.to_string(index=False))


# ============================================================
# 6. SELECT MODEL
# ============================================================

# Select the model with the lowest test MAE.
best_row = evaluation.sort_values("MAE").iloc[0]
best_name = best_row["Model_Name"]

print(f"\nSelected model by test MAE: {best_name}")

# Refit the selected model on all labelled historical rows
# for scoring current customers.
best_model = trained_models[best_name]
best_model.fit(df[features], df[target])


# ============================================================
# 7. SAVE MODEL ARTIFACT
# ============================================================

artifact = {
    "model_name": best_name,
    "model": best_model,
    "feature_columns": features,
    "target_column": target,
    "target_horizon_months": 12,
    "target_definition": "Revenue over the following 12 months",
    "test_start_date": str(test_start_date.date()),
    "evaluation_mae": float(best_row["MAE"]),
}

joblib.dump(artifact, MODEL_FILE)
joblib.dump(features, FEATURES_FILE)


# ============================================================
# 8. SAVE HELD-OUT TEST PREDICTIONS
# ============================================================

test_output = test_df[
    ["Customer_ID", "Snapshot_Date", target]
].copy()

test_output["Predicted_Future_Revenue_12M"] = test_predictions[
    best_name
]

test_output["Absolute_Error"] = abs(
    test_output[target]
    - test_output["Predicted_Future_Revenue_12M"]
)

test_output["Model_Name"] = best_name

test_output.to_csv(TEST_PREDICTIONS_FILE, index=False)

print("\nFiles saved:")
print(MODEL_FILE)
print(FEATURES_FILE)
print(EVALUATION_FILE)
print(TEST_PREDICTIONS_FILE)

print("\nCLV model training completed.")
print(
    "\nNote: The prediction target is future 12-month revenue, "
    "not discounted lifetime profit."
)
