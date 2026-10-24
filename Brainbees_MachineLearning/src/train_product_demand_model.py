
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "monthly_product_demand.csv"
)

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD AND VALIDATE DATA
# =====================================================

df = pd.read_csv(DATA_PATH)

required = [
    "Product_ID",
    "Demand_Month",
    "Units_Sold",
    "Month_Number",
    "Time_Index",
    "Demand_Lag_1",
    "Demand_Lag_2",
    "Demand_Lag_3",
    "Demand_Rolling_Mean_3",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["Demand_Month"] = pd.to_datetime(
    df["Demand_Month"], errors="raise"
)

df["Units_Sold"] = pd.to_numeric(
    df["Units_Sold"], errors="raise"
)

df = df.sort_values(
    ["Product_ID", "Demand_Month"]
).reset_index(drop=True)

if df.duplicated(["Product_ID", "Demand_Month"]).any():
    raise ValueError("Duplicate product-month rows found.")

if df["Units_Sold"].isna().any() or (df["Units_Sold"] < 0).any():
    raise ValueError("Invalid demand values found.")


# =====================================================
# 3. PREPARE MODEL DATA
# =====================================================

features = [
    "Month_Number",
    "Time_Index",
    "Demand_Lag_1",
    "Demand_Lag_2",
    "Demand_Lag_3",
    "Demand_Rolling_Mean_3",
]

target = "Units_Sold"

model_data = df.dropna(subset=features + [target]).copy()

if model_data.empty:
    raise ValueError("No usable rows remain after preparing features.")

# Split chronologically: reserve the latest 20% of
# distinct months for testing across all products.
all_months = sorted(model_data["Demand_Month"].unique())

if len(all_months) < 12:
    raise ValueError(
        f"Only {len(all_months)} usable months found. "
        "More historical data is recommended."
    )

split_position = max(1, int(len(all_months) * 0.8))
split_position = min(split_position, len(all_months) - 1)
test_start = all_months[split_position]

train = model_data[model_data["Demand_Month"] < test_start].copy()
test = model_data[model_data["Demand_Month"] >= test_start].copy()

if train.empty or test.empty:
    raise ValueError("Chronological train/test split failed.")

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\n========== DATASET SUMMARY ==========")
print("Products:", df["Product_ID"].nunique())
print("Total product-month rows:", len(df))
print("Usable rows:", len(model_data))
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Test starts:", pd.Timestamp(test_start).strftime("%Y-%m"))


# =====================================================
# 4. BASELINE: PREVIOUS MONTH'S DEMAND
# =====================================================

baseline_predictions = test["Demand_Lag_1"].to_numpy()


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Avoid division by zero in MAPE.
    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_Percent": mape,
    }


baseline_metrics = metrics(y_test, baseline_predictions)


# =====================================================
# 5. TRAIN RANDOM FOREST
# =====================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

rf_predictions = np.maximum(model.predict(X_test), 0)
rf_metrics = metrics(y_test, rf_predictions)


# =====================================================
# 6. COMPARE MODELS
# =====================================================

evaluation = pd.DataFrame([
    {"Model": "Naive_Last_Month", **baseline_metrics},
    {"Model": "RandomForestRegressor", **rf_metrics},
])

evaluation_path = OUTPUT_DIR / "product_demand_model_evaluation.csv"
evaluation.to_csv(evaluation_path, index=False)

print("\n========== MODEL EVALUATION ==========")
print(evaluation.to_string(index=False))

# Select model based on lower holdout MAE.
# The holdout is preliminary; further backtesting is
# recommended before operational use.
if rf_metrics["MAE"] < baseline_metrics["MAE"]:
    selected_name = "RandomForestRegressor"
    selected_model = model
else:
    selected_name = "Naive_Last_Month"
    selected_model = None

artifact = {
    "model_name": selected_name,
    "model": selected_model,
    "feature_columns": features,
    "target_column": target,
}

model_path = MODEL_DIR / "product_demand_model.joblib"
joblib.dump(artifact, model_path)

prediction_output = test[
    ["Product_ID", "Demand_Month", "Units_Sold"]
].copy()

prediction_output["Baseline_Predicted_Units"] = baseline_predictions
prediction_output["RandomForest_Predicted_Units"] = rf_predictions
prediction_output["Selected_Model"] = selected_name

prediction_path = OUTPUT_DIR / "product_demand_test_predictions.csv"
prediction_output.to_csv(prediction_path, index=False)

print("\nSelected model:", selected_name)
print("Saved model:", model_path)
print("Saved evaluation:", evaluation_path)
print("Saved test predictions:", prediction_path)
print("\nTraining and evaluation completed.")
