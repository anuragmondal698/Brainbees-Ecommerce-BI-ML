
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "monthly_orders.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run prepare_order_data.py first."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [
    "Order_Month",
    "Total_Orders",
    "Month_Number",
    "Time_Index",
    "Orders_Lag_1",
    "Orders_Lag_2",
    "Orders_Lag_3",
    "Orders_Rolling_Mean_3",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df["Order_Month"] = pd.to_datetime(
    df["Order_Month"], errors="coerce"
)

df["Total_Orders"] = pd.to_numeric(
    df["Total_Orders"], errors="coerce"
)

df = df.sort_values("Order_Month").reset_index(drop=True)

if df["Order_Month"].isna().any():
    raise ValueError("Invalid dates exist in Order_Month.")

if df["Order_Month"].duplicated().any():
    raise ValueError("Duplicate months found in the dataset.")

if df["Total_Orders"].isna().any():
    raise ValueError(
        "Monthly order counts contain missing values. "
        "Resolve missing months before training."
    )

if (df["Total_Orders"] < 0).any():
    raise ValueError("Negative monthly order counts found.")

# Require consecutive monthly observations.
expected_months = pd.date_range(
    start=df["Order_Month"].min(),
    end=df["Order_Month"].max(),
    freq="MS",
)

actual_months = pd.DatetimeIndex(
    df["Order_Month"].dt.to_period("M").dt.to_timestamp()
)

if not actual_months.equals(expected_months):
    raise ValueError(
        "The dataset does not contain a complete monthly sequence."
    )


# --------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# --------------------------------------------------

feature_columns = [
    "Month_Number",
    "Time_Index",
    "Orders_Lag_1",
    "Orders_Lag_2",
    "Orders_Lag_3",
    "Orders_Rolling_Mean_3",
]

target_column = "Total_Orders"

# Initial rows normally lack historical lag features.
model_data = df.dropna(
    subset=feature_columns + [target_column]
).copy()

if len(model_data) < 12:
    raise ValueError(
        f"Only {len(model_data)} usable monthly rows are available. "
        "At least 12 are required by this script; more history is "
        "strongly preferable for reliable evaluation."
    )

X = model_data[feature_columns]
y = model_data[target_column]

# Reserve the latest 20% for testing.
split_index = int(len(model_data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_months = model_data["Order_Month"].iloc[split_index:]

print("\n========== DATASET SUMMARY ==========")
print(f"Total monthly rows: {len(df)}")
print(f"Usable rows after lag features: {len(model_data)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Test period: {test_months.min():%Y-%m} to "
      f"{test_months.max():%Y-%m}")


# --------------------------------------------------
# 4. EVALUATION FUNCTION
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE is undefined when actual values are zero.
    nonzero_mask = actual != 0

    if nonzero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (actual[nonzero_mask] - predicted[nonzero_mask])
                    / actual[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_Percent": mape,
    }


# --------------------------------------------------
# 5. BASELINE: LAST MONTH'S ORDERS
# --------------------------------------------------

baseline_predictions = X_test["Orders_Lag_1"].to_numpy()

baseline_metrics = calculate_metrics(
    y_test, baseline_predictions
)


# --------------------------------------------------
# 6. TRAIN RANDOM FOREST
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_metrics = calculate_metrics(
    y_test, rf_predictions
)


# --------------------------------------------------
# 7. COMPARE MODELS
# --------------------------------------------------

evaluation = pd.DataFrame([
    {
        "Model": "Naive_Last_Month",
        **baseline_metrics,
    },
    {
        "Model": "RandomForestRegressor",
        **rf_metrics,
    },
])

evaluation_path = OUTPUT_DIR / "order_model_evaluation.csv"
evaluation.to_csv(evaluation_path, index=False)

print("\n========== MODEL EVALUATION ==========")
print(evaluation.to_string(index=False))

# Select using test MAE. This is a small holdout, so the
# result is preliminary and should not be treated as proof
# of future performance.
if rf_metrics["MAE"] < baseline_metrics["MAE"]:
    best_model_name = "RandomForestRegressor"
    selected_model = rf_model
else:
    best_model_name = "Naive_Last_Month"
    selected_model = None

print(f"\nSelected model: {best_model_name}")


# --------------------------------------------------
# 8. SAVE THE SELECTED MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "order_forecasting_model.joblib"
features_path = MODEL_DIR / "order_feature_columns.joblib"

artifact = {
    "model_name": best_model_name,
    "model": selected_model,
    "feature_columns": feature_columns,
    "target_column": target_column,
}

joblib.dump(artifact, model_path)
joblib.dump(feature_columns, features_path)


# --------------------------------------------------
# 9. SAVE TEST PREDICTIONS
# --------------------------------------------------

if best_model_name == "RandomForestRegressor":
    best_predictions = rf_predictions
else:
    best_predictions = baseline_predictions

test_results = pd.DataFrame({
    "Order_Month": test_months.to_numpy(),
    "Actual_Orders": y_test.to_numpy(),
    "Baseline_Predicted_Orders": baseline_predictions,
    "RandomForest_Predicted_Orders": rf_predictions,
    "Selected_Model_Predicted_Orders": best_predictions,
})

test_results_path = OUTPUT_DIR / "order_test_predictions.csv"
test_results.to_csv(test_results_path, index=False)

print("\n========== SAVED FILES ==========")
print(f"Model artifact: {model_path}")
print(f"Feature columns: {features_path}")
print(f"Evaluation report: {evaluation_path}")
print(f"Test predictions: {test_results_path}")

print("\nTraining and evaluation completed.")
