
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "monthly_revenue_prepared.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["Month_Start"])
df = df.sort_values("Month_Start").reset_index(drop=True)

# Forecast revenue using calendar features and past revenue.
df["month_number"] = df["Month_Start"].dt.month
df["time_index"] = np.arange(len(df))
df["revenue_lag_1"] = df["Revenue"].shift(1)
df["revenue_lag_2"] = df["Revenue"].shift(2)
df["revenue_lag_3"] = df["Revenue"].shift(3)
df["revenue_rolling_mean_3"] = df["Revenue"].shift(1).rolling(3).mean()

feature_columns = [
    "month_number",
    "time_index",
    "revenue_lag_1",
    "revenue_lag_2",
    "revenue_lag_3",
    "revenue_rolling_mean_3",
]

model_df = df.dropna(subset=feature_columns + ["Revenue"]).copy()

if len(model_df) < 12:
    raise ValueError(
        "Too few usable months after creating lag features. "
        "Review the source data before training."
    )

# Keep the last 20% of months for chronological testing.
test_size = max(4, int(np.ceil(len(model_df) * 0.20)))
train_df = model_df.iloc[:-test_size].copy()
test_df = model_df.iloc[-test_size:].copy()

X_train = train_df[feature_columns]
y_train = train_df["Revenue"]
X_test = test_df[feature_columns]
y_test = test_df["Revenue"]

# Baseline: last month's revenue as the forecast.
baseline_predictions = test_df["revenue_lag_1"].to_numpy()

# Random Forest model.
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_percent": (
            np.mean(np.abs((actual - predicted) / actual)) * 100
            if np.all(actual != 0) else np.nan
        ),
    }

baseline_metrics = calculate_metrics(
    y_test.to_numpy(), baseline_predictions
)
model_metrics = calculate_metrics(
    y_test.to_numpy(), predictions
)

print("\n=== REVENUE FORECAST MODEL ===")
print("Training months:", len(train_df))
print("Testing months:", len(test_df))

print("\nBaseline metrics:")
for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.2f}")

print("\nRandom Forest metrics:")
for metric, value in model_metrics.items():
    print(f"{metric}: {value:.2f}")

results = test_df[["Month_Start", "Revenue"]].copy()
results["Baseline_Prediction"] = baseline_predictions
results["RandomForest_Prediction"] = predictions
results.to_csv(
    OUTPUT_DIR / "revenue_model_evaluation.csv",
    index=False,
    encoding="utf-8-sig",
)

joblib.dump(model, MODEL_DIR / "revenue_random_forest.joblib")
joblib.dump(feature_columns, MODEL_DIR / "revenue_feature_columns.joblib")

print("\nSaved model:", MODEL_DIR / "revenue_random_forest.joblib")
print("Saved evaluation:", OUTPUT_DIR / "revenue_model_evaluation.csv")
print("\nTraining and evaluation completed.")
