
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "monthly_orders.csv"
MODEL_PATH = BASE_DIR / "models" / "order_forecasting_model.joblib"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "order_forecast_12_months.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["Order_Month"] = pd.to_datetime(
    df["Order_Month"], errors="raise"
)

df["Total_Orders"] = pd.to_numeric(
    df["Total_Orders"], errors="raise"
)

df = df.sort_values("Order_Month").reset_index(drop=True)

if df.empty:
    raise ValueError("Monthly order dataset is empty.")

if df["Total_Orders"].isna().any():
    raise ValueError("Missing historical order counts detected.")

if (df["Total_Orders"] < 0).any():
    raise ValueError("Negative historical order counts detected.")

if df["Order_Month"].duplicated().any():
    raise ValueError("Duplicate historical months detected.")

expected_months = pd.date_range(
    start=df["Order_Month"].min(),
    end=df["Order_Month"].max(),
    freq="MS",
)

actual_months = pd.DatetimeIndex(
    df["Order_Month"].dt.to_period("M").dt.to_timestamp()
)

if not actual_months.equals(expected_months):
    raise ValueError("Historical months are not consecutive.")

artifact = joblib.load(MODEL_PATH)

model_name = artifact["model_name"]
model = artifact["model"]
feature_columns = artifact["feature_columns"]

if model is None:
    raise ValueError(
        f"Selected model is {model_name}, but no trained "
        "model object was saved. Check the evaluation results."
    )

print("Selected model:", model_name)
print("Last historical month:", df["Order_Month"].max().date())


# --------------------------------------------------
# 3. PREPARE RECURSIVE FORECAST
# --------------------------------------------------

history = df["Total_Orders"].astype(float).tolist()

last_month = df["Order_Month"].max()
last_time_index = int(df["Time_Index"].iloc[-1])

forecast_rows = []

for step in range(1, 13):

    forecast_month = (
        last_month + pd.DateOffset(months=step)
    )

    time_index = last_time_index + step
    month_number = forecast_month.month

    lag_1 = history[-1]
    lag_2 = history[-2]
    lag_3 = history[-3]

    rolling_mean_3 = float(np.mean(history[-3:]))

    input_row = {
        "Month_Number": month_number,
        "Time_Index": time_index,
        "Orders_Lag_1": lag_1,
        "Orders_Lag_2": lag_2,
        "Orders_Lag_3": lag_3,
        "Orders_Rolling_Mean_3": rolling_mean_3,
    }

    input_df = pd.DataFrame([input_row])[feature_columns]

    prediction = float(model.predict(input_df)[0])

    # Order counts cannot be negative or fractional.
    predicted_orders = max(0, int(round(prediction)))

    forecast_rows.append({
        "Forecast_Month": forecast_month.strftime("%Y-%m-%d"),
        "Predicted_Orders": predicted_orders,
        "Model_Name": model_name,
    })

    # Add prediction to history for the next month's features.
    history.append(predicted_orders)


# --------------------------------------------------
# 4. SAVE FORECAST
# --------------------------------------------------

forecast_df = pd.DataFrame(forecast_rows)

if len(forecast_df) != 12:
    raise ValueError("Expected exactly 12 forecast months.")

if forecast_df["Forecast_Month"].duplicated().any():
    raise ValueError("Duplicate forecast months generated.")

forecast_df.to_csv(OUTPUT_PATH, index=False)

print("\n========== 12-MONTH ORDER FORECAST ==========")
print(forecast_df.to_string(index=False))

print("\nTotal predicted orders across 12 months:",
      int(forecast_df["Predicted_Orders"].sum()))

print("\nForecast saved to:")
print(OUTPUT_PATH)
