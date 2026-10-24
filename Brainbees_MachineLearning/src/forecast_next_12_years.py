from pathlib import Path

import joblib
import pandas as pd

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "monthly_revenue_prepared.csv"
)

MODEL_PATH = (
    BASE_DIR / "models"
    / "revenue_random_forest.joblib"
)

FEATURE_PATH = (
    BASE_DIR / "models"
    / "revenue_feature_columns.joblib"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed"
    / "revenue_forecast_12_years.csv"
)

# --------------------------------------------------
# 2. Load historical data and saved model
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

df["Month_Start"] = pd.to_datetime(df["Month_Start"])
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="raise")

df = (
    df[["Month_Start", "Revenue"]]
    .dropna()
    .sort_values("Month_Start")
    .reset_index(drop=True)
)

if df["Month_Start"].duplicated().any():
    raise ValueError(
        "Duplicate months found. Fix the monthly data before forecasting."
    )

if len(df) < 4:
    raise ValueError("At least four historical months are required.")

# Verify that historical months are consecutive
expected_months = pd.date_range(
    start=df["Month_Start"].min(),
    end=df["Month_Start"].max(),
    freq="MS"
)

if not df["Month_Start"].equals(
    pd.Series(expected_months, name="Month_Start")
):
    raise ValueError(
        "Historical data has missing months or dates not aligned "
        "to month-start. Correct the data before forecasting."
    )

# --------------------------------------------------
# 3. Initialize historical revenue sequence
# --------------------------------------------------

revenues = df["Revenue"].astype(float).tolist()

latest_month = df["Month_Start"].iloc[-1]
next_month = latest_month + pd.offsets.MonthBegin(1)

# Continue the time index used during model training
time_index = len(df) + 1

forecast_rows = []

# --------------------------------------------------
# 4. Forecast the next 144 months recursively
# --------------------------------------------------

for step in range(144):

    forecast_month = next_month + pd.DateOffset(months=step)

    row = {
        "month_number": forecast_month.month,
        "time_index": time_index + step,
        "revenue_lag_1": revenues[-1],
        "revenue_lag_2": revenues[-2],
        "revenue_lag_3": revenues[-3],
        "revenue_rolling_mean_3": sum(revenues[-3:]) / 3
    }

    input_data = pd.DataFrame([row])

    # Preserve the exact training feature order
    input_data = input_data[feature_columns]

    predicted_revenue = float(
        model.predict(input_data)[0]
    )

    predicted_revenue = max(0.0, predicted_revenue)

    forecast_rows.append({
        "Forecast_Month": forecast_month,
        "Predicted_Revenue": round(predicted_revenue, 2),
        "Model_Name": "RandomForestRegressor"
    })

    # Use the forecast to calculate the next month's lag features
    revenues.append(predicted_revenue)

# --------------------------------------------------
# 5. Save forecast output
# --------------------------------------------------

forecast_df = pd.DataFrame(forecast_rows)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
forecast_df.to_csv(OUTPUT_PATH, index=False)

print("\n12-YEAR REVENUE FORECAST COMPLETED")
print("----------------------------------")
print("Last historical month:", latest_month.date())
print("First forecast month:", forecast_df["Forecast_Month"].iloc[0].date())
print("Last forecast month:", forecast_df["Forecast_Month"].iloc[-1].date())
print("Forecast months:", len(forecast_df))
print("Saved file:", OUTPUT_PATH)

print("\nFirst 12 forecast months:")
print(forecast_df.head(12).to_string(index=False))

print("\nYearly forecast summary:")
yearly = forecast_df.assign(
    Forecast_Year=forecast_df["Forecast_Month"].dt.year
).groupby("Forecast_Year").agg(
    Annual_Revenue=("Predicted_Revenue", "sum"),
    Average_Monthly_Revenue=("Predicted_Revenue", "mean")
).round(2)

print(yearly.to_string())
