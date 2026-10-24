
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "monthly_product_demand.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "product_demand_model.joblib"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_demand_forecast_3_years.csv"
)

FEATURES = [
    "Month_Number",
    "Time_Index",
    "Demand_Lag_1",
    "Demand_Lag_2",
    "Demand_Lag_3",
    "Demand_Rolling_Mean_3",
]


# =====================================================
# 2. CHECK FILES
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found: {MODEL_PATH}"
    )


# =====================================================
# 3. LOAD DATA AND MODEL ARTIFACT
# =====================================================

df = pd.read_csv(DATA_PATH)

artifact = joblib.load(MODEL_PATH)

model_name = artifact["model_name"]
model = artifact["model"]
feature_columns = artifact["feature_columns"]

if feature_columns != FEATURES:
    raise ValueError(
        "Feature mismatch.\n"
        f"Model features: {feature_columns}\n"
        f"Expected features: {FEATURES}"
    )

required_columns = [
    "Product_ID",
    "Product_Name",
    "Category_ID",
    "Demand_Month",
    "Units_Sold",
    "Time_Index",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )


# =====================================================
# 4. CLEAN AND VALIDATE HISTORICAL DATA
# =====================================================

df["Demand_Month"] = pd.to_datetime(
    df["Demand_Month"],
    errors="raise",
)

df["Units_Sold"] = pd.to_numeric(
    df["Units_Sold"],
    errors="raise",
)

df["Time_Index"] = pd.to_numeric(
    df["Time_Index"],
    errors="raise",
)

df = df.sort_values(
    ["Product_ID", "Demand_Month"]
).reset_index(drop=True)

if df.empty:
    raise ValueError("The demand dataset is empty.")

if df["Product_ID"].isna().any():
    raise ValueError("Missing Product_ID values found.")

if df["Units_Sold"].isna().any():
    raise ValueError("Missing historical demand values found.")

if (df["Units_Sold"] < 0).any():
    raise ValueError("Negative historical demand found.")

if df.duplicated(
    ["Product_ID", "Demand_Month"]
).any():
    raise ValueError(
        "Duplicate Product_ID and Demand_Month records found."
    )

# Ensure every product has a consecutive monthly history.
for product_id, group in df.groupby("Product_ID"):
    months = pd.DatetimeIndex(
        group["Demand_Month"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    expected_months = pd.date_range(
        start=months.min(),
        end=months.max(),
        freq="MS",
    )

    if not months.equals(expected_months):
        raise ValueError(
            f"Non-consecutive monthly history for product "
            f"{product_id}. Rerun the data preparation step."
        )


# =====================================================
# 5. VALIDATE SELECTED MODEL
# =====================================================

supported_models = {
    "Naive_Last_Month",
    "RandomForestRegressor",
}

if model_name not in supported_models:
    raise ValueError(
        f"Unsupported selected model: {model_name}"
    )

if (
    model_name == "RandomForestRegressor"
    and model is None
):
    raise ValueError(
        "Random Forest was selected, but the model object is missing."
    )

if (
    model_name == "Naive_Last_Month"
    and model is not None
):
    print(
        "Notice: Baseline selected; the saved model object "
        "will not be used."
    )


# =====================================================
# 6. DEFINE NEXT 36 FORECAST MONTHS
# =====================================================

last_historical_month = df["Demand_Month"].max()

forecast_months = pd.date_range(
    start=last_historical_month + pd.offsets.MonthBegin(1),
    periods=36,
    freq="MS",
)

forecast_rows = []

skipped_products = []


# =====================================================
# 7. FORECAST EACH PRODUCT FOR 36 MONTHS
# =====================================================

for product_id, group in df.groupby(
    "Product_ID",
    sort=False,
):

    group = group.sort_values("Demand_Month")

    history = (
        group["Units_Sold"]
        .astype(float)
        .tolist()
    )

    if len(history) < 3:
        skipped_products.append(product_id)
        continue

    product_name = group["Product_Name"].iloc[-1]
    category_id = group["Category_ID"].iloc[-1]

    last_time_index = int(
        group["Time_Index"].iloc[-1]
    )

    for step, forecast_month in enumerate(
        forecast_months,
        start=1,
    ):

        # Construct features using history available
        # before the month being forecast.
        input_row = {
            "Month_Number": int(forecast_month.month),
            "Time_Index": last_time_index + step,
            "Demand_Lag_1": history[-1],
            "Demand_Lag_2": history[-2],
            "Demand_Lag_3": history[-3],
            "Demand_Rolling_Mean_3": float(
                np.mean(history[-3:])
            ),
        }

        input_df = pd.DataFrame(
            [input_row]
        )[feature_columns]

        # ---------------------------------------------
        # BASELINE OR RANDOM FOREST PREDICTION
        # ---------------------------------------------

        if model_name == "Naive_Last_Month":
            # The forecast equals the previous month's demand.
            predicted_units = history[-1]

        elif model_name == "RandomForestRegressor":
            predicted_units = float(
                model.predict(input_df)[0]
            )

        else:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        # Units sold cannot be negative.
        predicted_units = max(
            0.0,
            predicted_units,
        )

        # Save one product-month prediction.
        forecast_rows.append({
            "Product_ID": product_id,
            "Product_Name": product_name,
            "Category_ID": category_id,
            "Forecast_Month": forecast_month.strftime(
                "%Y-%m-%d"
            ),
            "Predicted_Units": round(
                predicted_units,
                2,
            ),
            "Model_Name": model_name,
        })

        # Feed the forecast into future lag features.
        history.append(predicted_units)


# =====================================================
# 8. VALIDATE FORECAST OUTPUT
# =====================================================

forecast_df = pd.DataFrame(forecast_rows)

if forecast_df.empty:
    raise ValueError(
        "No forecasts generated. Check product history."
    )

forecast_df["Forecast_Month"] = pd.to_datetime(
    forecast_df["Forecast_Month"],
    errors="raise",
)

if forecast_df.duplicated(
    ["Product_ID", "Forecast_Month"]
).any():
    raise ValueError(
        "Duplicate product-month forecasts generated."
    )

actual_forecast_months = forecast_df[
    "Forecast_Month"
].nunique()

if actual_forecast_months != 36:
    raise ValueError(
        f"Expected 36 forecast months, "
        f"generated {actual_forecast_months}."
    )

expected_rows = (
    forecast_df["Product_ID"].nunique() * 36
)

if len(forecast_df) != expected_rows:
    raise ValueError(
        "Forecast row count does not match eligible "
        "products multiplied by 36 months."
    )


# =====================================================
# 9. SAVE FORECAST CSV
# =====================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# =====================================================
# 10. PRINT SUMMARY
# =====================================================

print("\n========== PRODUCT DEMAND FORECAST ==========")

print("Selected model:", model_name)

print(
    "Historical data ends:",
    last_historical_month.strftime("%Y-%m"),
)

print(
    "Products forecast:",
    forecast_df["Product_ID"].nunique(),
)

print(
    "Forecast months:",
    actual_forecast_months,
)

print(
    "Total forecast rows:",
    len(forecast_df),
)

print(
    "First forecast month:",
    forecast_df["Forecast_Month"].min().strftime("%Y-%m"),
)

print(
    "Last forecast month:",
    forecast_df["Forecast_Month"].max().strftime("%Y-%m"),
)

print(
    "Total predicted units across all products/months:",
    round(forecast_df["Predicted_Units"].sum(), 2),
)

print(
    "Products skipped due to insufficient history:",
    len(skipped_products),
)

if skipped_products:
    print("Skipped Product IDs:", skipped_products)

print("\nForecast saved to:")
print(OUTPUT_PATH)

print("\nProduct demand forecasting completed successfully.")

