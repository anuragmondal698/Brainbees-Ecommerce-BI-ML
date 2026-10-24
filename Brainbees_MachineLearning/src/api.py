from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# 1. Define project and model paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "revenue_random_forest.joblib"
FEATURE_PATH = BASE_DIR / "models" / "revenue_feature_columns.joblib"

# --------------------------------------------------
# 2. Load trained model and feature columns
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(f"Feature file not found: {FEATURE_PATH}")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

print("Revenue model loaded successfully.")
print("Features:", feature_columns)

# --------------------------------------------------
# 3. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Brainbees Revenue Forecast API",
    description="API for monthly revenue forecasting",
    version="1.0.0"
)


class RevenuePredictionRequest(BaseModel):
    month_number: int = Field(ge=1, le=12)
    time_index: int = Field(ge=1)
    revenue_lag_1: float = Field(ge=0)
    revenue_lag_2: float = Field(ge=0)
    revenue_lag_3: float = Field(ge=0)
    revenue_rolling_mean_3: float = Field(ge=0)


class RevenuePredictionResponse(BaseModel):
    predicted_revenue: float


# --------------------------------------------------
# 4. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Brainbees Revenue Forecast API is running"
    }


# --------------------------------------------------
# 5. Revenue prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=RevenuePredictionResponse
)
def predict_revenue(request: RevenuePredictionRequest):
    try:
        input_data = pd.DataFrame([request.model_dump()])

        # Match the exact feature order used during training
        input_data = input_data[feature_columns]

        prediction = float(model.predict(input_data)[0])

        return {
            "predicted_revenue": max(0.0, prediction)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

@app.get("/forecast/12-years")
def get_12_year_forecast():
    base_dir = Path(__file__).resolve().parents[1]

    csv_path = (
        base_dir
        / "data"
        / "processed"
        / "revenue_forecast_12_years.csv"
    )

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="12-year forecast CSV not found."
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "Forecast_Month",
        "Predicted_Revenue",
        "Model_Name",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing columns: {missing}"
        )

    if len(df) != 144:
        raise HTTPException(
            status_code=500,
            detail=f"Expected 144 forecasts, found {len(df)}."
        )

    df["Forecast_Month"] = pd.to_datetime(
        df["Forecast_Month"]
    ).dt.strftime("%Y-%m-%d")

    df["Predicted_Revenue"] = pd.to_numeric(
        df["Predicted_Revenue"],
        errors="raise"
    )

    return {
        "count": len(df),
        "forecasts": df[required_columns].to_dict(
            orient="records"
        ),
    }
    

# ==========================================================
# ORDER FORECASTING API
# Add this section to your existing src/api.py
# ==========================================================

from pydantic import BaseModel, Field
import numpy as np


# ----------------------------------------------------------
# 1. LOAD TRAINED ORDER MODEL
# ----------------------------------------------------------

ORDER_MODEL_PATH = (
    BASE_DIR / "models" / "order_forecasting_model.joblib"
)

ORDER_FEATURE_PATH = (
    BASE_DIR / "models" / "order_feature_columns.joblib"
)

ORDER_FORECAST_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "order_forecast_12_months.csv"
)

order_artifact = joblib.load(ORDER_MODEL_PATH)

order_model_name = order_artifact["model_name"]
order_model = order_artifact["model"]
order_feature_columns = joblib.load(ORDER_FEATURE_PATH)


# ----------------------------------------------------------
# 2. REQUEST AND RESPONSE SCHEMAS
# ----------------------------------------------------------

class OrderPredictionRequest(BaseModel):
    Month_Number: int = Field(ge=1, le=12)
    Time_Index: int = Field(ge=1)
    Orders_Lag_1: float = Field(ge=0)
    Orders_Lag_2: float = Field(ge=0)
    Orders_Lag_3: float = Field(ge=0)
    Orders_Rolling_Mean_3: float = Field(ge=0)


class OrderPredictionResponse(BaseModel):
    predicted_orders: int
    model_name: str


# ----------------------------------------------------------
# 3. PREDICT ORDERS FOR ONE MONTH
# ----------------------------------------------------------

@app.post(
    "/predict-orders",
    response_model=OrderPredictionResponse
)
def predict_orders(request: OrderPredictionRequest):

    if order_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The selected order model is a baseline, "
                "not a trained prediction model. "
                "A baseline prediction endpoint is not configured."
            ),
        )

    input_data = pd.DataFrame(
        [request.model_dump()]
    )

    input_data = input_data[order_feature_columns]

    prediction = float(
        order_model.predict(input_data)[0]
    )

    predicted_orders = max(
        0,
        int(round(prediction))
    )

    return {
        "predicted_orders": predicted_orders,
        "model_name": order_model_name,
    }


# ----------------------------------------------------------
# 4. GET THE NEXT 12 MONTHS OF ORDER FORECASTS
# ----------------------------------------------------------

@app.get("/forecast/orders/12-months")
def get_order_forecast():

    if not ORDER_FORECAST_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Order forecast CSV not found. "
                "Run forecast_future_orders.py first."
            ),
        )

    try:
        forecast_df = pd.read_csv(ORDER_FORECAST_CSV)

        required_columns = [
            "Forecast_Month",
            "Predicted_Orders",
            "Model_Name",
        ]

        missing = [
            col for col in required_columns
            if col not in forecast_df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Missing forecast columns: {missing}",
            )

        if len(forecast_df) != 12:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Expected 12 forecast rows, "
                    f"found {len(forecast_df)}."
                ),
            )

        forecast_df = forecast_df[required_columns].copy()

        forecast_df["Forecast_Month"] = (
            pd.to_datetime(
                forecast_df["Forecast_Month"],
                errors="raise",
            ).dt.strftime("%Y-%m-%d")
        )

        forecast_df["Predicted_Orders"] = (
            pd.to_numeric(
                forecast_df["Predicted_Orders"],
                errors="raise",
            ).astype(int)
        )

        if (forecast_df["Predicted_Orders"] < 0).any():
            raise HTTPException(
                status_code=500,
                detail="Negative forecast values found.",
            )

        return {
            "count": len(forecast_df),
            "model_name": order_model_name,
            "forecasts": forecast_df.to_dict(
                orient="records"
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read order forecast: {exc}",
        )


# Add these imports at the top of api.py if not already present
@app.get("/forecast/product-demand/3-years")
def get_product_demand_forecast():
    csv_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "product_demand_forecast_3_years.csv"
    )

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Product demand forecast CSV not found.",
        )

    try:
        df = pd.read_csv(csv_path)

        return {
            "model": "product_demand_forecasting",
            "forecast_horizon_months": 36,
            "total_rows": len(df),
            "forecasts": df.to_dict(orient="records"),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

@app.get("/forecast/product-demand/3-years")
def get_product_demand_forecast():
    csv_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "product_demand_forecast_3_years.csv"
    )

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Forecast CSV not found: {csv_path}",
        )

    df = pd.read_csv(csv_path)

    return {
        "model": "product_demand_forecasting",
        "forecast_horizon_months": 36,
        "total_rows": len(df),
        "forecasts": df.to_dict(orient="records"),
    }
    
        