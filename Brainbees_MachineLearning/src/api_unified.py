from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from Brainbees_MachineLearning.src.api import app as forecasting_app
from Brainbees_MachineLearning.src.api_churn import app as churn_app
from Brainbees_MachineLearning.src.api_clv import app as clv_app

BASE_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="Brainbees Unified ML API",
    description="Unified API for Brainbees ML forecasting and predictions",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "status": "running",
        "application": "Brainbees Unified ML API",
        "models": [
            "Revenue forecasting",
            "Order forecasting",
            "Product demand forecasting",
            "Customer churn prediction",
            "Customer lifetime value prediction",
        ],
        "documentation": "/docs",
        "api_routes": {
            "revenue_and_orders": "/forecasting",
            "customer_churn": "/churn",
            "customer_clv": "/clv",
            "product_demand": "/product-demand/3-years",
        },
    }


@app.get("/product-demand/3-years")
def product_demand_forecast():
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
            detail=f"Unable to read product demand forecast: {exc}",
        )


# Keep each existing API's implementation intact.
app.mount("/forecasting", forecasting_app)
app.mount("/churn", churn_app)
app.mount("/clv", clv_app)