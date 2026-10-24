
from fastapi import FastAPI
import requests

app = FastAPI(
    title="Brainbees Unified ML API",
    description="Combined API for all five ML models",
    version="1.0.0",
)

API_ENDPOINTS = {
    "revenue_forecasting": (
        "http://127.0.0.1:8000/forecast/12-years"
    ),
    "order_forecasting": (
        "http://127.0.0.1:8000/forecast/orders/12-months"
    ),
    "product_demand_forecasting": (
        "http://127.0.0.1:8000/forecast/product-demand/3-years"
    ),
    "customer_churn": (
        "http://127.0.0.1:8001/predictions/customer-churn/2-years"
    ),
    "customer_lifetime_value": (
        "http://127.0.0.1:8002/predictions/customer-clv/12-months"
    ),
}


@app.get("/")
def home():
    return {
        "message": "Brainbees Unified ML API is running",
        "models": list(API_ENDPOINTS.keys()),
        "combined_endpoint": "/all-forecasts",
        "documentation": "/docs",
    }


@app.get("/all-forecasts")
def get_all_forecasts():
    results = {}

    for model_name, url in API_ENDPOINTS.items():
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            results[model_name] = response.json()

        except requests.RequestException as error:
            results[model_name] = {
                "status": "error",
                "message": str(error),
            }

    successful = sum(
        1 for result in results.values()
        if result.get("status") != "error"
    )

    return {
        "total_models": len(API_ENDPOINTS),
        "successful_models": successful,
        "failed_models": len(API_ENDPOINTS) - successful,
        "results": results,
    }
