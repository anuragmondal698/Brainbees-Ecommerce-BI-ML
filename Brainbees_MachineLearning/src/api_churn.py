
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import URL, create_engine, text


# ============================================================
# 1. APP AND DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="Brainbees Customer Churn API",
    description="Customer churn predictions for the next 24 months",
    version="1.0.0",
)

required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise RuntimeError(
        f"Missing environment variables in .env: {missing_env}"
    )

engine = create_engine(
    URL.create(
        "mysql+pymysql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME"),
        query={"charset": "utf8mb4"},
    ),
    pool_pre_ping=True,
)


# ============================================================
# 2. HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Brainbees Customer Churn API is running",
        "forecast_horizon_months": 24,
    }


# ============================================================
# 3. CUSTOMER CHURN PREDICTIONS
# ============================================================

@app.get("/predictions/customer-churn/2-years")
def get_customer_churn_predictions(
    risk_level: str | None = Query(
        default=None,
        description="Optional filter: Low, Medium, or High",
    ),
    limit: int = Query(default=1000, ge=1, le=5000),
):
    query = """
        SELECT
            Customer_ID,
            Snapshot_Date,
            Total_Orders,
            Recency_Days,
            Customer_Tenure_Days,
            Active_Months,
            Orders_Per_Active_Month,
            Churn_Probability,
            Predicted_Churn_24_Months,
            Risk_Level,
            Model_Name,
            Forecast_Horizon_Months,
            Prediction_Date
        FROM ml_customer_churn_2_years
    """

    params = {"limit": limit}

    if risk_level:
        allowed_levels = {"LOW", "MEDIUM", "HIGH"}

        if risk_level.strip().upper() not in allowed_levels:
            raise HTTPException(
                status_code=400,
                detail="risk_level must be Low, Medium, or High.",
            )

        query += " WHERE UPPER(Risk_Level) = :risk_level"
        params["risk_level"] = risk_level.strip().upper()

    query += """
        ORDER BY Churn_Probability DESC
        LIMIT :limit
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params=params,
            )

        for col in ["Snapshot_Date", "Prediction_Date"]:
            df[col] = pd.to_datetime(
                df[col], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        predictions = df.astype(object).where(
            pd.notna(df), None
        ).to_dict(orient="records")

        return {
            "status": "success",
            "forecast_horizon_months": 24,
            "total_records": len(predictions),
            "predictions": predictions,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve churn predictions: {str(exc)}",
        )


# ============================================================
# 4. HIGH-RISK CUSTOMERS
# ============================================================

@app.get("/predictions/customer-churn/high-risk")
def get_high_risk_customers(
    limit: int = Query(default=1000, ge=1, le=5000),
):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("""
                    SELECT
                        Customer_ID,
                        Churn_Probability,
                        Risk_Level,
                        Recency_Days,
                        Total_Orders,
                        Forecast_Horizon_Months,
                        Prediction_Date
                    FROM ml_customer_churn_2_years
                    WHERE UPPER(Risk_Level) = 'HIGH'
                    ORDER BY Churn_Probability DESC
                    LIMIT :limit
                """),
                conn,
                params={"limit": limit},
            )

        for col in ["Prediction_Date"]:
            df[col] = pd.to_datetime(
                df[col], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        customers = df.astype(object).where(
            pd.notna(df), None
        ).to_dict(orient="records")

        return {
            "status": "success",
            "forecast_horizon_months": 24,
            "total_records": len(customers),
            "customers": customers,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve high-risk customers: {str(exc)}",
        )



