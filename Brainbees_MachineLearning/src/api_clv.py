
from pathlib import Path
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent

# Load environment variables from either location
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=False)

app = FastAPI(
    title="Brainbees Customer CLV API",
    description="API for 12-month predicted customer revenue.",
    version="1.0.0",
)

required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_env = [key for key in required_env if not os.getenv(key)]

engine = None

if not missing_env:
    db_url = URL.create(
        "mysql+pymysql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME"),
    )
    engine = create_engine(db_url, pool_pre_ping=True)


def get_connection():
    if missing_env:
        raise HTTPException(
            status_code=500,
            detail=f"Missing .env settings: {missing_env}",
        )

    try:
        with engine.connect() as connection:
            yield connection
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Database query failed. Check MySQL and table setup.",
        ) from exc


@app.get("/")
def health_check():
    return {
        "status": "running",
        "api": "Brainbees Customer CLV API",
        "horizon_months": 12,
    }


@app.get("/predictions/customer-clv/12-months")
def get_clv_predictions(
    clv_segment: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=5000),
):
    if engine is None:
        raise HTTPException(
            status_code=500,
            detail=f"Missing .env settings: {missing_env}",
        )

    query = """
        SELECT
            Customer_ID,
            Snapshot_Date,
            Total_Orders_12M,
            Revenue_12M,
            Profit_12M,
            Units_12M,
            Active_Months_12M,
            Average_Order_Value_12M,
            Recency_Days,
            Predicted_Future_Revenue_12M,
            CLV_Segment,
            CLV_Model_Name,
            Prediction_Horizon_Months,
            Prediction_Date
        FROM ml_customer_clv
    """

    params = {"limit": limit}

    if clv_segment:
        query += " WHERE CLV_Segment = :clv_segment"
        params["clv_segment"] = clv_segment

    query += """
        ORDER BY Predicted_Future_Revenue_12M DESC
        LIMIT :limit
    """

    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(query), params
            ).mappings().all()

        return {
            "count": len(rows),
            "horizon_months": 12,
            "predictions": [dict(row) for row in rows],
        }

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve CLV predictions. "
                   "Check that ml_customer_clv exists.",
        ) from exc


@app.get("/predictions/customer-clv/high-value")
def get_high_value_customers(
    limit: int = Query(default=100, ge=1, le=5000),
):
    if engine is None:
        raise HTTPException(
            status_code=500,
            detail=f"Missing .env settings: {missing_env}",
        )

    query = text("""
        SELECT
            Customer_ID,
            Revenue_12M,
            Predicted_Future_Revenue_12M,
            CLV_Segment,
            Recency_Days,
            Prediction_Date
        FROM ml_customer_clv
        WHERE CLV_Segment IN ('High Value', 'Very High Value')
        ORDER BY Predicted_Future_Revenue_12M DESC
        LIMIT :limit
    """)

    try:
        with engine.connect() as connection:
            rows = connection.execute(
                query, {"limit": limit}
            ).mappings().all()

        return {
            "count": len(rows),
            "customers": [dict(row) for row in rows],
        }

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve high-value customers.",
        ) from exc


@app.get("/predictions/customer-clv/summary")
def get_clv_summary():
    if engine is None:
        raise HTTPException(
            status_code=500,
            detail=f"Missing .env settings: {missing_env}",
        )

    query = text("""
        SELECT
            COUNT(*) AS Total_Customers,
            ROUND(SUM(Predicted_Future_Revenue_12M), 2)
                AS Total_Predicted_Revenue,
            ROUND(AVG(Predicted_Future_Revenue_12M), 2)
                AS Average_Predicted_Revenue,
            SUM(CASE
                WHEN CLV_Segment = 'Low Value' THEN 1 ELSE 0
            END) AS Low_Value_Customers,
            SUM(CASE
                WHEN CLV_Segment = 'Medium Value' THEN 1 ELSE 0
            END) AS Medium_Value_Customers,
            SUM(CASE
                WHEN CLV_Segment = 'High Value' THEN 1 ELSE 0
            END) AS High_Value_Customers,
            SUM(CASE
                WHEN CLV_Segment = 'Very High Value' THEN 1 ELSE 0
            END) AS Very_High_Value_Customers
        FROM ml_customer_clv
    """)

    try:
        with engine.connect() as connection:
            row = connection.execute(query).mappings().one()

        return {"summary": dict(row)}

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve CLV summary.",
        ) from exc
