
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=False)

DATA_PATH = (
    BASE_DIR / "data" / "processed" /
    "customer_clv_predictions.csv"
)

required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise EnvironmentError(
        f"Missing database settings in .env: {missing_env}"
    )

db_url = URL.create(
    "mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME"),
)

engine = create_engine(db_url, pool_pre_ping=True)

def main():
    df = pd.read_csv(DATA_PATH)

    if "Customer_ID" not in df.columns:
        raise ValueError("Customer_ID column is missing.")

    if df["Customer_ID"].isna().any():
        raise ValueError("Customer_ID contains missing values.")

    if df["Customer_ID"].duplicated().any():
        raise ValueError(
            "Duplicate Customer_ID values found. "
            "Resolve them before loading."
        )

    columns = [
        "Customer_ID",
        "Snapshot_Date",
        "Total_Orders_12M",
        "Revenue_12M",
        "Profit_12M",
        "Units_12M",
        "Active_Months_12M",
        "Average_Order_Value_12M",
        "Recency_Days",
        "Predicted_Future_Revenue_12M",
        "CLV_Segment",
        "CLV_Model_Name",
        "Prediction_Horizon_Months",
        "Prediction_Date",
    ]

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df = df[columns].copy()

    for col in ["Snapshot_Date", "Prediction_Date"]:
        df[col] = pd.to_datetime(
            df[col], errors="coerce"
        ).dt.date

    df["Customer_ID"] = df["Customer_ID"].astype(str)
    df = df.where(pd.notna(df), None)

    insert_sql = text("""
        INSERT INTO ml_customer_clv (
            Customer_ID, Snapshot_Date, Total_Orders_12M,
            Revenue_12M, Profit_12M, Units_12M,
            Active_Months_12M, Average_Order_Value_12M,
            Recency_Days, Predicted_Future_Revenue_12M,
            CLV_Segment, CLV_Model_Name,
            Prediction_Horizon_Months, Prediction_Date
        )
        VALUES (
            :Customer_ID, :Snapshot_Date, :Total_Orders_12M,
            :Revenue_12M, :Profit_12M, :Units_12M,
            :Active_Months_12M, :Average_Order_Value_12M,
            :Recency_Days, :Predicted_Future_Revenue_12M,
            :CLV_Segment, :CLV_Model_Name,
            :Prediction_Horizon_Months, :Prediction_Date
        )
        ON DUPLICATE KEY UPDATE
            Snapshot_Date = VALUES(Snapshot_Date),
            Total_Orders_12M = VALUES(Total_Orders_12M),
            Revenue_12M = VALUES(Revenue_12M),
            Profit_12M = VALUES(Profit_12M),
            Units_12M = VALUES(Units_12M),
            Active_Months_12M = VALUES(Active_Months_12M),
            Average_Order_Value_12M = VALUES(Average_Order_Value_12M),
            Recency_Days = VALUES(Recency_Days),
            Predicted_Future_Revenue_12M =
                VALUES(Predicted_Future_Revenue_12M),
            CLV_Segment = VALUES(CLV_Segment),
            CLV_Model_Name = VALUES(CLV_Model_Name),
            Prediction_Horizon_Months =
                VALUES(Prediction_Horizon_Months),
            Prediction_Date = VALUES(Prediction_Date)
    """)

    records = df.to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(insert_sql, records)

        count = connection.execute(
            text("SELECT COUNT(*) FROM ml_customer_clv")
        ).scalar_one()

    print("CLV predictions loaded successfully.")
    print(f"Rows in ml_customer_clv: {count}")
    print(f"Rows processed in this run: {len(df)}")

if __name__ == "__main__":
    main()
