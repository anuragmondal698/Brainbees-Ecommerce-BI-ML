from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env")

url = URL.create(
    "mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "127.0.0.1"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME", "brainbees_ecommerce"),
)

csv_path = BASE_DIR / "data" / "processed" / "revenue_forecast_12_years.csv"
df = pd.read_csv(csv_path)

df["Forecast_Month"] = pd.to_datetime(
    df["Forecast_Month"]
).dt.date

required = {
    "Forecast_Month",
    "Predicted_Revenue",
    "Model_Name",
}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

if len(df) != 144:
    raise ValueError(f"Expected 144 forecast rows, found {len(df)}")

if df["Forecast_Month"].duplicated().any():
    raise ValueError("Duplicate forecast months found.")

engine = create_engine(url)

try:
    with engine.begin() as conn:
        # Replace the forecast horizon as a single transaction.
        conn.exec_driver_sql(
            "DELETE FROM ml_revenue_forecast_12_years"
        )

        df.to_sql(
            "ml_revenue_forecast_12_years",
            con=conn,
            if_exists="append",
            index=False,
            chunksize=100,
        )

    print("Successfully saved 144 monthly forecasts to MySQL.")

finally:
    engine.dispose()