
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

CSV_PATH = (
    BASE_DIR / "data" / "processed" / "order_forecast_12_months.csv"
)

required_env = [
    "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"
]

missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise ValueError(f"Missing .env settings: {missing_env}")

engine_url = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    database=os.getenv("DB_NAME"),
)

engine = create_engine(engine_url)

df = pd.read_csv(CSV_PATH)

required_columns = [
    "Forecast_Month",
    "Predicted_Orders",
    "Model_Name",
]

if not all(column in df.columns for column in required_columns):
    raise ValueError("Forecast CSV is missing required columns.")

df = df[required_columns].copy()
df["Forecast_Month"] = pd.to_datetime(
    df["Forecast_Month"], errors="raise"
).dt.date

df["Predicted_Orders"] = pd.to_numeric(
    df["Predicted_Orders"], errors="raise"
)

if len(df) != 12:
    raise ValueError(f"Expected 12 rows; found {len(df)}.")

if df["Forecast_Month"].duplicated().any():
    raise ValueError("Duplicate forecast months found.")

if (df["Predicted_Orders"] < 0).any():
    raise ValueError("Negative predicted orders found.")

sql = text("""
    INSERT INTO ml_order_forecast
        (Forecast_Month, Predicted_Orders, Model_Name)
    VALUES
        (:forecast_month, :predicted_orders, :model_name)
    ON DUPLICATE KEY UPDATE
        Predicted_Orders = VALUES(Predicted_Orders),
        Model_Name = VALUES(Model_Name),
        Created_At = CURRENT_TIMESTAMP
""")

records = [
    {
        "forecast_month": row.Forecast_Month,
        "predicted_orders": int(row.Predicted_Orders),
        "model_name": row.Model_Name,
    }
    for row in df.itertuples(index=False)
]

with engine.begin() as connection:
    connection.execute(sql, records)

print(f"Successfully saved/updated {len(records)} forecast rows.")

with engine.connect() as connection:
    result = connection.execute(text("""
        SELECT Forecast_Month, Predicted_Orders, Model_Name
        FROM ml_order_forecast
        ORDER BY Forecast_Month
    """))

    print(pd.DataFrame(result.fetchall(), columns=result.keys()).to_string(index=False))

engine.dispose()
