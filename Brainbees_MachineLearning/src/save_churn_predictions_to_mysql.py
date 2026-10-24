
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

CSV_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_churn_predictions_2_years.csv"
)

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Prediction CSV not found: {CSV_FILE}\n"
        "Run predict_customer_churn_2_years.py first."
    )

required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise EnvironmentError(
        f"Missing environment variables: {missing_env}"
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
# 2. LOAD AND VALIDATE CSV
# ============================================================

df = pd.read_csv(CSV_FILE)

required_columns = [
    "Customer_ID",
    "Snapshot_Date",
    "Total_Orders",
    "Recency_Days",
    "Customer_Tenure_Days",
    "Active_Months",
    "Orders_Per_Active_Month",
    "Churn_Probability",
    "Predicted_Churn_24_Months",
    "Risk_Level",
    "Model_Name",
    "Forecast_Horizon_Months",
    "Prediction_Date",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required CSV columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("Prediction CSV contains no records.")

df["Customer_ID"] = df["Customer_ID"].astype("string").str.strip()

if df["Customer_ID"].isna().any() or df["Customer_ID"].eq("").any():
    raise ValueError("Customer_ID contains missing or empty values.")

if df["Customer_ID"].duplicated().any():
    raise ValueError("Duplicate Customer_ID values found.")

numeric_columns = [
    "Total_Orders",
    "Recency_Days",
    "Customer_Tenure_Days",
    "Active_Months",
    "Orders_Per_Active_Month",
    "Churn_Probability",
    "Predicted_Churn_24_Months",
    "Forecast_Horizon_Months",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

date_columns = ["Snapshot_Date", "Prediction_Date"]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")

if df[["Churn_Probability", "Predicted_Churn_24_Months"]].isna().any().any():
    raise ValueError("Missing churn probability or prediction values.")

if not df["Churn_Probability"].between(0, 1).all():
    raise ValueError("Churn probabilities must be between 0 and 1.")

if not df["Predicted_Churn_24_Months"].isin([0, 1]).all():
    raise ValueError("Predicted churn labels must be 0 or 1.")

# Check optional text values.
df["Risk_Level"] = df["Risk_Level"].fillna("Unknown").astype(str)
df["Model_Name"] = df["Model_Name"].fillna("Unknown").astype(str)


# ============================================================
# 3. PREPARE DATABASE RECORDS
# ============================================================

records = []

for row in df.itertuples(index=False):
    records.append({
        "Customer_ID": str(row.Customer_ID),
        "Snapshot_Date": (
            None if pd.isna(row.Snapshot_Date)
            else row.Snapshot_Date.date()
        ),
        "Total_Orders": (
            None if pd.isna(row.Total_Orders)
            else int(row.Total_Orders)
        ),
        "Recency_Days": (
            None if pd.isna(row.Recency_Days)
            else int(row.Recency_Days)
        ),
        "Customer_Tenure_Days": (
            None if pd.isna(row.Customer_Tenure_Days)
            else int(row.Customer_Tenure_Days)
        ),
        "Active_Months": (
            None if pd.isna(row.Active_Months)
            else int(row.Active_Months)
        ),
        "Orders_Per_Active_Month": (
            None if pd.isna(row.Orders_Per_Active_Month)
            else float(row.Orders_Per_Active_Month)
        ),
        "Churn_Probability": float(row.Churn_Probability),
        "Predicted_Churn_24_Months": int(
            row.Predicted_Churn_24_Months
        ),
        "Risk_Level": row.Risk_Level,
        "Model_Name": row.Model_Name,
        "Forecast_Horizon_Months": int(
            row.Forecast_Horizon_Months
        ),
        "Prediction_Date": (
            None if pd.isna(row.Prediction_Date)
            else row.Prediction_Date.date()
        ),
    })


# ============================================================
# 4. INSERT OR UPDATE MYSQL TABLE
# ============================================================

upsert_sql = text("""
    INSERT INTO ml_customer_churn_2_years (
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
    )
    VALUES (
        :Customer_ID,
        :Snapshot_Date,
        :Total_Orders,
        :Recency_Days,
        :Customer_Tenure_Days,
        :Active_Months,
        :Orders_Per_Active_Month,
        :Churn_Probability,
        :Predicted_Churn_24_Months,
        :Risk_Level,
        :Model_Name,
        :Forecast_Horizon_Months,
        :Prediction_Date
    )
    ON DUPLICATE KEY UPDATE
        Snapshot_Date = VALUES(Snapshot_Date),
        Total_Orders = VALUES(Total_Orders),
        Recency_Days = VALUES(Recency_Days),
        Customer_Tenure_Days = VALUES(Customer_Tenure_Days),
        Active_Months = VALUES(Active_Months),
        Orders_Per_Active_Month = VALUES(Orders_Per_Active_Month),
        Churn_Probability = VALUES(Churn_Probability),
        Predicted_Churn_24_Months = VALUES(Predicted_Churn_24_Months),
        Risk_Level = VALUES(Risk_Level),
        Model_Name = VALUES(Model_Name),
        Forecast_Horizon_Months = VALUES(Forecast_Horizon_Months),
        Prediction_Date = VALUES(Prediction_Date)
""")

try:
    with engine.begin() as conn:
        # Confirm the table exists before inserting.
        conn.execute(text("""
            SELECT 1
            FROM ml_customer_churn_2_years
            LIMIT 1
        """))

        conn.execute(upsert_sql, records)

    print("Customer churn predictions loaded successfully.")
    print(f"Rows processed: {len(records):,}")

    with engine.connect() as conn:
        summary = pd.read_sql(
            text("""
                SELECT
                    COUNT(*) AS Total_Customers,
                    SUM(Predicted_Churn_24_Months = 1)
                        AS Predicted_Churned,
                    SUM(Predicted_Churn_24_Months = 0)
                        AS Predicted_Not_Churned,
                    ROUND(AVG(Churn_Probability), 4)
                        AS Average_Churn_Probability
                FROM ml_customer_churn_2_years
            """),
            conn,
        )

    print("\nMySQL verification:")
    print(summary.to_string(index=False))

finally:
    engine.dispose()
