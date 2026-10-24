
from pathlib import Path
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# --------------------------------------------------
# 1. Project paths and database configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env")

required_env = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise ValueError(
        f"Missing database settings in .env: {missing_env}"
    )

engine_url = URL.create(
    "mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME", "brainbees_ecommerce"),
)

engine = create_engine(engine_url)

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load monthly order counts from MySQL
# --------------------------------------------------

monthly_sql = text("""
    SELECT
        DATE_FORMAT(Order_Date, '%Y-%m-01') AS Order_Month,
        COUNT(DISTINCT Order_ID) AS Total_Orders
    FROM fact_orders
    WHERE Order_Date IS NOT NULL
      AND Order_ID IS NOT NULL
    GROUP BY DATE_FORMAT(Order_Date, '%Y-%m-01')
    ORDER BY Order_Month
""")

with engine.connect() as connection:
    monthly = pd.read_sql(monthly_sql, connection)

    quality_sql = text("""
        SELECT
            COUNT(*) AS Total_Rows,
            COUNT(DISTINCT Order_ID) AS Unique_Orders,
            SUM(Order_ID IS NULL) AS Missing_Order_ID,
            SUM(Customer_ID IS NULL) AS Missing_Customer_ID,
            SUM(Order_Date IS NULL) AS Missing_Order_Date
        FROM fact_orders
    """)

    quality = pd.read_sql(quality_sql, connection)

    duplicate_sql = text("""
        SELECT
            Order_ID,
            COUNT(*) AS Occurrences
        FROM fact_orders
        WHERE Order_ID IS NOT NULL
        GROUP BY Order_ID
        HAVING COUNT(*) > 1
    """)

    duplicate_orders = pd.read_sql(duplicate_sql, connection)


# --------------------------------------------------
# 3. Validate and prepare calendar months
# --------------------------------------------------

if monthly.empty:
    raise ValueError(
        "No valid monthly order data found in fact_orders."
    )

monthly["Order_Month"] = pd.to_datetime(
    monthly["Order_Month"], errors="coerce"
)
monthly["Total_Orders"] = pd.to_numeric(
    monthly["Total_Orders"], errors="coerce"
)

monthly = monthly.dropna(
    subset=["Order_Month", "Total_Orders"]
).sort_values("Order_Month")

if monthly["Order_Month"].duplicated().any():
    raise ValueError(
        "Duplicate monthly records found. Investigate the SQL output."
    )

if (monthly["Total_Orders"] < 0).any():
    raise ValueError("Negative monthly order counts detected.")

# Build the full monthly calendar.
# Missing months remain NaN until their meaning is verified.
full_calendar = pd.date_range(
    start=monthly["Order_Month"].min(),
    end=monthly["Order_Month"].max(),
    freq="MS",
)

monthly = (
    monthly.set_index("Order_Month")
    .reindex(full_calendar)
    .rename_axis("Order_Month")
    .reset_index()
)

monthly["Is_Missing_Month"] = (
    monthly["Total_Orders"].isna()
)

missing_months = monthly.loc[
    monthly["Is_Missing_Month"],
    ["Order_Month"]
].copy()

# Do not replace missing monthly counts with zero automatically.
# Verify that missing months genuinely had no orders first.


# --------------------------------------------------
# 4. Create time-series features
# --------------------------------------------------

monthly["Month_Number"] = monthly["Order_Month"].dt.month

monthly["Time_Index"] = np.arange(
    1, len(monthly) + 1
)

monthly["Orders_Lag_1"] = (
    monthly["Total_Orders"].shift(1)
)

monthly["Orders_Lag_2"] = (
    monthly["Total_Orders"].shift(2)
)

monthly["Orders_Lag_3"] = (
    monthly["Total_Orders"].shift(3)
)

# Shift first so the current month's orders are excluded.
monthly["Orders_Rolling_Mean_3"] = (
    monthly["Total_Orders"]
    .shift(1)
    .rolling(window=3, min_periods=3)
    .mean()
)


# --------------------------------------------------
# 5. Create data-quality report
# --------------------------------------------------

summary = quality.iloc[0].to_dict()

report_rows = [
    {
        "Check": "Total source rows",
        "Result": int(summary["Total_Rows"]),
        "Status": "INFO",
    },
    {
        "Check": "Distinct non-null order IDs",
        "Result": int(summary["Unique_Orders"]),
        "Status": "INFO",
    },
    {
        "Check": "Missing Order_ID",
        "Result": int(summary["Missing_Order_ID"] or 0),
        "Status": (
            "PASS"
            if not summary["Missing_Order_ID"]
            else "REVIEW"
        ),
    },
    {
        "Check": "Missing Customer_ID",
        "Result": int(summary["Missing_Customer_ID"] or 0),
        "Status": (
            "PASS"
            if not summary["Missing_Customer_ID"]
            else "REVIEW"
        ),
    },
    {
        "Check": "Missing Order_Date",
        "Result": int(summary["Missing_Order_Date"] or 0),
        "Status": (
            "PASS"
            if not summary["Missing_Order_Date"]
            else "REVIEW"
        ),
    },
    {
        "Check": "Repeated non-null Order_IDs",
        "Result": len(duplicate_orders),
        "Status": (
            "PASS" if duplicate_orders.empty else "REVIEW"
        ),
    },
    {
        "Check": "Missing calendar months",
        "Result": len(missing_months),
        "Status": (
            "PASS" if missing_months.empty else "REVIEW"
        ),
    },
]

report = pd.DataFrame(report_rows)

# --------------------------------------------------
# 6. Save outputs
# --------------------------------------------------

monthly_path = PROCESSED_DIR / "monthly_orders.csv"
report_path = PROCESSED_DIR / "order_data_quality_report.csv"
missing_path = PROCESSED_DIR / "missing_order_months.csv"
duplicates_path = PROCESSED_DIR / "duplicate_order_ids.csv"

monthly.to_csv(monthly_path, index=False)
report.to_csv(report_path, index=False)
missing_months.to_csv(missing_path, index=False)
duplicate_orders.to_csv(duplicates_path, index=False)

print("\nORDER DATA PREPARATION COMPLETED")
print("--------------------------------")
print(f"Source rows: {summary['Total_Rows']}")
print(f"Distinct order IDs: {summary['Unique_Orders']}")
print(f"Monthly calendar rows: {len(monthly)}")
print(f"Missing calendar months: {len(missing_months)}")
print(f"Repeated Order_IDs: {len(duplicate_orders)}")
print(f"\nMonthly data: {monthly_path}")
print(f"Quality report: {report_path}")
print(f"Missing months: {missing_path}")
print(f"Duplicate IDs: {duplicates_path}")

print("\nData-quality report:")
print(report.to_string(index=False))

engine.dispose()
