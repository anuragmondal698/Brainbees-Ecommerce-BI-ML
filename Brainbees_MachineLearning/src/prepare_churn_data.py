
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text


# ============================================================
# 1. PATHS AND DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_FILE = OUTPUT_DIR / "customer_churn_training_data.csv"
SCORING_FILE = OUTPUT_DIR / "customer_churn_scoring_data.csv"
REPORT_FILE = OUTPUT_DIR / "customer_churn_quality_report.csv"

required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise EnvironmentError(
        f"Missing .env variables: {missing_env}"
    )

connection_url = URL.create(
    "mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME"),
    query={"charset": "utf8mb4"},
)

engine = create_engine(
    connection_url,
    pool_pre_ping=True,
)


# ============================================================
# 2. LOAD HISTORICAL ORDERS
# ============================================================

query = text("""
    SELECT
        Customer_ID,
        Order_ID,
        Order_Date
    FROM fact_orders
    WHERE Customer_ID IS NOT NULL
      AND Order_ID IS NOT NULL
      AND Order_Date IS NOT NULL
""")

with engine.connect() as conn:
    orders = pd.read_sql(query, conn)

engine.dispose()

if orders.empty:
    raise ValueError("No usable order records were found.")

orders["Customer_ID"] = orders["Customer_ID"].astype("string").str.strip()
orders["Order_ID"] = orders["Order_ID"].astype("string").str.strip()
orders["Order_Date"] = pd.to_datetime(
    orders["Order_Date"], errors="coerce"
)

orders = orders.dropna(
    subset=["Customer_ID", "Order_ID", "Order_Date"]
)

orders = orders.drop_duplicates(
    subset=["Customer_ID", "Order_ID"]
)

orders = orders.sort_values("Order_Date")

data_start = orders["Order_Date"].min().normalize()
data_end = orders["Order_Date"].max().normalize()

print("=" * 65)
print("BRAINBEES CUSTOMER CHURN DATA PREPARATION")
print("=" * 65)
print(f"First order date: {data_start.date()}")
print(f"Last order date : {data_end.date()}")
print(f"Usable orders   : {len(orders):,}")
print(
    "Unique customers:",
    orders["Customer_ID"].nunique()
)


# ============================================================
# 3. DEFINE THE HISTORICAL 24-MONTH LABEL
# ============================================================

HORIZON_MONTHS = 24
FEATURE_WINDOW_MONTHS = 12

# Only use a historical snapshot for which the full
# 24-month outcome window is observable.
snapshot_date = (
    data_end - pd.DateOffset(months=HORIZON_MONTHS)
).normalize()

feature_start = (
    snapshot_date - pd.DateOffset(months=FEATURE_WINDOW_MONTHS)
).normalize()

outcome_end = (
    snapshot_date + pd.DateOffset(months=HORIZON_MONTHS)
).normalize()

print(f"\nHistorical snapshot: {snapshot_date.date()}")
print(f"Feature window starts: {feature_start.date()}")
print(f"Outcome window ends: {outcome_end.date()}")

if feature_start < data_start:
    raise ValueError(
        "Not enough historical data for a complete 12-month "
        "feature window followed by a 24-month outcome window."
    )

# Feature period: orders before the snapshot, within the
# previous 12 months.
feature_orders = orders[
    (orders["Order_Date"] >= feature_start)
    & (orders["Order_Date"] <= snapshot_date)
].copy()

# Outcome period: orders strictly after the snapshot and
# no later than the 24-month horizon.
outcome_orders = orders[
    (orders["Order_Date"] > snapshot_date)
    & (orders["Order_Date"] <= outcome_end)
].copy()

if feature_orders.empty:
    raise ValueError(
        "No orders exist in the historical feature window."
    )


# ============================================================
# 4. BUILD CUSTOMER FEATURES
# ============================================================

def build_features(
    source_orders: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:

    grouped = source_orders.groupby("Customer_ID")

    features = grouped.agg(
        Total_Orders=("Order_ID", "nunique"),
        First_Order_Date=("Order_Date", "min"),
        Last_Order_Date=("Order_Date", "max"),
        Active_Order_Days=("Order_Date", "nunique"),
    ).reset_index()

    features["Recency_Days"] = (
        reference_date - features["Last_Order_Date"]
    ).dt.days

    features["Customer_Tenure_Days"] = (
        features["Last_Order_Date"]
        - features["First_Order_Date"]
    ).dt.days

    features["Active_Months"] = (
        (
            features["Last_Order_Date"].dt.year
            - features["First_Order_Date"].dt.year
        ) * 12
        + (
            features["Last_Order_Date"].dt.month
            - features["First_Order_Date"].dt.month
        )
        + 1
    )

    features["Orders_Per_Active_Month"] = (
        features["Total_Orders"]
        / features["Active_Months"].clip(lower=1)
    )

    return features


training_features = build_features(
    feature_orders,
    snapshot_date,
)


# ============================================================
# 5. CREATE OBSERVED 24-MONTH CHURN LABELS
# ============================================================

future_order_counts = (
    outcome_orders.groupby("Customer_ID")["Order_ID"]
    .nunique()
)

training_features["Orders_In_Next_24_Months"] = (
    training_features["Customer_ID"]
    .map(future_order_counts)
    .fillna(0)
    .astype(int)
)

# Operational definition:
# Churned = customer made zero orders in the next 24 months.
training_features["Churn_24_Months"] = (
    training_features["Orders_In_Next_24_Months"] == 0
).astype(int)

training_features["Snapshot_Date"] = snapshot_date.date()

training_columns = [
    "Customer_ID",
    "Snapshot_Date",
    "Total_Orders",
    "Recency_Days",
    "Customer_Tenure_Days",
    "Active_Months",
    "Orders_Per_Active_Month",
    "Orders_In_Next_24_Months",
    "Churn_24_Months",
]

training_data = training_features[training_columns].copy()

training_data.to_csv(TRAINING_FILE, index=False)


# ============================================================
# 6. BUILD CURRENT CUSTOMER SCORING DATA
# ============================================================

# Use the latest 12 months of known history to create features
# with the same feature definitions as the historical cohort.
scoring_start = (
    data_end - pd.DateOffset(months=FEATURE_WINDOW_MONTHS)
).normalize()

scoring_orders = orders[
    (orders["Order_Date"] >= scoring_start)
    & (orders["Order_Date"] <= data_end)
].copy()

scoring_data = build_features(scoring_orders, data_end)
scoring_data["Snapshot_Date"] = data_end.date()

scoring_columns = [
    "Customer_ID",
    "Snapshot_Date",
    "Total_Orders",
    "Recency_Days",
    "Customer_Tenure_Days",
    "Active_Months",
    "Orders_Per_Active_Month",
]

scoring_data = scoring_data[scoring_columns].copy()
scoring_data.to_csv(SCORING_FILE, index=False)


# ============================================================
# 7. DATA QUALITY REPORT
# ============================================================

report = pd.DataFrame([
    {
        "Metric": "Historical data start",
        "Value": str(data_start.date()),
    },
    {
        "Metric": "Historical data end",
        "Value": str(data_end.date()),
    },
    {
        "Metric": "Historical snapshot date",
        "Value": str(snapshot_date.date()),
    },
    {
        "Metric": "Feature window months",
        "Value": FEATURE_WINDOW_MONTHS,
    },
    {
        "Metric": "Churn outcome horizon months",
        "Value": HORIZON_MONTHS,
    },
    {
        "Metric": "Training customers",
        "Value": len(training_data),
    },
    {
        "Metric": "Historical churned customers",
        "Value": int(training_data["Churn_24_Months"].sum()),
    },
    {
        "Metric": "Historical non-churned customers",
        "Value": int((training_data["Churn_24_Months"] == 0).sum()),
    },
    {
        "Metric": "Current customers to score",
        "Value": len(scoring_data),
    },
])

report.to_csv(REPORT_FILE, index=False)

print("\nChurn label distribution:")
print(training_data["Churn_24_Months"].value_counts().sort_index())

print("\nTraining dataset:", TRAINING_FILE)
print("Scoring dataset :", SCORING_FILE)
print("Quality report  :", REPORT_FILE)

print("\nQuality report:")
print(report.to_string(index=False))

print("\nData preparation completed.")
print(
    "\nIMPORTANT: This creates one historical cohort. "
    "A single cohort is not enough to establish robust "
    "out-of-time model validation."
)
