
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text


# ============================================================
# 1. PATHS AND DATABASE CONNECTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_FILE = OUTPUT_DIR / "customer_clv_training_data.csv"
SCORING_FILE = OUTPUT_DIR / "customer_clv_scoring_data.csv"
REPORT_FILE = OUTPUT_DIR / "customer_clv_quality_report.csv"

required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise EnvironmentError(
        f"Missing .env variables: {missing_env}"
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
# 2. LOAD ORDER AND ORDER-ITEM DATA
# ============================================================

orders_query = text("""
    SELECT
        Order_ID,
        Customer_ID,
        Order_Date
    FROM fact_orders
    WHERE Order_ID IS NOT NULL
      AND Customer_ID IS NOT NULL
      AND Order_Date IS NOT NULL
""")

items_query = text("""
    SELECT
        Order_ID,
        Order_Item_ID,
        Quantity,
        Unit_Price,
        Discount_Amount,
        Line_Total,
        Profit_Amount
    FROM fact_order_items
    WHERE Order_ID IS NOT NULL
""")

try:
    with engine.connect() as conn:
        orders = pd.read_sql(orders_query, conn)
        items = pd.read_sql(items_query, conn)
finally:
    engine.dispose()

if orders.empty or items.empty:
    raise ValueError(
        "Orders or order items are empty. Check the source tables."
    )

orders["Order_Date"] = pd.to_datetime(
    orders["Order_Date"], errors="coerce"
)

orders["Order_ID"] = orders["Order_ID"].astype("string").str.strip()
orders["Customer_ID"] = (
    orders["Customer_ID"].astype("string").str.strip()
)
items["Order_ID"] = items["Order_ID"].astype("string").str.strip()

orders = orders.dropna(
    subset=["Order_ID", "Customer_ID", "Order_Date"]
)

# Convert financial and quantity columns safely.
for col in [
    "Quantity",
    "Unit_Price",
    "Discount_Amount",
    "Line_Total",
    "Profit_Amount",
]:
    items[col] = pd.to_numeric(items[col], errors="coerce")

# One record per order in fact_orders is expected.
duplicate_orders = orders["Order_ID"].duplicated().sum()

if duplicate_orders:
    raise ValueError(
        f"Found {duplicate_orders} duplicate Order_ID rows in "
        "fact_orders. Resolve these before preparing CLV data."
    )


# ============================================================
# 3. AGGREGATE ORDER-ITEM FINANCIALS BY ORDER
# ============================================================

order_financials = (
    items.groupby("Order_ID", as_index=False)
    .agg(
        Order_Revenue=("Line_Total", "sum"),
        Order_Profit=("Profit_Amount", "sum"),
        Order_Units=("Quantity", "sum"),
        Order_Item_Count=("Order_Item_ID", "nunique"),
        Missing_Line_Total=("Line_Total", lambda s: s.isna().sum()),
    )
)

# Join order financials to customer and date information.
transactions = orders.merge(
    order_financials,
    on="Order_ID",
    how="inner",
    validate="one_to_one",
)

if transactions.empty:
    raise ValueError(
        "No orders matched order items. Check Order_ID formats."
    )

# Exclude records whose total revenue cannot be calculated.
missing_revenue_orders = transactions["Order_Revenue"].isna().sum()

transactions = transactions.dropna(
    subset=["Customer_ID", "Order_Date", "Order_Revenue"]
)

transactions = transactions.sort_values("Order_Date")

data_start = transactions["Order_Date"].min().normalize()
data_end = transactions["Order_Date"].max().normalize()

print("=" * 65)
print("BRAINBEES CUSTOMER LIFETIME VALUE DATA PREPARATION")
print("=" * 65)
print(f"First transaction: {data_start.date()}")
print(f"Last transaction : {data_end.date()}")
print(f"Matched orders   : {transactions['Order_ID'].nunique():,}")
print(f"Matched customers: {transactions['Customer_ID'].nunique():,}")
print(f"Orders with missing revenue excluded: {missing_revenue_orders:,}")


# ============================================================
# 4. DEFINE FEATURE AND TARGET WINDOWS
# ============================================================

LOOKBACK_MONTHS = 12
TARGET_MONTHS = 12

# Historical snapshot dates are monthly.
# A complete 12-month lookback and 12-month outcome window
# must both fit inside the observed dataset.
first_snapshot = (
    data_start.to_period("M").to_timestamp()
    + pd.DateOffset(months=LOOKBACK_MONTHS + 1)
)

last_snapshot = (
    data_end.to_period("M").to_timestamp()
    - pd.DateOffset(months=TARGET_MONTHS)
    + pd.DateOffset(days=1)
)

snapshot_dates = pd.date_range(
    start=first_snapshot,
    end=last_snapshot,
    freq="MS",
)

if len(snapshot_dates) == 0:
    raise ValueError(
        "Not enough historical data for a 12-month lookback "
        "and a complete 12-month future revenue target."
    )

print(f"\nHistorical snapshots available: {len(snapshot_dates)}")


# ============================================================
# 5. BUILD FEATURES FOR A SNAPSHOT
# ============================================================

def build_customer_features(
    source: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:

    lookback_start = (
        snapshot_date - pd.DateOffset(months=LOOKBACK_MONTHS)
    )

    history = source[
        (source["Order_Date"] >= lookback_start)
        & (source["Order_Date"] < snapshot_date)
    ].copy()

    # Customers known before the snapshot.
    known_customers = source[
        source["Order_Date"] < snapshot_date
    ]["Customer_ID"].dropna().unique()

    if len(known_customers) == 0:
        return pd.DataFrame()

    if history.empty:
        result = pd.DataFrame({
            "Customer_ID": known_customers
        })
        result["Total_Orders_12M"] = 0
        result["Revenue_12M"] = 0.0
        result["Profit_12M"] = 0.0
        result["Units_12M"] = 0
        result["Average_Order_Value_12M"] = 0.0
        result["Recency_Days"] = LOOKBACK_MONTHS * 30
        result["Active_Months_12M"] = 0
        result["Snapshot_Date"] = snapshot_date
        return result

    features = (
        history.groupby("Customer_ID")
        .agg(
            Total_Orders_12M=("Order_ID", "nunique"),
            Revenue_12M=("Order_Revenue", "sum"),
            Profit_12M=("Order_Profit", "sum"),
            Units_12M=("Order_Units", "sum"),
            First_Order_Date=("Order_Date", "min"),
            Last_Order_Date=("Order_Date", "max"),
            Active_Months_12M=(
                "Order_Date",
                lambda s: s.dt.to_period("M").nunique(),
            ),
        )
        .reset_index()
    )

    # Retain known customers with zero orders in the lookback.
    features = pd.DataFrame({
        "Customer_ID": known_customers
    }).merge(
        features,
        on="Customer_ID",
        how="left",
    )

    features["Total_Orders_12M"] = (
        features["Total_Orders_12M"].fillna(0).astype(int)
    )

    for col in ["Revenue_12M", "Profit_12M", "Units_12M"]:
        features[col] = features[col].fillna(0)

    features["Active_Months_12M"] = (
        features["Active_Months_12M"].fillna(0).astype(int)
    )

    features["Average_Order_Value_12M"] = (
        features["Revenue_12M"]
        / features["Total_Orders_12M"].replace(0, pd.NA)
    ).fillna(0)

    features["Recency_Days"] = (
        snapshot_date - features["Last_Order_Date"]
    ).dt.days

    features["Recency_Days"] = (
        features["Recency_Days"]
        .fillna(LOOKBACK_MONTHS * 30)
        .clip(lower=0)
    )

    features["Snapshot_Date"] = snapshot_date

    return features


# ============================================================
# 6. BUILD HISTORICAL TRAINING SNAPSHOTS
# ============================================================

training_frames = []

for snapshot_date in snapshot_dates:
    features = build_customer_features(
        transactions,
        snapshot_date,
    )

    if features.empty:
        continue

    target_end = snapshot_date + pd.DateOffset(
        months=TARGET_MONTHS
    )

    future = transactions[
        (transactions["Order_Date"] >= snapshot_date)
        & (transactions["Order_Date"] < target_end)
    ]

    future_revenue = (
        future.groupby("Customer_ID")["Order_Revenue"]
        .sum()
        .rename("Future_Revenue_12M")
    )

    features = features.merge(
        future_revenue,
        on="Customer_ID",
        how="left",
    )

    features["Future_Revenue_12M"] = (
        features["Future_Revenue_12M"].fillna(0)
    )

    training_frames.append(features)

if not training_frames:
    raise ValueError("No historical CLV training snapshots were created.")

training_data = pd.concat(
    training_frames,
    ignore_index=True,
)

training_data["Snapshot_Date"] = pd.to_datetime(
    training_data["Snapshot_Date"]
)

training_data.to_csv(TRAINING_FILE, index=False)


# ============================================================
# 7. BUILD CURRENT CUSTOMER SCORING DATA
# ============================================================

current_snapshot = data_end.normalize() + pd.Timedelta(days=1)

scoring_data = build_customer_features(
    transactions,
    current_snapshot,
)

if scoring_data.empty:
    raise ValueError("No current customer scoring data was created.")

scoring_data.to_csv(SCORING_FILE, index=False)


# ============================================================
# 8. DATA QUALITY REPORT
# ============================================================

report = pd.DataFrame([
    {
        "Metric": "First transaction date",
        "Value": str(data_start.date()),
    },
    {
        "Metric": "Last transaction date",
        "Value": str(data_end.date()),
    },
    {
        "Metric": "Lookback window months",
        "Value": LOOKBACK_MONTHS,
    },
    {
        "Metric": "Future CLV target months",
        "Value": TARGET_MONTHS,
    },
    {
        "Metric": "Historical snapshot count",
        "Value": len(snapshot_dates),
    },
    {
        "Metric": "Training rows",
        "Value": len(training_data),
    },
    {
        "Metric": "Training customers",
        "Value": training_data["Customer_ID"].nunique(),
    },
    {
        "Metric": "Training snapshots",
        "Value": training_data["Snapshot_Date"].nunique(),
    },
    {
        "Metric": "Current customers to score",
        "Value": scoring_data["Customer_ID"].nunique(),
    },
    {
        "Metric": "Training target missing values",
        "Value": int(training_data["Future_Revenue_12M"].isna().sum()),
    },
    {
        "Metric": "Orders missing line totals",
        "Value": missing_revenue_orders,
    },
])

report.to_csv(REPORT_FILE, index=False)

print("\nTraining data:", TRAINING_FILE)
print("Scoring data :", SCORING_FILE)
print("Quality report:", REPORT_FILE)

print("\nQuality report:")
print(report.to_string(index=False))

print("\nHistorical future revenue summary:")
print(
    training_data["Future_Revenue_12M"]
    .describe()
    .to_string()
)

print("\nCLV data preparation completed.")
print(
    "\nNote: Future_Revenue_12M is a revenue-based customer-value "
    "target, not discounted net profit or a guaranteed lifetime value."
)
