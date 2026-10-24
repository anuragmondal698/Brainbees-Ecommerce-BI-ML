
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

source_file = RAW_DIR / "vw_monthly_sales.csv"
df = pd.read_csv(source_file)

required = [
    "Order_Year",
    "Order_Month",
    "Revenue",
    "Total_Orders",
    "Units_Sold",
    "Profit",
]

missing_columns = [col for col in required if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Convert year, month and metrics to numeric values.
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[["Order_Year", "Order_Month"]].isna().any().any():
    raise ValueError("Some records have missing or invalid year/month values.")

if not df["Order_Year"].mod(1).eq(0).all():
    raise ValueError("Order_Year contains non-integer values.")

if not df["Order_Month"].mod(1).eq(0).all():
    raise ValueError("Order_Month contains non-integer values.")

if not df["Order_Month"].between(1, 12).all():
    raise ValueError("Order_Month must be between 1 and 12.")

# Build a proper monthly date.
df["Month_Start"] = pd.to_datetime(
    {
        "year": df["Order_Year"].astype(int),
        "month": df["Order_Month"].astype(int),
        "day": 1,
    },
    errors="raise",
)

df = df.sort_values("Month_Start").reset_index(drop=True)

print("\n=== DATASET OVERVIEW ===")
print("Rows:", len(df))
print("First month:", df["Month_Start"].min().date())
print("Last month:", df["Month_Start"].max().date())

# Check duplicate months.
duplicate_months = df[
    df["Month_Start"].duplicated(keep=False)
]

print("\n=== DUPLICATE MONTHS ===")
if duplicate_months.empty:
    print("No duplicate months found.")
else:
    print(duplicate_months[
        ["Month_Start", "Revenue"]
    ].to_string(index=False))

# Check gaps in the monthly sequence.
expected_months = pd.date_range(
    start=df["Month_Start"].min(),
    end=df["Month_Start"].max(),
    freq="MS",
)

actual_months = pd.DatetimeIndex(df["Month_Start"].unique())
missing_months = expected_months.difference(actual_months)

print("\n=== MISSING MONTHS ===")
if len(missing_months) == 0:
    print("No missing months found.")
else:
    print([d.strftime("%Y-%m") for d in missing_months])

# Check missing or invalid metrics.
metrics = ["Revenue", "Total_Orders", "Units_Sold", "Profit"]

print("\n=== MISSING METRIC VALUES ===")
print(df[metrics].isna().sum().to_string())

print("\n=== NEGATIVE REVENUE RECORDS ===")
print(df.loc[df["Revenue"] < 0, ["Month_Start", "Revenue"]].to_string(index=False))

# Flag months after the current month. Do not train on these
# until their provenance is confirmed.
current_month = pd.Timestamp.today().to_period("M").to_timestamp()
future_rows = df[df["Month_Start"] > current_month]

print("\n=== MONTHS AFTER THE CURRENT MONTH ===")
if future_rows.empty:
    print("No future months found.")
else:
    print(future_rows[
        ["Month_Start", "Revenue", "Total_Orders", "Units_Sold"]
    ].to_string(index=False))

# Save a diagnostic report without silently dropping data.
df.to_csv(
    PROCESSED_DIR / "monthly_revenue_prepared.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nPrepared dataset saved to:")
print(PROCESSED_DIR / "monthly_revenue_prepared.csv")

if not duplicate_months.empty:
    print("\nWARNING: Resolve duplicate months before training.")
elif len(missing_months) > 0:
    print("\nWARNING: Investigate missing months before training.")
else:
    print("\nMonthly sequence check: PASSED")
