
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text


# =====================================================
# 1. PATHS AND DATABASE CONNECTION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

required_env = [
    "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"
]

missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise ValueError(f"Missing .env settings: {missing_env}")

engine_url = URL.create(
    "mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    database=os.getenv("DB_NAME"),
)

engine = create_engine(engine_url)


# =====================================================
# 2. LOAD PRODUCT-LEVEL MONTHLY DEMAND
# =====================================================

query = text("""
    SELECT
        oi.Product_ID,
        COALESCE(dp.Product_Name, 'Unknown Product') AS Product_Name,
        COALESCE(dp.Category_ID, -1) AS Category_ID,
        DATE_FORMAT(o.Order_Date, '%Y-%m-01') AS Demand_Month,
        SUM(oi.Quantity) AS Units_Sold,
        COUNT(DISTINCT o.Order_ID) AS Orders_Count
    FROM fact_order_items AS oi
    INNER JOIN fact_orders AS o
        ON oi.Order_ID = o.Order_ID
    LEFT JOIN dim_products AS dp
        ON oi.Product_ID = dp.Product_ID
    WHERE
        oi.Product_ID IS NOT NULL
        AND o.Order_Date IS NOT NULL
        AND oi.Quantity IS NOT NULL
        AND oi.Quantity >= 0
    GROUP BY
        oi.Product_ID,
        COALESCE(dp.Product_Name, 'Unknown Product'),
        COALESCE(dp.Category_ID, -1),
        DATE_FORMAT(o.Order_Date, '%Y-%m-01')
    ORDER BY oi.Product_ID, Demand_Month
""")

with engine.connect() as connection:
    demand = pd.read_sql(query, connection)

if demand.empty:
    engine.dispose()
    raise ValueError("No product demand data was returned from MySQL.")

demand["Demand_Month"] = pd.to_datetime(
    demand["Demand_Month"], errors="coerce"
)

demand["Units_Sold"] = pd.to_numeric(
    demand["Units_Sold"], errors="coerce"
)

demand["Orders_Count"] = pd.to_numeric(
    demand["Orders_Count"], errors="coerce"
)

demand = demand.dropna(
    subset=["Product_ID", "Demand_Month", "Units_Sold"]
).copy()

demand["Units_Sold"] = demand["Units_Sold"].astype(float)

if (demand["Units_Sold"] < 0).any():
    raise ValueError("Negative demand values found.")

if demand.duplicated(["Product_ID", "Demand_Month"]).any():
    raise ValueError("Duplicate product-month rows found.")


# =====================================================
# 3. CHECK HISTORICAL COVERAGE
# =====================================================

first_month = demand["Demand_Month"].min()
last_month = demand["Demand_Month"].max()

products = demand[
    ["Product_ID", "Product_Name", "Category_ID"]
].drop_duplicates("Product_ID")

print("\n========== SOURCE SUMMARY ==========")
print(f"Products with recorded sales: {len(products)}")
print(f"First historical month: {first_month:%Y-%m}")
print(f"Last historical month: {last_month:%Y-%m}")
print(f"Monthly product observations: {len(demand)}")


# =====================================================
# 4. CREATE A COMPLETE MONTHLY SERIES PER PRODUCT
# =====================================================

# For each product, begin at its first recorded sale.
# Months with no recorded sales after that point are
# represented as zero units sold.

all_products = []

for product_id, group in demand.groupby("Product_ID"):
    group = group.sort_values("Demand_Month").copy()

    product_name = group["Product_Name"].iloc[0]
    category_id = group["Category_ID"].iloc[0]

    months = pd.date_range(
        start=group["Demand_Month"].min(),
        end=last_month,
        freq="MS",
    )

    group = group.set_index("Demand_Month")

    monthly = group.reindex(months)
    monthly.index.name = "Demand_Month"

    monthly["Product_ID"] = product_id
    monthly["Product_Name"] = product_name
    monthly["Category_ID"] = category_id

    # Missing product-month records are assumed to mean
    # no recorded sales. Verify this assumption before use.
    monthly["Units_Sold"] = monthly["Units_Sold"].fillna(0)
    monthly["Orders_Count"] = monthly["Orders_Count"].fillna(0)

    monthly["Is_Zero_Demand"] = (
        monthly["Units_Sold"] == 0
    ).astype(int)

    monthly = monthly.reset_index()

    all_products.append(monthly)

monthly_demand = pd.concat(
    all_products, ignore_index=True
)

monthly_demand = monthly_demand.sort_values(
    ["Product_ID", "Demand_Month"]
).reset_index(drop=True)


# =====================================================
# 5. CREATE TIME-SERIES FEATURES
# =====================================================

monthly_demand["Month_Number"] = (
    monthly_demand["Demand_Month"].dt.month
)

monthly_demand["Time_Index"] = (
    monthly_demand.groupby("Product_ID").cumcount() + 1
)

for lag in [1, 2, 3]:
    monthly_demand[f"Demand_Lag_{lag}"] = (
        monthly_demand.groupby("Product_ID")["Units_Sold"]
        .shift(lag)
    )

monthly_demand["Demand_Rolling_Mean_3"] = (
    monthly_demand.groupby("Product_ID")["Units_Sold"]
    .transform(
        lambda series: series.shift(1).rolling(
            window=3, min_periods=3
        ).mean()
    )
)


# =====================================================
# 6. QUALITY REPORTS
# =====================================================

quality = pd.DataFrame([
    {
        "Check": "Products with recorded sales",
        "Result": int(monthly_demand["Product_ID"].nunique()),
    },
    {
        "Check": "Monthly product rows after calendar completion",
        "Result": len(monthly_demand),
    },
    {
        "Check": "Duplicate product-month rows",
        "Result": int(
            monthly_demand.duplicated(
                ["Product_ID", "Demand_Month"]
            ).sum()
        ),
    },
    {
        "Check": "Missing product IDs",
        "Result": int(monthly_demand["Product_ID"].isna().sum()),
    },
    {
        "Check": "Missing monthly demand values",
        "Result": int(monthly_demand["Units_Sold"].isna().sum()),
    },
    {
        "Check": "Product-months with zero recorded demand",
        "Result": int(monthly_demand["Is_Zero_Demand"].sum()),
    },
])

# Products without matching dimension records
missing_dimension = (
    demand.loc[
        demand["Product_Name"].eq("Unknown Product"),
        ["Product_ID"],
    ]
    .drop_duplicates()
)

quality = pd.concat([
    quality,
    pd.DataFrame([{
        "Check": "Products missing dimension match",
        "Result": len(missing_dimension),
    }])
], ignore_index=True)


# =====================================================
# 7. SAVE OUTPUT FILES
# =====================================================

monthly_path = OUTPUT_DIR / "monthly_product_demand.csv"
quality_path = OUTPUT_DIR / "product_demand_quality_report.csv"
missing_dimension_path = OUTPUT_DIR / "products_missing_dimension.csv"

monthly_demand.to_csv(monthly_path, index=False)
quality.to_csv(quality_path, index=False)
missing_dimension.to_csv(missing_dimension_path, index=False)

print("\n========== DATA QUALITY REPORT ==========")
print(quality.to_string(index=False))

print("\n========== OUTPUT FILES ==========")
print(monthly_path)
print(quality_path)
print(missing_dimension_path)

engine.dispose()

print("\nProduct demand dataset preparation completed.")
