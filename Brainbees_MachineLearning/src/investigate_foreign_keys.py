
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    return pd.read_csv(RAW_DIR / f"{name}.csv", dtype=str)


# -----------------------------------
# 1. Investigate missing customer IDs
# -----------------------------------
orders = read_csv("fact_orders")
customers = read_csv("dim_customers")

orders["Customer_ID"] = orders["Customer_ID"].str.strip()
customers["Customer_ID"] = customers["Customer_ID"].str.strip()

valid_customers = set(customers["Customer_ID"].dropna())

unmatched_orders = orders[
    orders["Customer_ID"].isna()
    | ~orders["Customer_ID"].isin(valid_customers)
].copy()

print("=== CUSTOMER ID INVESTIGATION ===")
print("Unmatched order rows:", len(unmatched_orders))
print(
    "Distinct unmatched customer IDs:",
    unmatched_orders["Customer_ID"].nunique()
)
print(unmatched_orders["Customer_ID"].value_counts().head(20))

unmatched_orders.to_csv(
    OUT_DIR / "unmatched_customer_orders.csv",
    index=False,
    encoding="utf-8-sig"
)


# -----------------------------------
# 2. Investigate missing product IDs
# -----------------------------------
items = read_csv("fact_order_items")
products = read_csv("dim_products")

items["Product_ID"] = items["Product_ID"].str.strip()
products["Product_ID"] = products["Product_ID"].str.strip()

valid_products = set(products["Product_ID"].dropna())

unmatched_items = items[
    items["Product_ID"].isna()
    | ~items["Product_ID"].isin(valid_products)
].copy()

print("\n=== PRODUCT ID INVESTIGATION ===")
print("Unmatched order-item rows:", len(unmatched_items))
print(
    "Distinct unmatched product IDs:",
    unmatched_items["Product_ID"].nunique()
)
print(unmatched_items["Product_ID"].value_counts().head(20))

unmatched_items.to_csv(
    OUT_DIR / "unmatched_product_order_items.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nInvestigation reports saved in:", OUT_DIR)
