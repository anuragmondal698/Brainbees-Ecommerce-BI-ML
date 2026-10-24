
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    return pd.read_csv(RAW / f"{name}.csv", dtype=str)


# -------------------------------
# Customer IDs
# -------------------------------
orders = read_csv("fact_orders")
customers = read_csv("dim_customers")
stg_customers = read_csv("stg_customers")

order_customer_ids = set(
    orders["Customer_ID"].dropna().str.strip()
)
dimension_customer_ids = set(
    customers["Customer_ID"].dropna().str.strip()
)
staging_customer_ids = set(
    stg_customers["Customer_ID"].dropna().str.strip()
)

missing_customer_ids = order_customer_ids - dimension_customer_ids

customer_results = pd.DataFrame({
    "Customer_ID": sorted(missing_customer_ids)
})
customer_results["Exists_in_stg_customers"] = (
    customer_results["Customer_ID"].isin(staging_customer_ids)
)

print("=== CUSTOMER MATCH CHECK ===")
print("Distinct IDs missing from dimension:", len(customer_results))
print(customer_results["Exists_in_stg_customers"].value_counts())
customer_results.to_csv(
    OUT / "customer_staging_match.csv",
    index=False,
    encoding="utf-8-sig",
)


# -------------------------------
# Product IDs
# -------------------------------
items = read_csv("fact_order_items")
products = read_csv("dim_products")
stg_products = read_csv("stg_products")

item_product_ids = set(
    items["Product_ID"].dropna().str.strip()
)
dimension_product_ids = set(
    products["Product_ID"].dropna().str.strip()
)
staging_product_ids = set(
    stg_products["Product_ID"].dropna().str.strip()
)

missing_product_ids = item_product_ids - dimension_product_ids

product_results = pd.DataFrame({
    "Product_ID": sorted(missing_product_ids)
})
product_results["Exists_in_stg_products"] = (
    product_results["Product_ID"].isin(staging_product_ids)
)

print("\n=== PRODUCT MATCH CHECK ===")
print("Distinct IDs missing from dimension:", len(product_results))
print(product_results["Exists_in_stg_products"].value_counts())
product_results.to_csv(
    OUT / "product_staging_match.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nReports saved in:", OUT)
