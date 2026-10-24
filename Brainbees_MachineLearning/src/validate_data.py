
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

checks = []


def load_csv(name):
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        print(f"[SKIP] File not found: {path.name}")
        return None
    return pd.read_csv(path)


def add_check(table, check, count, status):
    checks.append({
        "table": table,
        "check": check,
        "issue_count": int(count),
        "status": status,
    })


# Orders: dates, IDs, amounts and duplicate order IDs
orders = load_csv("fact_orders")
if orders is not None:
    for col in ["Order_ID", "Customer_ID", "Order_Date", "Order_Total"]:
        if col in orders.columns:
            add_check(
                "fact_orders",
                f"Missing {col}",
                orders[col].isna().sum(),
                "REVIEW" if orders[col].isna().any() else "PASS",
            )

    if "Order_Date" in orders.columns:
        parsed = pd.to_datetime(orders["Order_Date"], errors="coerce")
        add_check(
            "fact_orders",
            "Unparseable Order_Date",
            parsed.isna().sum() - orders["Order_Date"].isna().sum(),
            "REVIEW" if (parsed.isna() & orders["Order_Date"].notna()).any()
            else "PASS",
        )

    if "Order_ID" in orders.columns:
        duplicate_ids = orders["Order_ID"].duplicated(keep=False).sum()
        add_check(
            "fact_orders",
            "Rows with repeated Order_ID",
            duplicate_ids,
            "REVIEW" if duplicate_ids else "PASS",
        )

    if "Order_Total" in orders.columns:
        amount = pd.to_numeric(orders["Order_Total"], errors="coerce")
        invalid = amount.isna().sum() - orders["Order_Total"].isna().sum()
        add_check(
            "fact_orders",
            "Non-numeric Order_Total",
            invalid,
            "REVIEW" if invalid else "PASS",
        )


# Order items: identifiers, quantity, revenue and duplicate item IDs
items = load_csv("fact_order_items")
if items is not None:
    for col in ["Order_Item_ID", "Order_ID", "Product_ID", "Quantity", "Line_Total"]:
        if col in items.columns:
            add_check(
                "fact_order_items",
                f"Missing {col}",
                items[col].isna().sum(),
                "REVIEW" if items[col].isna().any() else "PASS",
            )

    if "Order_Item_ID" in items.columns:
        repeated = items["Order_Item_ID"].duplicated(keep=False).sum()
        add_check(
            "fact_order_items",
            "Rows with repeated Order_Item_ID",
            repeated,
            "REVIEW" if repeated else "PASS",
        )

    for col in ["Quantity", "Line_Total"]:
        if col in items.columns:
            numeric = pd.to_numeric(items[col], errors="coerce")
            invalid = numeric.isna().sum() - items[col].isna().sum()
            add_check(
                "fact_order_items",
                f"Non-numeric {col}",
                invalid,
                "REVIEW" if invalid else "PASS",
            )

            if col == "Quantity":
                nonpositive = (numeric <= 0).sum()
                add_check(
                    "fact_order_items",
                    "Non-positive Quantity",
                    nonpositive,
                    "REVIEW" if nonpositive else "PASS",
                )


# Customer and product foreign-key checks
customers = load_csv("dim_customers")
products = load_csv("dim_products")

if orders is not None and customers is not None:
    if {"Customer_ID"}.issubset(orders.columns) and "Customer_ID" in customers.columns:
        valid_ids = set(customers["Customer_ID"].dropna())
        invalid = (
            orders["Customer_ID"].notna()
            & ~orders["Customer_ID"].isin(valid_ids)
        ).sum()
        add_check(
            "fact_orders",
            "Customer_ID not found in dim_customers",
            invalid,
            "REVIEW" if invalid else "PASS",
        )

if items is not None and products is not None:
    if "Product_ID" in items.columns and "Product_ID" in products.columns:
        valid_ids = set(products["Product_ID"].dropna())
        invalid = (
            items["Product_ID"].notna()
            & ~items["Product_ID"].isin(valid_ids)
        ).sum()
        add_check(
            "fact_order_items",
            "Product_ID not found in dim_products",
            invalid,
            "REVIEW" if invalid else "PASS",
        )

report = pd.DataFrame(checks)
report.to_csv(
    OUT_DIR / "data_validation_report.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== DATA VALIDATION REPORT ===")
print(report.to_string(index=False))
print(f"\nSaved report: {OUT_DIR / 'data_validation_report.csv'}")
print("\nValidation checks completed. REVIEW means investigate; it does not automatically mean the data is wrong.")
