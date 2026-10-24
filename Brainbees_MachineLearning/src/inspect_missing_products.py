
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

missing_ids = [
    "PRD030", "PRD035", "PRD037", "PRD092",
    "PRD097", "PRD134", "PRD192"
]

products = pd.read_csv(
    RAW / "stg_products.csv",
    dtype=str
)
categories = pd.read_csv(
    RAW / "stg_categories.csv",
    dtype=str
)

products.columns = products.columns.str.strip()
categories.columns = categories.columns.str.strip()

products["Product_ID"] = products["Product_ID"].str.strip()
categories["Category_ID"] = categories["Category_ID"].str.strip()

matches = products[
    products["Product_ID"].isin(missing_ids)
].copy()

if "Category_ID" in matches.columns and "Category_ID" in categories.columns:
    valid_category_ids = set(categories["Category_ID"].dropna())
    matches["Category_Exists_In_Staging"] = (
        matches["Category_ID"].isin(valid_category_ids)
    )

print("=== MATCHING STAGING PRODUCT RECORDS ===")
if matches.empty:
    print("No matching product rows found.")
else:
    print(matches.to_string(index=False))

OUT.mkdir(parents=True, exist_ok=True)
matches.to_csv(
    OUT / "missing_product_staging_details.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved details to missing_product_staging_details.csv")


# Compare missing products against the dimension table
dim_products = pd.read_csv(
    RAW / "dim_products.csv",
    dtype=str
)
dim_categories = pd.read_csv(
    RAW / "dim_categories.csv",
    dtype=str
)

for df in [dim_products, dim_categories]:
    df.columns = df.columns.str.strip()

dim_products["Product_ID"] = (
    dim_products["Product_ID"].fillna("").str.strip()
)
dim_categories["Category_ID"] = (
    dim_categories["Category_ID"].fillna("").str.strip()
)

print("\n=== PRODUCT DIMENSION COMPARISON ===")

for product_id in ["PRD030", "PRD134"]:
    staging_row = products[
        products["Product_ID"] == product_id
    ]
    dimension_row = dim_products[
        dim_products["Product_ID"] == product_id
    ]

    print(f"\nProduct: {product_id}")
    print("In staging:", not staging_row.empty)
    print("In dimension:", not dimension_row.empty)

    if not staging_row.empty:
        category_id = staging_row.iloc[0]["Category_ID"]
        category_exists = category_id in set(
            dim_categories["Category_ID"]
        )
        print("Category ID:", category_id)
        print("Category exists in dimension:", category_exists)
