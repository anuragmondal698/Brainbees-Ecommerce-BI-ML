
from pathlib import Path
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.dialects.mysql import insert


# ============================================================
# 1. PROJECT PATHS AND ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_demand_forecast_3_years.csv"
)

TABLE_NAME = "ml_product_demand_forecast"

load_dotenv(ENV_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Forecast CSV not found:\n{CSV_PATH}\n"
        "Run forecast_product_demand_3_years.py first."
    )


# ============================================================
# 2. MYSQL CONNECTION
# ============================================================

required_env = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing_env = [key for key in required_env if not os.getenv(key)]

if missing_env:
    raise EnvironmentError(
        f"Missing environment variables in {ENV_PATH}: {missing_env}"
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
    connect_args={"connect_timeout": 10},
)


# ============================================================
# 3. LOAD FORECAST CSV
# ============================================================

print("=" * 65)
print("PRODUCT DEMAND FORECAST - MYSQL LOADER")
print("=" * 65)

df = pd.read_csv(CSV_PATH)

print(f"\nSource file: {CSV_PATH}")
print(f"Rows loaded: {len(df):,}")
print("\nColumns found:")
print(df.columns.tolist())

if df.empty:
    raise ValueError("The forecast CSV contains no rows.")

# Remove accidental spaces from column names.
df.columns = df.columns.str.strip()

# Required source columns.
required_columns = [
    "Product_ID",
    "Forecast_Month",
    "Predicted_Units",
    "Model_Name",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from CSV: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Create optional columns if they are absent.
if "Product_Name" not in df.columns:
    df["Product_Name"] = pd.NA

if "Category_ID" not in df.columns:
    df["Category_ID"] = pd.NA


# ============================================================
# 4. CLEAN DATA TYPES
# ============================================================

# IDs must remain strings: examples PRD001 and CAT001.
df["Product_ID"] = (
    df["Product_ID"].astype("string").str.strip()
)

df["Category_ID"] = (
    df["Category_ID"].astype("string").str.strip()
)

df["Product_Name"] = (
    df["Product_Name"].astype("string").str.strip()
)

df["Model_Name"] = (
    df["Model_Name"].astype("string").str.strip()
)

# Convert empty strings to missing values.
for col in ["Product_ID", "Category_ID", "Product_Name", "Model_Name"]:
    df[col] = df[col].replace("", pd.NA)

# Parse dates and predicted units.
df["Forecast_Month"] = pd.to_datetime(
    df["Forecast_Month"],
    errors="coerce",
)

df["Predicted_Units"] = pd.to_numeric(
    df["Predicted_Units"],
    errors="coerce",
)

# Store forecast months as the first day of each month.
df["Forecast_Month"] = (
    df["Forecast_Month"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ============================================================
# 5. DIAGNOSE MISSING VALUES
# ============================================================

print("\nMissing values by column:")
print(df.isna().sum().to_string())

print("\nRows containing missing values:")
diagnostic_columns = [
    "Product_ID",
    "Product_Name",
    "Category_ID",
    "Forecast_Month",
    "Predicted_Units",
    "Model_Name",
]

missing_rows = df[
    df[diagnostic_columns].isna().any(axis=1)
]

if not missing_rows.empty:
    print(
        missing_rows[diagnostic_columns]
        .head(20)
        .to_string(index=False)
    )
    print(
        f"\nTotal rows with at least one missing value: "
        f"{len(missing_rows):,}"
    )

# Category_ID and Product_Name are allowed to be missing.
# The following fields are essential for a valid forecast.
critical_columns = [
    "Product_ID",
    "Forecast_Month",
    "Predicted_Units",
    "Model_Name",
]

critical_missing = df[critical_columns].isna().any(axis=1)

if critical_missing.any():
    print("\nRows with missing critical fields:")
    print(
        df.loc[critical_missing, diagnostic_columns]
        .head(50)
        .to_string(index=False)
    )

    raise ValueError(
        "Critical forecast values are missing. "
        "Fix the source CSV before loading. "
        "No rows have been inserted by this script."
    )

# Negative demand predictions are not valid for this use case.
negative_demand = df["Predicted_Units"] < 0

if negative_demand.any():
    print("\nRows with negative predicted demand:")
    print(
        df.loc[
            negative_demand,
            ["Product_ID", "Forecast_Month", "Predicted_Units"],
        ].head(20).to_string(index=False)
    )

    raise ValueError(
        "Negative predicted demand detected. "
        "Review the forecast-generation script before loading."
    )


# ============================================================
# 6. CHECK DUPLICATES
# ============================================================

duplicate_mask = df.duplicated(
    subset=["Product_ID", "Forecast_Month"],
    keep=False,
)

if duplicate_mask.any():
    print("\nDuplicate product/month forecast combinations:")
    print(
        df.loc[
            duplicate_mask,
            ["Product_ID", "Forecast_Month", "Predicted_Units"],
        ].head(30).to_string(index=False)
    )

    raise ValueError(
        "Duplicate Product_ID + Forecast_Month combinations found. "
        "Resolve duplicates before loading."
    )

print("\nDuplicate product/month combinations: 0")


# ============================================================
# 7. ENSURE MYSQL TABLE AND TEXT ID TYPES
# ============================================================

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ml_product_demand_forecast (
            Forecast_ID BIGINT AUTO_INCREMENT PRIMARY KEY,
            Product_ID VARCHAR(50) NOT NULL,
            Product_Name VARCHAR(255) NULL,
            Category_ID VARCHAR(50) NULL,
            Forecast_Month DATE NOT NULL,
            Predicted_Units DECIMAL(15,2) NOT NULL,
            Model_Name VARCHAR(100) NOT NULL,
            Created_At TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uq_product_forecast_month
                (Product_ID, Forecast_Month),
            INDEX idx_product_forecast_month (Forecast_Month),
            INDEX idx_product_forecast_category (Category_ID)
        )
    """))

    # Existing tables may have been created with numeric ID columns.
    # Modify them to VARCHAR so PRD001 and CAT001 are preserved.
    conn.execute(text("""
        ALTER TABLE ml_product_demand_forecast
        MODIFY COLUMN Product_ID VARCHAR(50) NOT NULL,
        MODIFY COLUMN Category_ID VARCHAR(50) NULL
    """))

print("\nMySQL table verified; IDs are stored as text.")


# ============================================================
# 8. RECOVER OPTIONAL PRODUCT DETAILS FROM DIMENSION TABLE
# ============================================================

# Fill missing product names/category IDs when a matching product
# exists in dim_products. Missing categories remain SQL NULL.
with engine.connect() as conn:
    try:
        products_dim = pd.read_sql(
            text("""
                SELECT Product_ID, Product_Name, Category_ID
                FROM dim_products
            """),
            conn,
        )
    except Exception as exc:
        print(
            "\nWarning: Could not read dim_products to recover "
            f"optional product details: {exc}"
        )
        products_dim = pd.DataFrame(
            columns=["Product_ID", "Product_Name", "Category_ID"]
        )

if not products_dim.empty:
    products_dim["Product_ID"] = (
        products_dim["Product_ID"].astype("string").str.strip()
    )

    if "Product_Name" in products_dim.columns:
        products_dim["Product_Name"] = (
            products_dim["Product_Name"].astype("string").str.strip()
        )

    if "Category_ID" in products_dim.columns:
        products_dim["Category_ID"] = (
            products_dim["Category_ID"].astype("string").str.strip()
        )

    # Only use one dimension record per Product_ID.
    products_dim = products_dim.drop_duplicates(
        subset=["Product_ID"],
        keep="first",
    )

    lookup = products_dim.set_index("Product_ID")

    name_lookup = (
        lookup["Product_Name"]
        if "Product_Name" in lookup.columns
        else pd.Series(dtype="string")
    )

    category_lookup = (
        lookup["Category_ID"]
        if "Category_ID" in lookup.columns
        else pd.Series(dtype="string")
    )

    missing_name = df["Product_Name"].isna()
    df.loc[missing_name, "Product_Name"] = (
        df.loc[missing_name, "Product_ID"].map(name_lookup)
    )

    missing_category = df["Category_ID"].isna()
    df.loc[missing_category, "Category_ID"] = (
        df.loc[missing_category, "Product_ID"].map(category_lookup)
    )

print("\nMissing values after dimension lookup:")
print(df[diagnostic_columns].isna().sum().to_string())

if df["Product_Name"].isna().any():
    print(
        "\nNote: Some Product_Name values remain missing because "
        "no matching product name was found. They will be stored as NULL."
    )

if df["Category_ID"].isna().any():
    print(
        "\nNote: Some Category_ID values remain missing. "
        "They will be stored as NULL, not converted to fake IDs."
    )


# ============================================================
# 9. PREPARE MYSQL RECORDS
# ============================================================

# Round to 2 decimal places for the DECIMAL(15,2) target column.
df["Predicted_Units"] = df["Predicted_Units"].round(2)

records = []

for row in df.itertuples(index=False):
    records.append({
        "Product_ID": str(row.Product_ID),
        "Product_Name": (
            None if pd.isna(row.Product_Name)
            else str(row.Product_Name)
        ),
        "Category_ID": (
            None if pd.isna(row.Category_ID)
            else str(row.Category_ID)
        ),
        "Forecast_Month": row.Forecast_Month.date(),
        "Predicted_Units": float(row.Predicted_Units),
        "Model_Name": str(row.Model_Name),
    })

print(f"\nRecords ready for MySQL: {len(records):,}")


# ============================================================
# 10. INSERT OR UPDATE FORECASTS
# ============================================================

if records:
    with engine.begin() as conn:
        stmt = insert(
            __import__("sqlalchemy").Table(
                TABLE_NAME,
                __import__("sqlalchemy").MetaData(),
                autoload_with=conn,
            )
        ).values(records)

        upsert_stmt = stmt.on_duplicate_key_update(
            Product_Name=stmt.inserted.Product_Name,
            Category_ID=stmt.inserted.Category_ID,
            Predicted_Units=stmt.inserted.Predicted_Units,
            Model_Name=stmt.inserted.Model_Name,
            Created_At=text("CURRENT_TIMESTAMP"),
        )

        result = conn.execute(upsert_stmt)

    print("\nForecast data loaded successfully.")
    print(f"Rows processed: {result.rowcount:,}")
else:
    print("\nNo records to load.")


# ============================================================
# 11. VERIFY MYSQL RESULTS
# ============================================================

with engine.connect() as conn:
    summary = pd.read_sql(
        text("""
            SELECT
                COUNT(*) AS Total_Forecast_Rows,
                COUNT(DISTINCT Product_ID) AS Total_Products,
                COUNT(DISTINCT Forecast_Month) AS Forecast_Months,
                MIN(Forecast_Month) AS First_Forecast_Month,
                MAX(Forecast_Month) AS Last_Forecast_Month
            FROM ml_product_demand_forecast
        """),
        conn,
    )

    print("\nMySQL forecast summary:")
    print(summary.to_string(index=False))

    sample = pd.read_sql(
        text("""
            SELECT
                Product_ID,
                Product_Name,
                Category_ID,
                Forecast_Month,
                Predicted_Units,
                Model_Name
            FROM ml_product_demand_forecast
            ORDER BY Product_ID, Forecast_Month
            LIMIT 10
        """),
        conn,
    )

    print("\nSample records:")
    print(sample.to_string(index=False))

engine.dispose()

print("\n" + "=" * 65)
print("PRODUCT DEMAND FORECAST LOAD COMPLETED")
print("=" * 65)

