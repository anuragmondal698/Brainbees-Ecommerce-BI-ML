
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = PROJECT_ROOT / "data" / "raw" / "vw_monthly_sales.csv"

if not FILE_PATH.exists():
    raise FileNotFoundError(f"File not found: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)

print("\n===== MONTHLY SALES DATA =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes.to_string())

print("\n===== FIRST 10 ROWS =====")
print(df.head(10).to_string(index=False))

print("\n===== MISSING VALUES =====")
print(df.isna().sum().to_string())

print("\n===== NUMERIC SUMMARY =====")
print(df.describe(include="number").to_string())
