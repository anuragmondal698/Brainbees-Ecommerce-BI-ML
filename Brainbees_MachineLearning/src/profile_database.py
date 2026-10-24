
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, inspect, text

# --------------------------------------------------
# 1. Project paths and environment configuration
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


# --------------------------------------------------
# 2. Connect to MySQL
# --------------------------------------------------
required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
missing = [key for key in required if not os.getenv(key)]

if missing:
    raise RuntimeError(
        f"Missing environment settings in .env: {', '.join(missing)}"
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
    connect_args={"connect_timeout": 10},
)


def discover_objects(engine):
    """Find fact, dimension, staging tables and vw_ SQL views."""
    inspector = inspect(engine)
    objects = []

    for name in inspector.get_table_names():
        lower_name = name.lower()

        if lower_name.startswith(("fact_", "dim_")):
            objects.append((name, "table"))
        elif lower_name.startswith("stg_") and env_flag("LOAD_STG"):
            objects.append((name, "staging_table"))

    for name in inspector.get_view_names():
        if name.lower().startswith("vw_"):
            objects.append((name, "view"))

    return sorted(objects, key=lambda item: item[0].lower())


def profile_dataframe(df, object_name, object_type):
    """Produce one profile row for every column."""
    rows = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)

        rows.append({
            "object_name": object_name,
            "object_type": object_type,
            "row_count": len(df),
            "column_name": str(column),
            "data_type": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_percentage": round(
                float(series.isna().mean() * 100), 2
            ) if len(df) else 0,
            "distinct_values": int(series.nunique(dropna=True)),
            "sample_values": " | ".join(
                non_null.astype(str).head(3).tolist()
            ),
            "minimum": (
                str(non_null.min())
                if is_numeric and not non_null.empty else ""
            ),
            "maximum": (
                str(non_null.max())
                if is_numeric and not non_null.empty else ""
            ),
        })

    return rows


# --------------------------------------------------
# 3. Load, profile and save each database object
# --------------------------------------------------
try:
    with engine.connect() as connection:
        db_name = connection.execute(
            text("SELECT DATABASE()")
        ).scalar_one()

        print("MySQL connection successful!")
        print("Database:", db_name)

    objects = discover_objects(engine)

    if not objects:
        raise RuntimeError(
            "No matching fact_, dim_, stg_ or vw_ objects found. "
            "Check your database and object names."
        )

    print(f"\nDiscovered {len(objects)} objects.")

    summaries = []
    column_profiles = []

    for name, object_type in objects:
        # Quote the discovered table/view identifier.
        quoted_name = "`" + name.replace("`", "``") + "`"

        df = pd.read_sql_query(
            text(f"SELECT * FROM {quoted_name}"),
            engine,
        )

        duplicate_rows = (
            int(df.duplicated().sum()) if not df.empty else 0
        )

        summaries.append({
            "object_name": name,
            "object_type": object_type,
            "row_count": len(df),
            "column_count": len(df.columns),
            "duplicate_rows": duplicate_rows,
            "load_status": "SUCCESS",
        })

        column_profiles.extend(
            profile_dataframe(df, name, object_type)
        )

        print(
            f"[OK] {name}: {len(df):,} rows x "
            f"{len(df.columns)} columns"
        )

        if env_flag("EXPORT_CSV", True):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

            df.to_csv(
                RAW_DIR / f"{safe_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        # Only one source DataFrame is retained at a time.
        del df

    # --------------------------------------------------
    # 4. Save profiling reports
    # --------------------------------------------------
    summary_df = pd.DataFrame(summaries)
    profile_df = pd.DataFrame(column_profiles)

    summary_df.to_csv(
        PROCESSED_DIR / "database_object_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    profile_df.to_csv(
        PROCESSED_DIR / "column_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== DATABASE OBJECT SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\nSaved object profile:")
    print(PROCESSED_DIR / "database_object_profile.csv")

    print("\nSaved column profile:")
    print(PROCESSED_DIR / "column_profile.csv")

    if env_flag("EXPORT_CSV", True):
        print("\nCSV snapshots saved in:")
        print(RAW_DIR)

    print("\nDATA LOADING AND PROFILING TEST: PASSED")

finally:
    engine.dispose()

