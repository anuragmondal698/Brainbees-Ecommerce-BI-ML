
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "customer_clv_scoring_data.csv"
MODEL_PATH = BASE_DIR / "models" / "customer_clv_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "customer_clv_feature_columns.joblib"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_clv_predictions.csv"

def main():
    print("Loading customer scoring data...")
    df = pd.read_csv(DATA_PATH)

    print("Loading trained CLV model...")
    artifact = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    missing_columns = [
        col for col in feature_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required model features: {missing_columns}"
        )

    X = df[feature_columns].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError(
            "Model features contain missing or non-numeric values. "
            "Check customer_clv_scoring_data.csv."
        )

    model = artifact["model"]
    predictions = model.predict(X)

    df["Predicted_Future_Revenue_12M"] = predictions.clip(min=0)
    df["CLV_Model_Name"] = artifact["model_name"]
    df["Prediction_Horizon_Months"] = 12
    df["Prediction_Date"] = pd.Timestamp.today().normalize()

    df["CLV_Segment"] = pd.cut(
        df["Predicted_Future_Revenue_12M"],
        bins=[-0.01, 1000, 5000, 15000, float("inf")],
        labels=[
            "Low Value",
            "Medium Value",
            "High Value",
            "Very High Value",
        ],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCLV predictions completed.")
    print(f"Customers scored: {len(df)}")
    print(f"Model: {artifact['model_name']}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nCLV segment counts:")
    print(df["CLV_Segment"].value_counts(dropna=False))

    print("\nTop 10 customers by predicted future revenue:")
    print(
        df.sort_values(
            "Predicted_Future_Revenue_12M",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()