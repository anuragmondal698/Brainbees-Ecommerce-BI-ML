
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "customer_churn_training_data.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "customer_churn_model.joblib"
FEATURES_FILE = MODEL_DIR / "customer_churn_feature_columns.joblib"
EVALUATION_FILE = DATA_DIR / "customer_churn_model_evaluation.csv"
PREDICTIONS_FILE = DATA_DIR / "customer_churn_test_predictions.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Run prepare_churn_data.py first."
    )

df = pd.read_csv(DATA_FILE)

target = "Churn_24_Months"

features = [
    "Total_Orders",
    "Recency_Days",
    "Customer_Tenure_Days",
    "Active_Months",
    "Orders_Per_Active_Month",
]

required = ["Customer_ID", target] + features
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=["Customer_ID", target]).copy()
df[target] = df[target].astype(int)

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df[target]

print("=" * 60)
print("BRAINBEES CUSTOMER CHURN MODEL")
print("=" * 60)

print(f"Training rows: {len(df)}")
print("\nChurn class distribution:")
print(y.value_counts().sort_index())

if y.nunique() < 2:
    raise ValueError(
        "Only one churn class exists. A classifier cannot learn "
        "both churn and non-churn from this dataset. Check the "
        "churn labels and collect more suitable historical data."
    )

class_counts = y.value_counts()

if class_counts.min() < 2:
    raise ValueError(
        "One churn class has fewer than two examples. "
        "There is insufficient data for a stratified train/test split."
    )


# ============================================================
# 3. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test, ids_train, ids_test = (
    train_test_split(
        X,
        y,
        df["Customer_ID"],
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {
    "Dummy_Baseline": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="prior")),
    ]),
    "Random_Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
}


# ============================================================
# 5. TRAIN AND EVALUATE
# ============================================================

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    result = {
        "Model_Name": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": (
            roc_auc_score(y_test, probabilities)
            if y_test.nunique() == 2 else float("nan")
        ),
    }

    results.append(result)
    trained_models[name] = model

    print(f"\n--- {name} ---")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


evaluation = pd.DataFrame(results)
evaluation.to_csv(EVALUATION_FILE, index=False)

print("\nModel comparison:")
print(evaluation.to_string(index=False))


# ============================================================
# 6. SELECT MODEL
# ============================================================

# Select using F1 score, with ROC-AUC as a secondary criterion.
ranked = evaluation.sort_values(
    ["F1_Score", "ROC_AUC"],
    ascending=False,
    na_position="last",
)

best_name = ranked.iloc[0]["Model_Name"]
best_model = trained_models[best_name]

print(f"\nSelected model: {best_name}")


# ============================================================
# 7. SAVE MODEL AND TEST PREDICTIONS
# ============================================================

artifact = {
    "model_name": best_name,
    "model": best_model,
    "feature_columns": features,
    "target_column": target,
    "churn_horizon_months": 24,
    "churn_definition": "Zero orders in the next 24 months",
}

joblib.dump(artifact, MODEL_FILE)
joblib.dump(features, FEATURES_FILE)

test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = best_model.predict(X_test)

test_output = pd.DataFrame({
    "Customer_ID": ids_test.values,
    "Actual_Churn_24_Months": y_test.values,
    "Predicted_Churn_24_Months": test_predictions,
    "Churn_Probability": test_probabilities,
})

test_output["Risk_Level"] = pd.cut(
    test_output["Churn_Probability"],
    bins=[-0.001, 0.40, 0.70, 1.001],
    labels=["Low", "Medium", "High"],
)

test_output.to_csv(PREDICTIONS_FILE, index=False)

print("\nFiles saved:")
print(MODEL_FILE)
print(FEATURES_FILE)
print(EVALUATION_FILE)
print(PREDICTIONS_FILE)

print("\nTraining completed.")
print(
    "Note: This random split evaluates one historical cohort; "
    "it does not establish reliable real-world 24-month performance. "
    "Use time-based validation before deployment."
)
