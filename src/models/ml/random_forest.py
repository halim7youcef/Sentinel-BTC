import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DATA_PATH = "data/processed/btc_features_ml.csv"

MODEL_DIR = "models/ml"
METRICS_DIR = "results/metrics"
PRED_DIR = "results/predictions"
CURVES_DIR = "results/curves"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CURVES_DIR, exist_ok=True)

print("Loading dataset for SOTA Random Forest...")
df = pd.read_csv(DATA_PATH)
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

exclude = [
    "timestamp",
    "close",
    "target",
    "future_return"
]

features = [c for c in df.columns if c not in exclude]

X = df[features]
y = df["target"]

# -----------------------------
# Robust Purged Time-Series Split
# -----------------------------
split = int(len(df) * 0.8)
gap = 6  # the target looks 6 periods ahead, we must gap train and test to avoid leakage

X_train = X.iloc[:split]
y_train = y.iloc[:split]

X_test = X.iloc[split+gap:]
y_test = y.iloc[split+gap:]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

print("\nTraining SOTA RandomForest...")
# Random Forests on noisy trading data suffer easily; we use robust regularization parameters
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=9,
    min_samples_leaf=40,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print("\nSOTA RF Results")
print("----------------")
print("Accuracy:", accuracy)
print("ROC AUC:", auc)

metrics = {
    "accuracy": float(accuracy),
    "roc_auc": float(auc),
    "samples_train": int(len(X_train)),
    "samples_test": int(len(X_test)),
    "features": int(len(features))
}

with open(f"{METRICS_DIR}/random_forest_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": preds,
    "prob_up": probs
})
pred_df.to_csv(f"{PRED_DIR}/random_forest_predictions.csv", index=False)

importances = model.feature_importances_
imp_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8,6))
plt.barh(imp_df["feature"][:20], imp_df["importance"][:20])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importance (SOTA RF)")
plt.tight_layout()

plt.savefig(f"{CURVES_DIR}/random_forest_feature_importance.png")

model_path = f"{MODEL_DIR}/random_forest.pkl"
joblib.dump(model, model_path)

print("\nSaved SOTA Random Forest outputs.")