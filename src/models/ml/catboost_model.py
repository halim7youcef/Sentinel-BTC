import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = "data/processed/btc_features_ml.csv"

MODEL_DIR = "models/ml"
METRICS_DIR = "results/metrics"
PRED_DIR = "results/predictions"
CURVES_DIR = "results/curves"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CURVES_DIR, exist_ok=True)

print("Loading dataset for SOTA CatBoost...")
df = pd.read_csv(DATA_PATH)
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

exclude = [
    "timestamp", "close", "target", "future_return"
]
features = [c for c in df.columns if c not in exclude]

X = df[features]
y = df["target"]

print("Total samples:", len(df))
print("Features:", len(features))

split = int(len(df) * 0.8)
gap = 6 # purged valid cutoff

X_train = X.iloc[:split]
y_train = y.iloc[:split]

X_test = X.iloc[split+gap:]
y_test = y.iloc[split+gap:]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# Creating CatBoost Pools
train_pool = Pool(X_train, y_train)
valid_pool = Pool(X_test, y_test)

# SOTA parameters for CatBoost specifically tuned for non-stationary numeric financial grids
params = {
    "iterations": 1000,
    "learning_rate": 0.01,
    "depth": 6,
    "eval_metric": 'AUC',
    "random_seed": 42,
    "auto_class_weights": 'Balanced',
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "use_best_model": True,
    "task_type": "CPU", # Adjust to GPU if available and configured
    "verbose": 50
}

print("\nTraining SOTA CatBoost...")
model = CatBoostClassifier(**params)
model.fit(
    train_pool,
    eval_set=valid_pool,
    early_stopping_rounds=150
)

print("\nGenerating predictions...")
prob_up = model.predict_proba(X_test)[:, 1]
preds = (prob_up > 0.5).astype(int)

accuracy = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, prob_up)

print("\nModel Performance")
print("-------------------")
print("Accuracy:", accuracy)
print("ROC AUC:", auc)

metrics = {
    "accuracy": float(accuracy),
    "roc_auc": float(auc),
    "samples_train": int(len(X_train)),
    "samples_test": int(len(X_test)),
    "features": int(len(features))
}

with open(f"{METRICS_DIR}/catboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": preds,
    "prob_up": prob_up
})
pred_df.to_csv(f"{PRED_DIR}/catboost_predictions.csv", index=False)

# Feature Importance
importances = model.get_feature_importance(type='FeatureImportance')
imp_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8,6))
plt.barh(imp_df["feature"][:20], imp_df["importance"][:20])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importance (CatBoost)")
plt.tight_layout()
plt.savefig(f"{CURVES_DIR}/catboost_feature_importance.png")

model_path = f"{MODEL_DIR}/catboost_model.cbm"
model.save_model(model_path)
print("\nSaved SOTA CatBoost outputs.")
