import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
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

print("Loading dataset for SOTA LightGBM...")
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

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# SOTA parameters for low SNR financial data
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 2.0,
    "lambda_l2": 2.0,
    "is_unbalance": True,
    "verbosity": -1,
    "seed": 42
}

print("\nTraining SOTA LightGBM...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(50)]
)

print("\nGenerating predictions...")
prob_up = model.predict(X_test)
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

with open(f"{METRICS_DIR}/lightgbm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": preds,
    "prob_up": prob_up
})
pred_df.to_csv(f"{PRED_DIR}/lightgbm_predictions.csv", index=False)

importances = model.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8,6))
plt.barh(imp_df["feature"][:20], imp_df["importance"][:20])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importance (LightGBM Gain)")
plt.tight_layout()
plt.savefig(f"{CURVES_DIR}/lightgbm_feature_importance.png")

model_path = f"{MODEL_DIR}/lightgbm_model.pkl"
joblib.dump(model, model_path)
print("\nSaved SOTA LightGBM outputs.")