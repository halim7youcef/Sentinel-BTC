import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
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

print("Loading dataset for SOTA XGBoost...")
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

# SOTA parameters for XGBoost treating financial data
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "alpha": 2.0,       # L1 regularization
    "lambda": 2.0,      # L2 regularization
    "tree_method": "hist",
    "seed": 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

print("\nTraining SOTA XGBoost...")
evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=150,
    verbose_eval=50,
    evals_result=evals_result
)

print("\nGenerating predictions...")
prob_up = model.predict(dvalid)
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

with open(f"{METRICS_DIR}/xgboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pred_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": preds,
    "prob_up": prob_up
})
pred_df.to_csv(f"{PRED_DIR}/xgboost_predictions.csv", index=False)

# XGBoost Feature Importance
importance_dict = model.get_score(importance_type='gain')
imp_df = pd.DataFrame({
    "feature": list(importance_dict.keys()),
    "importance": list(importance_dict.values())
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8,6))
plt.barh(imp_df["feature"][:20], imp_df["importance"][:20])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importance (XGBoost Gain)")
plt.tight_layout()
plt.savefig(f"{CURVES_DIR}/xgboost_feature_importance.png")

model_path = f"{MODEL_DIR}/xgboost_model.json"
model.save_model(model_path)
print("\nSaved SOTA XGBoost outputs.")
