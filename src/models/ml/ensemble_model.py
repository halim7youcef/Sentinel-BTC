import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import json

PRED_DIR = "results/predictions"
METRICS_DIR = "results/metrics"

os.makedirs(METRICS_DIR, exist_ok=True)

print("Loading individual predictions for Ensemble...")

# Get all prediction files excluding the ensemble itself if it exists
prediction_files = [f for f in glob.glob(f"{PRED_DIR}/*_predictions.csv") if "ensemble" not in f.lower()]

if not prediction_files:
    raise FileNotFoundError("No prediction files found to ensemble!")

ensemble_df = None
all_probs = []

for file in prediction_files:
    model_name = os.path.basename(file).replace('_predictions.csv', '')
    print(f"Loading {model_name}...")
    
    df = pd.read_csv(file)
    all_probs.append(df["prob_up"].values)
    
    if ensemble_df is None:
        # Initialize the base dataframe with the true labels
        ensemble_df = pd.DataFrame({"y_true": df["y_true"].values})

# Average the probabilities (Soft Voting Ensemble)
mean_prob = np.mean(all_probs, axis=0)

ensemble_df["prob_up"] = mean_prob
ensemble_df["y_pred"] = (mean_prob > 0.5).astype(int)

# Extract y_true
y_test = ensemble_df["y_true"]

# Calculate metrics
accuracy = accuracy_score(y_test, ensemble_df["y_pred"])
auc = roc_auc_score(y_test, mean_prob)

print("\nEnsemble Model Performance (Soft Voting Average)")
print("-------------------")
print("Accuracy:", accuracy)
print("ROC AUC:", auc)

metrics = {
    "accuracy": float(accuracy),
    "roc_auc": float(auc),
    "models_ensembled": len(prediction_files),
    "model_list": [os.path.basename(f).replace('_predictions.csv', '') for f in prediction_files]
}

with open(f"{METRICS_DIR}/ensemble_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save the ensemble predictions
ensemble_df.to_csv(f"{PRED_DIR}/ensemble_predictions.csv", index=False)
print(f"\nSaved SOTA Ensemble outputs to {PRED_DIR}/ensemble_predictions.csv.")
