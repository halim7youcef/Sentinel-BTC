"""
build_rl_features.py
====================
Builds the complete, self-contained RL observation dataset.

Unlike the ML pipeline which stores predictions in a separate CSV, this script:
  1. Builds all SOTA market features (same as build_ml_features.py)
  2. Trains or loads all 4 ML models (XGBoost, LightGBM, CatBoost, RF) inline
  3. Runs each model to produce out-of-sample probability predictions
  4. Calibrates each model's probabilities (isotonic regression) before ensembling
  5. Bakes the calibrated ensemble probability DIRECTLY into the observation feature table

Output: data/processed/btc_features_rl.csv
  - All 38+ market features
  - ml_prob_up (calibrated ensemble probability)
  - gt_future_return (next candle's actual return — used by poly_env.py as ground truth)
  - close (used by poly_env.py for price tracking)

Changes from v1
---------------
  FIX 1 — Scaler leakage: RobustScaler was fitted on the full dataset (train+test).
           It is now fitted exclusively on X_train, preventing test-set statistics
           from leaking into the feature space used for evaluation.

  FIX 2 — Probability calibration: Raw model probabilities were uncalibrated,
           causing a disconnect where ensemble_prob ≈ 0.50 while downstream
           confidence values read 0.83+. Each model's probabilities are now
           calibrated via isotonic regression on a held-out calibration fold
           before the soft-vote ensemble is computed.

  FIX 3 — gt_future_return alignment: Previously used pct_change().shift(-1)
           (1-candle lookahead) while labels were built on FUTURE_PERIODS=6
           lookahead. gt_future_return now consistently uses FUTURE_PERIODS
           to match the resolution horizon in poly_env.py.

  FIX 4 — Train/cal/test split: A calibration fold (10% of data) is carved
           out between train and test with its own purge gap, so calibration
           does not bleed into the test evaluation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
RAW_PATH        = "data/raw/btc_5m.csv"
OUT_ML          = "data/processed/btc_features_ml.csv"
OUT_RL          = "data/processed/btc_features_rl.csv"
SCALER_PATH     = "models/ml/robust_scaler.pkl"
MODEL_DIR       = "models/ml"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
LABEL_THRESHOLD = 0.0015   # 0.15% — overcomes exchange fees
FUTURE_PERIODS  = 6        # 30-minute lookahead target
TRAIN_RATIO     = 0.70     # FIX 4: 70% train / 10% calibration / 20% test
CAL_RATIO       = 0.80     # calibration ends at 80% mark
GAP             = 6        # purge gap between each split boundary

# -------------------------------------------------------------------
# Feature helpers
# -------------------------------------------------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12     = ema(series, 12)
    ema26     = ema(series, 26)
    macd_line = ema12 - ema26
    signal    = ema(macd_line, 9)
    return macd_line, signal, macd_line - signal

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calibrate_probs(raw_probs_train, y_train_cal, raw_probs_test):
    """
    Isotonic regression calibration on a held-out calibration fold.
    Maps raw model probabilities to well-calibrated [0,1] values so that
    a prediction of 0.80 means the event occurs ~80% of the time.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs_train, y_train_cal)
    return iso.transform(raw_probs_test), iso


# ===================================================================
# STEP 1 — Load raw data and build SOTA features
# ===================================================================
print("=" * 60)
print("STEP 1: Building SOTA Feature Set")
print("=" * 60)

df = pd.read_csv(RAW_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# --- Log returns ---
df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
for h in [3, 5, 10, 20, 50, 100]:
    df[f"log_ret_{h}"] = np.log(df["close"] / df["close"].shift(h))
df["acceleration"] = df["log_ret_1"] - df["log_ret_3"]

# --- Target ---
df["future_return"] = df["close"].shift(-FUTURE_PERIODS) / df["close"] - 1
df["target_raw"] = 0
df.loc[df["future_return"] >  LABEL_THRESHOLD, "target_raw"] =  1
df.loc[df["future_return"] < -LABEL_THRESHOLD, "target_raw"] = -1

# --- Microstructure ---
df["hl_range"]         = (df["high"] - df["low"]) / df["close"]
df["body_size"]        = (df["close"] - df["open"]) / df["close"]
df["micro_clv"]        = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-8)
df["garman_klass_vol"] = np.sqrt(
    0.5 * np.log(df["high"] / df["low"])**2
    - (2*np.log(2) - 1) * np.log(df["close"] / df["open"])**2
)

# --- Rolling z-scores ---
for w in [10, 20, 50, 200]:
    sma  = df["close"].rolling(w).mean()
    std  = df["close"].rolling(w).std()
    df[f"zscore_close_{w}"]  = (df["close"]  - sma) / (std + 1e-8)
    vsma = df["volume"].rolling(w).mean()
    vstd = df["volume"].rolling(w).std()
    df[f"zscore_volume_{w}"] = (df["volume"] - vsma) / (vstd + 1e-8)

# --- Momentum ---
df["rsi14"] = rsi(df["close"], 14)
_, _, hist  = macd(df["close"])
df["macd_hist"]   = hist
df["macd_zscore"] = (hist - hist.rolling(20).mean()) / (hist.rolling(20).std() + 1e-8)
bb = df["close"].rolling(20)
df["bb_width"] = (bb.mean() + 2*bb.std() - (bb.mean() - 2*bb.std())) / (bb.mean() + 1e-8)
df["bb_pos"]   = (df["close"] - (bb.mean() - 2*bb.std()))             / (4*bb.std() + 1e-8)

# --- Volatility ---
for w in [10, 20, 50]:
    df[f"realized_vol_{w}"] = df["log_ret_1"].rolling(w).std() * np.sqrt(288)
df["atr14"]     = atr(df["high"], df["low"], df["close"])
df["atr_ratio"] = df["atr14"] / df["close"]

# --- Cyclical time ---
df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour      / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour      / 24)
df["day_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek /  7)
df["day_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek /  7)

# --- Rolling moments ---
for w in [20, 50]:
    df[f"skew_{w}"] = df["log_ret_1"].rolling(w).skew()
    df[f"kurt_{w}"] = df["log_ret_1"].rolling(w).kurt()

# FIX 3: gt_future_return now uses FUTURE_PERIODS lookahead to match the
# label construction horizon. The previous pct_change().shift(-1) only looked
# 1 candle ahead while labels were built on a 6-candle (30m) horizon —
# poly_env.py was resolving trades against the wrong price point.
df["gt_future_return"] = df["close"].pct_change(FUTURE_PERIODS).shift(-FUTURE_PERIODS)

# --- Clean ---
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# --- ML dataset (filter indeterminate 0-class rows) ---
df_ml = df[df["target_raw"] != 0].copy()
df_ml["target"] = (df_ml["target_raw"] == 1).astype(int)

EXCLUDE_FROM_FEATURES = [
    "timestamp", "open", "high", "low", "close", "volume",
    "future_return", "target_raw", "target", "gt_future_return"
]
feature_cols = [c for c in df_ml.columns if c not in EXCLUDE_FROM_FEATURES]

X = df_ml[feature_cols]
y = df_ml["target"]

print(f"Dataset: {len(df_ml)} rows | {len(feature_cols)} features")

# --- Train / calibration / test split (time-ordered, purge gaps applied) ---
n          = len(df_ml)
train_end  = int(n * TRAIN_RATIO)
cal_end    = int(n * CAL_RATIO)

X_train, y_train = X.iloc[:train_end],                   y.iloc[:train_end]
X_cal,   y_cal   = X.iloc[train_end + GAP : cal_end],    y.iloc[train_end + GAP : cal_end]
X_test,  y_test  = X.iloc[cal_end + GAP:],               y.iloc[cal_end + GAP:]

print(f"Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

# FIX 1: Scaler fitted on X_train ONLY.
# Previously fitted on all of X (train+cal+test), leaking test-set statistics
# into the normalisation applied during RL evaluation.
scaler = RobustScaler()
scaler.fit(X_train.values)
joblib.dump(scaler, SCALER_PATH)
print(f"Saved RobustScaler (fitted on train only): {SCALER_PATH}")

# Save ML checkpoint
ml_save_cols = ["timestamp", "close"] + feature_cols + ["target", "future_return"]
df_ml[ml_save_cols].to_csv(OUT_ML, index=False)
print(f"Saved ML checkpoint: {OUT_ML}")


# ===================================================================
# STEP 2 — Train models, calibrate on cal fold, predict on test fold
# ===================================================================
print("\n" + "=" * 60)
print("STEP 2: Training + Calibrating ML Models")
print("=" * 60)

all_probs      = {}   # calibrated test probabilities
cal_models     = {}   # saved isotonic regressors

# ---------------------- XGBoost ----------------------
xgb_path = f"{MODEL_DIR}/xgboost_rl.json"
dtrain   = xgb.DMatrix(X_train, label=y_train)
dcal     = xgb.DMatrix(X_cal,   label=y_cal)
dtest    = xgb.DMatrix(X_test,  label=y_test)
xgb_params = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.01, "max_depth": 6,
    "subsample": 0.8, "colsample_bytree": 0.5,
    "alpha": 2.0, "lambda": 2.0,
    "seed": 42, "tree_method": "hist"
}
print("  Training XGBoost...")
xgb_model = xgb.train(
    xgb_params, dtrain,
    num_boost_round=1000,
    evals=[(dcal, "val")],
    callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
    verbose_eval=False
)
xgb_model.save_model(xgb_path)
raw_cal  = xgb_model.predict(dcal)
raw_test = xgb_model.predict(dtest)
cal_probs, cal_models["xgboost"] = calibrate_probs(raw_cal, y_cal, raw_test)
all_probs["xgboost"] = cal_probs
print(f"    XGBoost raw range:  [{raw_test.min():.3f}, {raw_test.max():.3f}]")
print(f"    XGBoost cal range:  [{cal_probs.min():.3f}, {cal_probs.max():.3f}]")

# ---------------------- LightGBM ----------------------
lgb_path = f"{MODEL_DIR}/lightgbm_rl.txt"
ltrain   = lgb.Dataset(X_train, label=y_train)
lcal     = lgb.Dataset(X_cal,   label=y_cal,  reference=ltrain)
lgb_params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.01, "num_leaves": 31, "max_depth": 6,
    "lambda_l1": 2.0, "lambda_l2": 2.0,
    "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 5,
    "is_unbalance": True, "verbose": -1, "seed": 42
}
print("  Training LightGBM...")
lgb_model = lgb.train(
    lgb_params, ltrain,
    num_boost_round=1000,
    valid_sets=[lcal],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
)
lgb_model.save_model(lgb_path)
raw_cal  = lgb_model.predict(X_cal)
raw_test = lgb_model.predict(X_test)
cal_probs, cal_models["lightgbm"] = calibrate_probs(raw_cal, y_cal, raw_test)
all_probs["lightgbm"] = cal_probs
print(f"    LightGBM raw range: [{raw_test.min():.3f}, {raw_test.max():.3f}]")
print(f"    LightGBM cal range: [{cal_probs.min():.3f}, {cal_probs.max():.3f}]")

# ---------------------- CatBoost ----------------------
cb_path  = f"{MODEL_DIR}/catboost_rl.cbm"
cb_train = Pool(X_train, y_train)
cb_cal   = Pool(X_cal,   y_cal)
cb_test  = Pool(X_test,  y_test)
cb_params = {
    "iterations": 1000, "learning_rate": 0.01, "depth": 6,
    "eval_metric": "AUC", "random_seed": 42,
    "auto_class_weights": "Balanced", "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bernoulli", "subsample": 0.8,
    "use_best_model": True, "task_type": "CPU", "verbose": 0
}
print("  Training CatBoost...")
cb_model = CatBoostClassifier(**cb_params)
cb_model.fit(cb_train, eval_set=cb_cal, early_stopping_rounds=150, verbose=False)
cb_model.save_model(cb_path)
raw_cal  = cb_model.predict_proba(X_cal)[:, 1]
raw_test = cb_model.predict_proba(X_test)[:, 1]
cal_probs, cal_models["catboost"] = calibrate_probs(raw_cal, y_cal, raw_test)
all_probs["catboost"] = cal_probs
print(f"    CatBoost raw range: [{raw_test.min():.3f}, {raw_test.max():.3f}]")
print(f"    CatBoost cal range: [{cal_probs.min():.3f}, {cal_probs.max():.3f}]")

# ---------------------- Random Forest ----------------------
rf_path = f"{MODEL_DIR}/rf_rl.pkl"
print("  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=9, min_samples_leaf=40,
    max_features="sqrt", class_weight="balanced_subsample",
    n_jobs=-1, random_state=42
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, rf_path)
raw_cal  = rf_model.predict_proba(X_cal)[:, 1]
raw_test = rf_model.predict_proba(X_test)[:, 1]
cal_probs, cal_models["random_forest"] = calibrate_probs(raw_cal, y_cal, raw_test)
all_probs["random_forest"] = cal_probs
print(f"    RF raw range:       [{raw_test.min():.3f}, {raw_test.max():.3f}]")
print(f"    RF cal range:       [{cal_probs.min():.3f}, {cal_probs.max():.3f}]")

# Save calibration models for live_inference.py
joblib.dump(cal_models, f"{MODEL_DIR}/isotonic_calibrators.pkl")
print(f"\n  Saved isotonic calibrators: {MODEL_DIR}/isotonic_calibrators.pkl")


# ===================================================================
# STEP 3 — Calibrated Soft-Vote Ensemble
# ===================================================================
print("\n" + "=" * 60)
print("STEP 3: Computing Calibrated Ensemble Probability")
print("=" * 60)

ensemble_prob = np.mean(list(all_probs.values()), axis=0)
print(f"  Ensemble prob range: [{ensemble_prob.min():.4f}, {ensemble_prob.max():.4f}]")
print(f"  Ensemble prob mean:  {ensemble_prob.mean():.4f}  (expect ~0.50 if balanced)")


# ===================================================================
# STEP 4 — Build the final RL observation dataframe
# ===================================================================
print("\n" + "=" * 60)
print("STEP 4: Building RL Observation Table")
print("=" * 60)

df_test_rows = df_ml.iloc[cal_end + GAP:].copy().reset_index(drop=True)
df_test_rows["ml_prob_up"] = ensemble_prob

rl_cols = ["timestamp", "close", "gt_future_return", "ml_prob_up"] + feature_cols
df_rl   = df_test_rows[rl_cols].copy()
df_rl   = df_rl.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

df_rl.to_csv(OUT_RL, index=False)
print(f"Saved RL dataset: {OUT_RL}")
print(f"  Rows:     {len(df_rl)}")
print(f"  Columns:  {len(df_rl.columns)}")
print(f"  Includes: ml_prob_up (calibrated), gt_future_return ({FUTURE_PERIODS}-candle), "
      f"close, {len(feature_cols)} market features")
print("\nRL feature dataset ready. Use data/processed/btc_features_rl.csv in poly_env.py")