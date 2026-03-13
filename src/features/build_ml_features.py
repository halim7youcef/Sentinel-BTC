import os
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/btc_5m.csv"
OUT_FULL = "data/processed/btc_features_full.csv"
OUT_ML = "data/processed/btc_features_ml.csv"

os.makedirs("data/processed", exist_ok=True)

# Increased threshold: 0.15% return. Overcomes 0.04% fee easily.
LABEL_THRESHOLD = 0.0015

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100/(1+rs))

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high-low,
        (high-prev_close).abs(),
        (low-prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

print("Loading raw data for SOTA Pipeline...")
df = pd.read_csv(RAW_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ------------------------------------------------
# Advanced Log Returns & Target
# ------------------------------------------------
df["log_ret_1"] = np.log(df["close"]/df["close"].shift(1))
for h in [3, 5, 10, 20, 50, 100]:
    df[f"log_ret_{h}"] = np.log(df["close"]/df["close"].shift(h))

df["acceleration"] = df["log_ret_1"] - df["log_ret_3"]

# Target: predict return over the NEXT 6 periods (30 mins)
FUTURE_PERIODS = 6 
df["future_return"] = df["close"].shift(-FUTURE_PERIODS) / df["close"] - 1

df["target_raw"] = 0
df.loc[df["future_return"] > LABEL_THRESHOLD, "target_raw"] = 1
df.loc[df["future_return"] < -LABEL_THRESHOLD, "target_raw"] = -1

# ------------------------------------------------
# Microstructure & Candle Features
# ------------------------------------------------
df["hl_range"] = (df["high"] - df["low"]) / df["close"]
df["body_size"] = (df["close"] - df["open"]) / df["close"]
df["micro_clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-8)
df["garman_klass_vol"] = np.sqrt(0.5 * np.log(df["high"]/df["low"])**2 - (2*np.log(2)-1) * np.log(df["close"]/df["open"])**2)

# ------------------------------------------------
# Moving Averages & Rolling Z-Scores
# ------------------------------------------------
for w in [10, 20, 50, 200]:
    # Price z-score
    sma = df["close"].rolling(w).mean()
    std = df["close"].rolling(w).std()
    df[f"zscore_close_{w}"] = (df["close"] - sma) / (std + 1e-8)
    
    # Volume z-score
    vsma = df["volume"].rolling(w).mean()
    vstd = df["volume"].rolling(w).std()
    df[f"zscore_volume_{w}"] = (df["volume"] - vsma) / (vstd + 1e-8)

# ------------------------------------------------
# Momentum & Oscillators
# ------------------------------------------------
df["rsi14"] = rsi(df["close"], 14)
macd_line, signal, hist = macd(df["close"])
df["macd_hist"] = hist
df["macd_zscore"] = (hist - hist.rolling(20).mean()) / (hist.rolling(20).std() + 1e-8)

# Bollinger Bands width and positioning 
bb = df["close"].rolling(20)
df["bb_width"] = (bb.mean() + 2*bb.std() - (bb.mean() - 2*bb.std())) / bb.mean()
df["bb_pos"] = (df["close"] - (bb.mean() - 2*bb.std())) / (4*bb.std() + 1e-8)

# ------------------------------------------------
# Volatility Expansion & Smoothing
# ------------------------------------------------
for w in [10, 20, 50]:
    df[f"realized_vol_{w}"] = df["log_ret_1"].rolling(w).std() * np.sqrt(288)

df["atr14"] = atr(df["high"], df["low"], df["close"])
df["atr_ratio"] = df["atr14"] / df["close"]

# ------------------------------------------------
# Time encodings (cyclical)
# ------------------------------------------------
df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
df["day_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
df["day_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)

# ------------------------------------------------
# Statistics (Rolling moments)
# ------------------------------------------------
for w in [20, 50]:
    df[f"skew_{w}"] = df["log_ret_1"].rolling(w).skew()
    df[f"kurt_{w}"] = df["log_ret_1"].rolling(w).kurt()

# ------------------------------------------------
# Clean
# ------------------------------------------------
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

df.to_csv(OUT_FULL, index=False)
print("Saved full dataset!")

# ------------------------------------------------
# ML dataset (Filter indeterminate classes)
# ------------------------------------------------
df_ml = df[df["target_raw"] != 0].copy()
df_ml["target"] = (df_ml["target_raw"] == 1).astype(int)

exclude = [
    "timestamp", "open", "high", "low", "close", "volume",
    "future_return", "target_raw", "target"
]
features = [c for c in df_ml.columns if c not in exclude]
df_ml = df_ml[["timestamp", "close"] + features + ["target", "future_return"]]

df_ml.to_csv(OUT_ML, index=False)
print("Saved ML dataset")
print("Rows:", len(df_ml))
print("Features:", len(features))