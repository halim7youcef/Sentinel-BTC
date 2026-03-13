"""
app.py  –  FastAPI backend for the BTC ML Live Trading Dashboard
================================================================
Exposes:
  GET  /api/price          → latest BTC price snapshot
  GET  /api/signal         → run ML ensemble inference, return signal
  GET  /api/history        → last N trade log entries
  GET  /api/candles        → last N OHLCV candles for charting
  WS   /ws/live            → WebSocket push: price + signal every 5 s
"""

import os
import sys
import time
import asyncio
import datetime
import warnings
import traceback
import json

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from binance.client import Client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ── suppress noisy sklearn / numpy warnings ─────────────────────────
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────
# The server is always launched from btc_rl_project/ (see start.bat / run.py)
# so os.getcwd() == btc_rl_project/   ← most robust anchor
CWD         = os.getcwd()                                               # btc_rl_project/
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))               # dashboard/backend/
DASH_DIR    = os.path.dirname(BACKEND_DIR)                             # dashboard/
PROJECT_DIR = CWD                                                       # btc_rl_project/

# ── env & Binance ────────────────────────────────────────────────────
load_dotenv(os.path.join(PROJECT_DIR, ".env"))    # btc_rl_project/.env
load_dotenv(os.path.join(DASH_DIR, ".env"))       # dashboard/.env
load_dotenv()                                      # cwd fallback

API_KEY    = os.getenv("Binance_API") or os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("Binance_Secret") or os.getenv("BINANCE_API_SECRET")

# ── model paths (all relative to btc_rl_project/) ───────────────────
ML_DIR           = os.path.join(PROJECT_DIR, "models", "ml")
SCALER_PATH      = os.path.join(ML_DIR, "robust_scaler.pkl")
CALIBRATORS_PATH = os.path.join(ML_DIR, "isotonic_calibrators.pkl")
FEATURES_CSV     = os.path.join(PROJECT_DIR, "data", "processed", "btc_features_ml.csv")
LOG_PATH         = os.path.join(PROJECT_DIR, "results", "dashboard_trades.csv")

# ── Load canonical feature column order from training data ───────────
_EXCLUDE_COLS = {"timestamp", "open", "high", "low", "close", "volume",
                 "future_return", "target_raw", "target", "open_time"}
try:
    _hdr = pd.read_csv(FEATURES_CSV, nrows=0)
    CANONICAL_FEATURES = [c for c in _hdr.columns if c not in _EXCLUDE_COLS]
    print(f"[Config] Loaded {len(CANONICAL_FEATURES)} canonical feature columns from training CSV.")
except Exception as _e:
    CANONICAL_FEATURES = None
    print(f"[Config] WARNING: Could not read features CSV ({_e}). Will infer columns at runtime.")

SYMBOL   = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_5MINUTE

# Kelly thresholds
MIN_EDGE   = 0.55
MAX_STAKE  = 100.0
KELLY_FRAC = 0.25

# ────────────────────────────────────────────────────────────────────
# Feature engineering  (mirrors live_inference.py exactly)
# ────────────────────────────────────────────────────────────────────
def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def _atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def extract_features(df_raw: pd.DataFrame):
    df = df_raw.copy().sort_values("timestamp").reset_index(drop=True)

    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    for h in [3, 5, 10, 20, 50, 100]:
        df[f"log_ret_{h}"] = np.log(df["close"] / df["close"].shift(h))
    df["acceleration"] = df["log_ret_1"] - df["log_ret_3"]

    df["hl_range"]         = (df["high"] - df["low"]) / df["close"]
    df["body_size"]        = (df["close"] - df["open"]) / df["close"]
    df["micro_clv"]        = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-8)
    df["garman_klass_vol"] = np.sqrt(
        0.5 * np.log(df["high"] / df["low"])**2
        - (2*np.log(2) - 1) * np.log(df["close"] / df["open"])**2
    )

    for w in [10, 20, 50, 200]:
        df[f"zscore_close_{w}"]  = (df["close"]  - df["close"].rolling(w).mean())  / (df["close"].rolling(w).std()  + 1e-8)
        df[f"zscore_volume_{w}"] = (df["volume"] - df["volume"].rolling(w).mean()) / (df["volume"].rolling(w).std() + 1e-8)

    df["rsi14"]       = _rsi(df["close"], 14)
    e12               = _ema(df["close"], 12)
    e26               = _ema(df["close"], 26)
    macd_line         = e12 - e26
    df["macd_hist"]   = macd_line - _ema(macd_line, 9)
    df["macd_zscore"] = (df["macd_hist"] - df["macd_hist"].rolling(20).mean()) / (df["macd_hist"].rolling(20).std() + 1e-8)

    bb = df["close"].rolling(20)
    df["bb_width"] = (bb.mean() + 2*bb.std() - (bb.mean() - 2*bb.std())) / (bb.mean() + 1e-8)
    df["bb_pos"]   = (df["close"] - (bb.mean() - 2*bb.std()))             / (4*bb.std() + 1e-8)

    for w in [10, 20, 50]:
        df[f"realized_vol_{w}"] = df["log_ret_1"].rolling(w).std() * np.sqrt(288)
    df["atr14"]     = _atr(df["high"], df["low"], df["close"])
    df["atr_ratio"] = df["atr14"] / df["close"]

    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour      / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour      / 24)
    df["day_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek /  7)
    df["day_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek /  7)

    for w in [20, 50]:
        df[f"skew_{w}"] = df["log_ret_1"].rolling(w).skew()
        df[f"kurt_{w}"] = df["log_ret_1"].rolling(w).kurt()

    EXCLUDE       = ["timestamp", "open", "high", "low", "close", "volume", "open_time"]
    feature_names = [c for c in df.columns if c not in EXCLUDE]
    return df, feature_names


# ────────────────────────────────────────────────────────────────────
# ML Engine  (only ML, no PPO/RL)
# ────────────────────────────────────────────────────────────────────
class MLEngine:
    def __init__(self):
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier

        print("[MLEngine] Loading XGBoost...")
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(os.path.join(ML_DIR, "xgboost_rl.json"))

        print("[MLEngine] Loading LightGBM...")
        self.lgb_model = lgb.Booster(model_file=os.path.join(ML_DIR, "lightgbm_rl.txt"))

        print("[MLEngine] Loading CatBoost...")
        self.cb_model = CatBoostClassifier()
        self.cb_model.load_model(os.path.join(ML_DIR, "catboost_rl.cbm"))

        print("[MLEngine] Loading Random Forest...")
        self.rf_model = joblib.load(os.path.join(ML_DIR, "rf_rl.pkl"))

        print("[MLEngine] Loading scaler...")
        self.scaler = joblib.load(SCALER_PATH)

        if os.path.exists(CALIBRATORS_PATH):
            self.calibrators = joblib.load(CALIBRATORS_PATH)
            print("[MLEngine] Isotonic calibrators loaded.")
        else:
            self.calibrators = None
            print("[MLEngine] WARNING: No calibrators found — using raw probabilities.")

        print("[MLEngine] All models ready.")

    def predict_prob(self, X: pd.DataFrame) -> float:
        """Return calibrated ensemble P(next close > current close)."""
        import xgboost as xgb
        raw_xgb = float(self.xgb_model.predict(xgb.DMatrix(X))[0])
        raw_lgb = float(self.lgb_model.predict(X)[0])
        raw_cb  = float(self.cb_model.predict_proba(X)[0, 1])
        raw_rf  = float(self.rf_model.predict_proba(X)[0, 1])

        if self.calibrators:
            raw_xgb = float(self.calibrators["xgboost"].transform([raw_xgb])[0])
            raw_lgb = float(self.calibrators["lightgbm"].transform([raw_lgb])[0])
            raw_cb  = float(self.calibrators["catboost"].transform([raw_cb])[0])
            raw_rf  = float(self.calibrators["random_forest"].transform([raw_rf])[0])

        return float(np.mean([raw_xgb, raw_lgb, raw_cb, raw_rf]))

    def individual_probs(self, X: pd.DataFrame) -> dict:
        """Return per-model raw (uncalibrated) probabilities for display."""
        import xgboost as xgb
        raw_xgb = float(self.xgb_model.predict(xgb.DMatrix(X))[0])
        raw_lgb = float(self.lgb_model.predict(X)[0])
        raw_cb  = float(self.cb_model.predict_proba(X)[0, 1])
        raw_rf  = float(self.rf_model.predict_proba(X)[0, 1])
        return {"xgboost": raw_xgb, "lightgbm": raw_lgb, "catboost": raw_cb, "random_forest": raw_rf}


# ────────────────────────────────────────────────────────────────────
# Binance Data Fetcher
# ────────────────────────────────────────────────────────────────────
class BinanceFetcher:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET)
        print("[BinanceFetcher] Connected to Binance.")

    def fetch_candles(self, limit: int = 300) -> pd.DataFrame:
        klines  = self.client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=limit)
        columns = ["open_time","open","high","low","close","volume",
                   "ct","qv","nt","tb","tq","i"]
        df = pd.DataFrame(klines, columns=columns)[["open_time","open","high","low","close","volume"]]
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df

    def fetch_price(self) -> float:
        ticker = self.client.get_symbol_ticker(symbol=SYMBOL)
        return float(ticker["price"])


# ────────────────────────────────────────────────────────────────────
# Trade Logger
# ────────────────────────────────────────────────────────────────────
def log_trade(entry: dict):
    log_dir = os.path.dirname(LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    header_needed = not os.path.exists(LOG_PATH)
    df = pd.DataFrame([entry])
    df.to_csv(LOG_PATH, mode="a", header=header_needed, index=False)


def load_trade_history(n: int = 50) -> list:
    if not os.path.exists(LOG_PATH):
        return []
    df = pd.read_csv(LOG_PATH)
    return df.tail(n).to_dict(orient="records")


# ────────────────────────────────────────────────────────────────────
# Kelly stake helper
# ────────────────────────────────────────────────────────────────────
def kelly_stake(prob: float, balance: float) -> float:
    if prob < MIN_EDGE:
        return 0.0
    k = KELLY_FRAC * (2 * prob - 1)
    return round(min(balance * k, MAX_STAKE), 2)


# ────────────────────────────────────────────────────────────────────
# App startup
# ────────────────────────────────────────────────────────────────────
app = FastAPI(title="BTC ML Trading Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded globals
fetcher: BinanceFetcher = None
engine:  MLEngine       = None

# In-memory virtual balance
virtual_balance = 1000.0
pending_trade = None  # {direction, stake, entry_price}

@app.on_event("startup")
async def startup():
    global fetcher, engine
    print("[Startup] Initializing Binance fetcher...")
    fetcher = BinanceFetcher()
    print("[Startup] Initializing ML engine (this may take ~30s)...")
    engine = MLEngine()
    print("[Startup] Ready!")


# ────────────────────────────────────────────────────────────────────
# Helper: run full inference
# ────────────────────────────────────────────────────────────────────
def run_inference() -> dict:
    global virtual_balance, pending_trade

    df_raw = fetcher.fetch_candles(limit=300)
    current_price = float(df_raw["close"].iloc[-1])

    # Resolve pending trade
    pnl = 0.0
    if pending_trade:
        t   = pending_trade
        pct = (current_price - t["entry_price"]) / t["entry_price"]
        won = (pct > 0) if t["direction"] == "UP" else (pct < 0)
        pnl             = t["stake"] if won else -t["stake"]
        virtual_balance = max(0.0, virtual_balance + pnl)
        pending_trade   = None

    df_feat, feat_names = extract_features(df_raw)
    tail = df_feat.tail(1).dropna()
    if tail.empty:
        return {"error": "Not enough data"}

    # Use canonical feature order from training CSV if available
    cols = CANONICAL_FEATURES if CANONICAL_FEATURES else feat_names
    # Only keep columns that exist in the computed features
    cols = [c for c in cols if c in tail.columns]
    X = tail[cols]
    prob      = engine.predict_prob(X)
    ind_probs = engine.individual_probs(X)

    # Signal
    if prob >= MIN_EDGE:
        signal = "BUY UP"
        directional_prob = prob
    elif (1 - prob) >= MIN_EDGE:
        signal = "BUY DOWN"
        directional_prob = 1 - prob
    else:
        signal = "HOLD"
        directional_prob = prob

    stake = kelly_stake(directional_prob, virtual_balance) if signal != "HOLD" else 0.0

    if signal != "HOLD" and stake > 0:
        direction = "UP" if signal == "BUY UP" else "DOWN"
        pending_trade = {"direction": direction, "stake": stake, "entry_price": current_price}

    ts = datetime.datetime.utcnow().isoformat()

    entry = {
        "timestamp":      ts,
        "btc_price":      current_price,
        "signal":         signal,
        "ensemble_prob":  round(prob, 6),
        "confidence":     round(directional_prob * 100, 2),
        "stake":          stake,
        "virtual_balance": round(virtual_balance, 2),
        "pnl":            round(pnl, 2),
        "xgb_prob":       round(ind_probs["xgboost"], 4),
        "lgb_prob":       round(ind_probs["lightgbm"], 4),
        "cb_prob":        round(ind_probs["catboost"], 4),
        "rf_prob":        round(ind_probs["random_forest"], 4),
    }
    log_trade(entry)
    return entry


# ────────────────────────────────────────────────────────────────────
# REST endpoints
# ────────────────────────────────────────────────────────────────────
@app.get("/api/price")
def get_price():
    try:
        price = fetcher.fetch_price()
        return {"price": price, "symbol": SYMBOL, "ts": datetime.datetime.utcnow().isoformat()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/signal")
def get_signal():
    try:
        return run_inference()
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history")
def get_history(n: int = 50):
    return {"trades": load_trade_history(n)}


@app.get("/api/candles")
def get_candles(limit: int = 100):
    try:
        df = fetcher.fetch_candles(limit=limit)
        records = df[["timestamp","open","high","low","close","volume"]].copy()
        records["timestamp"] = records["timestamp"].astype(str)
        return {"candles": records.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/balance")
def get_balance():
    return {"virtual_balance": round(virtual_balance, 2)}


# ────────────────────────────────────────────────────────────────────
# WebSocket — pushes price + signal every 5 s
# ────────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                price = fetcher.fetch_price()
                await websocket.send_json({
                    "type": "price",
                    "price": price,
                    "ts": datetime.datetime.utcnow().isoformat()
                })
            except (WebSocketDisconnect, RuntimeError):
                # Client gone — exit cleanly
                break
            except Exception:
                # Binance hiccup etc. — skip this tick, keep looping
                pass
            await asyncio.sleep(5)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        manager.disconnect(websocket)


# ────────────────────────────────────────────────────────────────────
# Serve frontend
# ────────────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(DASH_DIR, "frontend")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

_static_dir = os.path.join(FRONTEND_DIR, "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
