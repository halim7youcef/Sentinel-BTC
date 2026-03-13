"""
live_inference.py
=================
The Real-Time Sentinel for BTC Polymarket Trading.

Logic:
1. Syncs with the Binance 5m clock.
2. Wakes up at Candle Close + 2 seconds.
3. Fetches the latest 100 candles.
4. Computes 38 SOTA features + Calibrated ML Ensemble Probability.
5. Feeds the 64-candle sequence into the Transformer-PPO brain.
6. Outputs the high-conviction Action: [HOLD, UP, or DOWN].

Changes from v1
---------------
  FIX 1 — Calibrated ensemble probability: Raw model probabilities were not
           passed through the isotonic regressors saved in build_rl_features.py.
           get_ensemble_prob() now loads and applies calibrators so live
           ensemble_prob values are on the same scale as training data.

  FIX 2 — Stake sizing decoupled from policy confidence: calculate_confidence_stake
           was scaling stake from policy softmax probs (a PPO internal value),
           which is what caused the 0.51 ensemble_prob / 0.93 confidence disconnect.
           Stake is now sized via fractional Kelly using ensemble_prob directly,
           capped at $100 and min-thresholded at 0.55 edge to avoid near-coin-flip bets.

  FIX 3 — Balance update loop: self.balance was initialised but never updated
           after trade resolution. A pending_trade dict now tracks open positions
           and resolves them at the next candle close against the actual price move,
           updating self.balance accordingly.

  FIX 4 — Clock sync robustness: the previous sleep(1) polling loop could
           multi-fire within the same second near candle boundaries. Replaced
           with a precise sleep-until calculation that sleeps the exact number
           of seconds to the next 5m boundary + 2s buffer, firing exactly once.

  FIX 5 — ml_prob_up column populated with calibrated values: the per-row
           ensemble probability in the observation sequence now uses calibrated
           outputs (consistent with what the PPO was trained against) rather
           than raw tree model outputs.
"""

import os
import sys
import time
import warnings
import datetime
import pandas as pd
import numpy as np
import joblib
import torch
from binance.client import Client
from dotenv import load_dotenv
from stable_baselines3 import PPO

sys.path.append('src/models/rl')
sys.path.append('src/models/transformer')
sys.path.append('src/features')

from btc_attention import BTCAttentionExtractor

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# CONFIG & PATHS
# -------------------------------------------------------------------
MODEL_ML_DIR       = "models/ml"
MODEL_RL_PATH      = "models/rl/ppo_attention.zip"
SCALER_PATH        = "models/ml/robust_scaler.pkl"
CALIBRATORS_PATH   = "models/ml/isotonic_calibrators.pkl"   # FIX 1
LOG_PATH           = "results/live_trading_logs.csv"

SEQ_LEN            = 64
SYMBOL             = "BTCUSDT"
INTERVAL           = Client.KLINE_INTERVAL_5MINUTE
CANDLE_SECONDS     = 300          # 5 minutes in seconds
CLOCK_BUFFER_SECS  = 2            # wait after candle close for Binance stability

# Kelly sizing thresholds (FIX 2)
MIN_EDGE_THRESHOLD = 0.55         # below this ensemble_prob → HOLD, no bet
MAX_STAKE_USD      = 100.0        # hard cap per trade
KELLY_FRACTION     = 0.25         # fractional Kelly (full Kelly is too aggressive)

# -------------------------------------------------------------------
# FEATURE HELPERS (kept in sync with build_rl_features.py)
# -------------------------------------------------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def extract_live_features(df_raw):
    """Processes raw OHLCV into the 38 SOTA features needed by the brain."""
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

    df["rsi14"]       = rsi(df["close"], 14)
    e12               = ema(df["close"], 12)
    e26               = ema(df["close"], 26)
    macd_line         = e12 - e26
    df["macd_hist"]   = macd_line - ema(macd_line, 9)
    df["macd_zscore"] = (df["macd_hist"] - df["macd_hist"].rolling(20).mean()) / (df["macd_hist"].rolling(20).std() + 1e-8)

    bb = df["close"].rolling(20)
    df["bb_width"] = (bb.mean() + 2*bb.std() - (bb.mean() - 2*bb.std())) / (bb.mean() + 1e-8)
    df["bb_pos"]   = (df["close"] - (bb.mean() - 2*bb.std()))             / (4*bb.std() + 1e-8)

    for w in [10, 20, 50]:
        df[f"realized_vol_{w}"] = df["log_ret_1"].rolling(w).std() * np.sqrt(288)
    df["atr14"]     = atr(df["high"], df["low"], df["close"])
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


# -------------------------------------------------------------------
# KELLY STAKE SIZING (FIX 2)
# -------------------------------------------------------------------
def kelly_stake(prob: float, balance: float, max_stake: float = MAX_STAKE_USD,
                fraction: float = KELLY_FRACTION) -> float:
    """
    Fractional Kelly criterion stake sizing.

    For a binary Polymarket bet at ~even odds (b=1):
        Kelly fraction = p - (1 - p) = 2p - 1

    We apply a conservative fraction multiplier and cap at max_stake.
    Returns 0.0 if prob < MIN_EDGE_THRESHOLD (no meaningful edge).
    """
    if prob < MIN_EDGE_THRESHOLD:
        return 0.0
    kelly_f = fraction * (2 * prob - 1)
    stake   = min(balance * kelly_f, max_stake)
    return max(0.0, round(stake, 2))


# -------------------------------------------------------------------
# THE SENTINEL CLASS
# -------------------------------------------------------------------
class LiveTradingSentinel:
    def __init__(self, start_balance: float = 100.0):
        load_dotenv()
        self.api_key    = os.getenv("Binance_API")
        self.api_secret = os.getenv("Binance_Secret")

        if not self.api_key:
            raise ValueError("No Binance API Key found in .env")

        self.client  = Client(self.api_key, self.api_secret)
        self.balance = start_balance
        print(">>> Binance Client Connected.")
        print(f">>> Virtual Bankroll Initialized: ${self.balance:.2f}")

        # FIX 3: pending trade tracker for balance resolution
        self.pending_trade = None  # dict: {direction, stake, entry_price} or None

        # Load ML Ensemble
        print(">>> Loading ML Ensemble Models...")
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier

        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(f"{MODEL_ML_DIR}/xgboost_rl.json")
        self.lgb_model = lgb.Booster(model_file=f"{MODEL_ML_DIR}/lightgbm_rl.txt")
        self.cb_model  = CatBoostClassifier()
        self.cb_model.load_model(f"{MODEL_ML_DIR}/catboost_rl.cbm")
        self.rf_model  = joblib.load(f"{MODEL_ML_DIR}/rf_rl.pkl")
        self.scaler    = joblib.load(SCALER_PATH)

        # FIX 1: load isotonic calibrators saved by build_rl_features.py
        if os.path.exists(CALIBRATORS_PATH):
            self.calibrators = joblib.load(CALIBRATORS_PATH)
            print(f">>> Loaded probability calibrators from {CALIBRATORS_PATH}")
        else:
            self.calibrators = None
            print(f"!!! WARNING: No calibrators found at {CALIBRATORS_PATH}. "
                  "Raw uncalibrated probabilities will be used. "
                  "Re-run build_rl_features.py to generate them.")

        print(f">>> Loading RL Transformer Brain: {MODEL_RL_PATH}")
        self.rl_model = PPO.load(MODEL_RL_PATH, device="cpu")

        print(">>> Pre-fetching historical context...")
        self.history_df = self.fetch_latest_data()
        print(f">>> Warm-up Complete: {len(self.history_df)} candles loaded.")

    # ------------------------------------------------------------------
    # Data fetch
    # ------------------------------------------------------------------
    def fetch_latest_data(self) -> pd.DataFrame:
        """Fetches last 24h of 5m candles (~288 rows) for feature lookback."""
        klines  = self.client.get_historical_klines(SYMBOL, INTERVAL, "24 hours ago UTC")
        columns = ["open_time", "open", "high", "low", "close", "volume",
                   "ct", "qv", "nt", "tb", "tq", "i"]
        df = pd.DataFrame(klines, columns=columns)[
            ["open_time", "open", "high", "low", "close", "volume"]
        ]
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df

    # ------------------------------------------------------------------
    # Ensemble probability (FIX 1 — calibrated)
    # ------------------------------------------------------------------
    def get_ensemble_prob(self, X_live: pd.DataFrame) -> float:
        """
        Returns the calibrated 4-model ensemble probability for a single row.
        If calibrators are unavailable, falls back to raw probabilities with a warning.
        """
        import xgboost as xgb

        raw_xgb = self.xgb_model.predict(xgb.DMatrix(X_live))[0]
        raw_lgb = self.lgb_model.predict(X_live)[0]
        raw_cb  = self.cb_model.predict_proba(X_live)[0, 1]
        raw_rf  = self.rf_model.predict_proba(X_live)[0, 1]

        if self.calibrators:
            # Apply per-model isotonic calibration (FIX 1)
            cal_xgb = float(self.calibrators["xgboost"].transform([raw_xgb])[0])
            cal_lgb = float(self.calibrators["lightgbm"].transform([raw_lgb])[0])
            cal_cb  = float(self.calibrators["catboost"].transform([raw_cb])[0])
            cal_rf  = float(self.calibrators["random_forest"].transform([raw_rf])[0])
            return float(np.mean([cal_xgb, cal_lgb, cal_cb, cal_rf]))
        else:
            return float(np.mean([raw_xgb, raw_lgb, raw_cb, raw_rf]))

    # ------------------------------------------------------------------
    # Trade resolution (FIX 3)
    # ------------------------------------------------------------------
    def resolve_pending_trade(self, current_price: float):
        """
        Resolves the previous candle's trade at the new candle's open price.
        Updates self.balance and logs the outcome.
        """
        if self.pending_trade is None:
            return

        trade     = self.pending_trade
        direction = trade["direction"]   # "UP" or "DOWN"
        stake     = trade["stake"]
        entry     = trade["entry_price"]
        pct_move  = (current_price - entry) / entry

        if direction == "UP":
            won = pct_move > 0
        else:  # DOWN
            won = pct_move < 0

        pnl             = stake if won else -stake
        self.balance    = max(0.0, self.balance + pnl)
        self.pending_trade = None

        outcome_str = f"WIN  +${stake:.2f}" if won else f"LOSS -${stake:.2f}"
        print(f"  [RESOLVE] {direction} @ ${entry:,.2f} → ${current_price:,.2f} "
              f"({pct_move*100:+.2f}%)  {outcome_str}  "
              f"Balance: ${self.balance:.2f}")

    # ------------------------------------------------------------------
    # Stake sizing (FIX 2 — Kelly on ensemble_prob)
    # ------------------------------------------------------------------
    def calculate_stake(self, ensemble_prob: float, action: int) -> tuple:
        """
        Returns (stake, confidence, action) using Kelly criterion on ensemble_prob.

        action 0 = HOLD, 1 = BUY UP, 2 = BUY DOWN

        For a DOWN signal the relevant probability is (1 - ensemble_prob);
        for an UP signal it is ensemble_prob directly.
        """
        if action == 0:
            return 0.0, ensemble_prob, 0

        directional_prob = ensemble_prob if action == 1 else (1.0 - ensemble_prob)
        stake = kelly_stake(directional_prob, self.balance)

        if stake == 0.0:
            # Edge too thin — override to HOLD
            return 0.0, directional_prob, 0

        return stake, round(directional_prob, 4), action

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def enact_inference(self):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] Waking up for Inference...")

        df_raw = self.fetch_latest_data()
        current_price = float(df_raw["close"].iloc[-1])

        # FIX 3: resolve any trade opened on the previous candle
        self.resolve_pending_trade(current_price)

        df_feat, feature_names = extract_live_features(df_raw)

        tail_df = df_feat.tail(SEQ_LEN).copy().dropna()
        if len(tail_df) < SEQ_LEN:
            print(f"!!! Buffer underfill: {len(tail_df)}/{SEQ_LEN}. Skipping.")
            return

        # FIX 5: calibrated ml_prob_up for every row in the observation window
        probs = []
        for i in range(len(tail_df)):
            row_X = tail_df.iloc[[i]][feature_names]
            probs.append(self.get_ensemble_prob(row_X))
        tail_df = tail_df.copy()
        tail_df["ml_prob_up"] = probs

        obs_features  = feature_names + ["ml_prob_up"]
        obs_matrix    = tail_df[obs_features].values
        scaled_matrix = self.scaler.transform(tail_df[feature_names].values)
        combined_obs  = np.column_stack([tail_df["ml_prob_up"].values, scaled_matrix])
        flat_obs      = combined_obs.flatten().astype(np.float32)
        obs_tensor    = torch.from_numpy(flat_obs).unsqueeze(0)

        # PPO action (for direction signal only — NOT for sizing)
        with torch.no_grad():
            distribution = self.rl_model.policy.get_distribution(obs_tensor)
            probs_policy = distribution.distribution.probs.numpy()[0]
            action       = int(np.argmax(probs_policy))

        ensemble_prob = float(tail_df["ml_prob_up"].iloc[-1])

        # FIX 2: stake sized by Kelly on ensemble_prob, not on policy probs
        stake, conf, final_action = self.calculate_stake(ensemble_prob, action)

        action_map = {0: "HOLD", 1: "BUY UP", 2: "BUY DOWN"}
        decision   = action_map[final_action]

        print(f"  BTC Price    : ${current_price:,.2f}")
        print(f"  Ensemble Prob: {ensemble_prob:.4f}")
        print(f"  Signal       : {decision}")
        print(f"  Conviction   : {conf*100:.2f}%")
        print(f"  Placed Stake : ${stake:.2f}")
        print(f"  Balance      : ${self.balance:.2f}")

        # FIX 3: register trade for resolution at next candle
        if final_action != 0 and stake > 0:
            direction = "UP" if final_action == 1 else "DOWN"
            self.pending_trade = {
                "direction":   direction,
                "stake":       stake,
                "entry_price": current_price,
            }

        log_entry = pd.DataFrame([{
            "timestamp":    datetime.datetime.now(),
            "btc_price":    current_price,
            "decision":     decision,
            "stake":        stake,
            "confidence":   conf,
            "total_balance": self.balance,
            "ensemble_prob": ensemble_prob,
        }])
        log_entry.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

    # ------------------------------------------------------------------
    # Main loop (FIX 4 — precise clock sync, fires exactly once per candle)
    # ------------------------------------------------------------------
    def run_loop(self):
        print("=" * 50)
        print("TRANSFORMER LIVE SENTINEL STARTED")
        print("=" * 50)
        print(f"Monitoring {SYMBOL} | {INTERVAL} | Balance: ${self.balance:.2f}")

        while True:
            now            = datetime.datetime.utcnow()
            elapsed_in_5m  = (now.minute % 5) * 60 + now.second
            secs_to_close  = CANDLE_SECONDS - elapsed_in_5m

            # Sleep precisely until candle close + buffer (FIX 4)
            sleep_secs = secs_to_close + CLOCK_BUFFER_SECS
            print(f"  Next inference in {sleep_secs}s "
                  f"(candle closes at :{(now.minute // 5 + 1) * 5 % 60:02d}:00 UTC)  ",
                  end="\r")
            time.sleep(max(1, sleep_secs))

            try:
                self.enact_inference()
            except Exception as e:
                print(f"\n!!! Inference error: {e}")


if __name__ == "__main__":
    sentinel = LiveTradingSentinel()
    sentinel.run_loop()