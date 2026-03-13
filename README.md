# BTC RL Project

> **Real-time Bitcoin trading intelligence powered by a 4-model ML ensemble and a PPO Transformer agent.**

A complete BTC/USDT algorithmic trading research system that fetches live 5-minute OHLCV data from Binance, computes 38 financial features, runs a calibrated soft-voting ML ensemble, and serves a live paper-trading dashboard. A separate PPO + Transformer reinforcement learning agent is included for research and backtesting.

**All trading is virtual / paper trading. No real orders are placed.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture](#3-architecture)
4. [Feature Engineering](#4-feature-engineering)
5. [Machine Learning Models](#5-machine-learning-models)
6. [Reinforcement Learning — PPO + Transformer](#6-reinforcement-learning--ppo--transformer)
7. [Live Inference Engine](#7-live-inference-engine)
8. [Live Trading Dashboard](#8-live-trading-dashboard)
9. [Data Flow](#9-data-flow)
10. [Setup & Installation](#10-setup--installation)
11. [How to Run](#11-how-to-run)
12. [Configuration Reference](#12-configuration-reference)
13. [Model Performance](#13-model-performance)
14. [Known Limitations](#14-known-limitations)

---

## 1. Project Overview

| Layer | Technology | Purpose |
|---|---|---|
| **ML Ensemble** | XGBoost + LightGBM + CatBoost + Random Forest | Predict 30-min directional probability from OHLCV features |
| **RL Agent** | Stable-Baselines3 PPO + custom Transformer extractor | Learn a sequential policy over a 128-candle sliding window, consuming the ML probability as an additional signal |

The **live trading dashboard** (`dashboard/`) uses **only the ML ensemble** — it is fast, interpretable, and requires no GPU. The PPO agent is a separate research component for backtesting and paper trading.

Stake sizing uses the **fractional Kelly criterion** on the calibrated ensemble probability for principled, bounded risk management.

---

## 2. Repository Structure

```
btc_rl_project/
│
├── data/
│   ├── raw/
│   │   └── btc_5m.csv                   ← Historical 5-minute OHLCV (Binance, 4 years)
│   └── processed/
│       ├── btc_features_ml.csv          ← ML training dataset (38 features + binary target)
│       └── btc_features_full.csv        ← Full feature dataset (includes indeterminate rows)
│
├── models/
│   └── ml/
│       ├── xgboost_rl.json              ← Trained XGBoost booster
│       ├── lightgbm_rl.txt              ← Trained LightGBM booster
│       ├── catboost_rl.cbm              ← Trained CatBoost model
│       ├── rf_rl.pkl                    ← Trained Random Forest (scikit-learn)
│       ├── robust_scaler.pkl            ← RobustScaler fit on ML features
│       └── isotonic_calibrators.pkl     ← Per-model isotonic probability calibrators
│
├── src/
│   ├── data/
│   │   └── fetch_btc_data.py            ← Download 4 years of 5m OHLCV from Binance
│   ├── features/
│   │   ├── build_ml_features.py         ← Generate btc_features_ml.csv from raw OHLCV
│   │   └── build_rl_features.py         ← Generate RL observation sequences + calibrators
│   ├── models/
│   │   ├── ml/
│   │   │   ├── xgboost_model.py         ← Train XGBoost
│   │   │   ├── lightgbm_model.py        ← Train LightGBM
│   │   │   ├── catboost_model.py        ← Train CatBoost
│   │   │   ├── random_forest.py         ← Train Random Forest
│   │   │   └── ensemble_model.py        ← Soft-vote ensemble evaluation
│   │   ├── rl/
│   │   │   ├── poly_env.py              ← Custom Gymnasium environment
│   │   │   ├── train_ppo.py             ← Baseline PPO trainer
│   │   │   └── train_ppo_attention.py   ← PPO + Transformer trainer
│   │   └── transformer/
│   │       └── btc_attention.py         ← Custom Transformer feature extractor (SB3)
│   ├── live_trading/
│   │   └── live_inference.py            ← Full Transformer-PPO live sentinel
│   └── evaluation/
│       ├── backtest.py                  ← Backtest evaluation script
│       ├── evaluate_rl.py               ← Evaluate trained PPO agent
│       └── evaluate_attention_rl.py     ← Evaluate PPO + Transformer agent
│
├── results/
│   ├── predictions/                     ← Per-model CSV predictions on test set
│   ├── metrics/                         ← JSON accuracy / AUC metrics
│   ├── curves/                          ← Feature importance plots
│   ├── backtest/                        ← Backtest output files
│   └── dashboard_trades.csv             ← Live trade log written by the dashboard
│
└── dashboard/                           ← Live trading dashboard (ML-only)
    ├── backend/
    │   ├── app.py                       ← FastAPI server
    │   └── requirements.txt             ← Python dependencies
    ├── frontend/
    │   ├── index.html                   ← Dashboard UI
    │   └── static/
    │       ├── style.css                ← Dark glassmorphism design system
    │       └── app.js                   ← WebSocket stream, charts, inference trigger
    ├── .env.example                     ← API key template
    ├── run.py                           ← Python launcher
    └── start.bat                        ← Windows double-click launcher
```

---

## 3. Architecture

```
                    ┌──────────────────────────────────────┐
                    │          Binance REST API             │
                    │   GET /api/v3/klines  (5m OHLCV)     │
                    └──────────────┬───────────────────────┘
                                   │  300 candles (~25 h)
                                   ▼
                    ┌──────────────────────────────────────┐
                    │       Feature Engineering             │
                    │  38 features: log-returns, RSI, MACD,│
                    │  Bollinger, ATR, Z-scores, skew, kurt,│
                    │  cyclical time encodings, realized vol│
                    └──────┬──────────────────┬────────────┘
                           │                  │
               ┌───────────▼──────────┐  ┌───▼───────────────────────┐
               │    ML Ensemble        │  │  Transformer-PPO Brain     │
               │  XGBoost             │  │  (live_inference.py)        │
               │  LightGBM            │  │  128-candle sequence         │
               │  CatBoost            │  │  + calibrated ensemble_prob  │
               │  Random Forest       │  │  → Transformer extractor    │
               │  ↓ Isotonic calib.   │  │  → PPO policy               │
               │  → P(UP) ∈ [0,1]    │  │  → HOLD / BUY UP / DOWN     │
               └───────────┬──────────┘  └───────────────────────────┘
                           │
               ┌───────────▼──────────┐
               │   Kelly Criterion     │
               │  f* = 0.25×(2p−1)    │
               │  cap: $100 / trade   │
               └───────────┬──────────┘
                           │
               ┌───────────▼──────────┐
               │  Live Dashboard       │
               │  FastAPI + WebSocket  │
               │  Candlestick chart    │
               │  Model breakdown      │
               │  Trade history        │
               └──────────────────────┘
```

---

## 4. Feature Engineering

**Script:** `src/features/build_ml_features.py`  
**Output:** `data/processed/btc_features_ml.csv`

All 38 features are computed from raw 5-minute OHLCV candles:

| Category | Features |
|---|---|
| **Log Returns** | `log_ret_1`, `log_ret_3`, `log_ret_5`, `log_ret_10`, `log_ret_20`, `log_ret_50`, `log_ret_100` |
| **Momentum** | `acceleration` = log_ret_1 − log_ret_3 |
| **Candle Microstructure** | `hl_range`, `body_size`, `micro_clv`, `garman_klass_vol` |
| **Rolling Z-scores** | `zscore_close_{10,20,50,200}`, `zscore_volume_{10,20,50,200}` |
| **Oscillators** | `rsi14`, `macd_hist`, `macd_zscore` |
| **Bollinger Bands** | `bb_width`, `bb_pos` |
| **Realized Volatility** | `realized_vol_{10,20,50}` |
| **ATR** | `atr14`, `atr_ratio` |
| **Time Encodings** | `hour_sin`, `hour_cos`, `day_sin`, `day_cos` |
| **Higher Moments** | `skew_{20,50}`, `kurt_{20,50}` |

### Target Variable

```
FUTURE_PERIODS = 6   (30 minutes ahead)
future_return  = close[t+6] / close[t] - 1

target = 1   if future_return >  +0.15%   (BUY signal)
target = 0   if future_return <  -0.15%   (SELL signal)
[removed]    if |future_return| ≤ 0.15%   (indeterminate — excluded from ML training)
```

The 0.15% threshold comfortably clears a typical 0.04% Binance taker fee on a round-trip.

---

## 5. Machine Learning Models

All four models are trained on `btc_features_ml.csv` with an 80/20 chronological train/test split and a **6-candle purge gap** to prevent lookahead bias.

### XGBoost — `src/models/ml/xgboost_model.py`

```python
params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "learning_rate":    0.01,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.5,
    "alpha":            2.0,   # L1
    "lambda":           2.0,   # L2
    "tree_method":      "hist",
}
num_boost_round = 1000
early_stopping  = 150
```

Heavy L1+L2 regularization prevents overfitting on the noisy 5-minute BTC time series.

### LightGBM — `src/models/ml/lightgbm_model.py`

Microsoft's leaf-wise tree learner. Faster than XGBoost on large datasets; uses a different split-finding algorithm, adding diversity to the ensemble.

### CatBoost — `src/models/ml/catboost_model.py`

Yandex's ordered-boosting algorithm using symmetric decision trees. More robust to outliers and contributes additional decorrelation in the ensemble.

### Random Forest — `src/models/ml/random_forest.py`

Scikit-learn `RandomForestClassifier`. Decorrelated from the boosting models via bootstrap aggregation; produces stable, well-calibrated probabilities from `predict_proba`.

### Ensemble — `src/models/ml/ensemble_model.py`

**Soft Voting** — arithmetic mean of the four models' class-1 probabilities:

```
ensemble_prob = (P_xgb + P_lgb + P_cb + P_rf) / 4
```

### Isotonic Calibration

**Script:** `src/features/build_rl_features.py`  
**File:** `models/ml/isotonic_calibrators.pkl`

Boosted models tend to be overconfident; forests tend to be underconfident. Isotonic regression is applied per-model on the validation set to map raw scores to calibrated probabilities:

```python
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(raw_probs_val, y_val)
calibrated_prob = calibrator.transform([raw_prob])[0]
```

After calibration, `P(UP) = 0.62` genuinely means "the price went up 62% of the time."

---

## 6. Reinforcement Learning — PPO + Transformer

> **Note:** The PPO agent is **not used** in the live dashboard. It requires `models/rl/ppo_attention.zip` and operates via `src/live_trading/live_inference.py` only.

### Environment — `src/models/rl/poly_env.py`

Custom Gymnasium environment wrapping the BTC 5-minute time series:

| Property | Value |
|---|---|
| **Observation** | 128-candle sliding window × (38 features + 1 calibrated ML prob) — flat vector `[128 × 39]` |
| **Action space** | `Discrete(3)` = `{0: HOLD, 1: BUY UP, 2: BUY DOWN}` |
| **Reward** | P&L from the previous action scaled by Kelly stake |
| **Episode** | One continuous episode over the full training period |

The `ml_prob_up` column (calibrated ensemble probability) is embedded in the observation so the PPO can learn *when to trust the ML signal* versus when to override it based on sequential market context.

### Transformer Feature Extractor — `src/models/transformer/btc_attention.py`

A custom `BaseFeaturesExtractor` (Stable-Baselines3 interface) that processes the flat 128-step observation as a sequence through stacked multi-head self-attention blocks, then feeds the encoded representation into the PPO actor/critic heads.

**Architecture:**
```
Flat obs [seq_len × n_features]
      │
      └─► Reshape → [batch, 128, 39]
              │
              ▼
   Linear Projection → [batch, 128, embed_dim=128]
              │
              ▼
   Sinusoidal Positional Encoding (causal-aware)
              │
              ▼
   TransformerEncoder × 2 layers
     ├─ MultiHeadAttention (4 heads, causal mask)
     ├─ FeedForward (embed_dim × 4)
     └─ LayerNorm + Dropout(0.1)
              │
              ▼
   Last-Token Pooling → [batch, 128]
              │
              ▼
   LayerNorm → Linear → features_dim=128
```

A causal attention mask ensures position `T` only attends to positions `0..T`, preventing future-price leakage during sequence processing.

### PPO Agent — `src/models/rl/train_ppo_attention.py`

```python
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs={"features_extractor_class": BTCAttentionExtractor},
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,   # entropy bonus prevents premature convergence
    device        = "cpu",
)
```

- **GAE (λ=0.95):** Generalized Advantage Estimation reduces policy gradient variance.
- **Entropy bonus (0.01):** Prevents collapse to a degenerate all-HOLD policy.
- **clip_range=0.2:** Standard PPO clipping for conservative policy updates.

An experimental SAC (Soft Actor-Critic) trainer is also available at `src/train/sac_btc_trainer.py`. The final research model uses PPO due to more stable training on non-stationary time series.

---

## 7. Live Inference Engine

### ML-Only Mode — Dashboard (`dashboard/backend/app.py`)

```python
def run_inference() -> dict:
    df_raw  = fetch_candles(limit=300)         # ~25 h of 5m candles
    df_feat = extract_features(df_raw)         # 38 features
    X       = tail_row[CANONICAL_FEATURES]     # exact training column order
    prob    = ensemble.predict_prob(X)         # calibrated P(UP)

    if prob >= 0.55:      signal = "BUY UP"
    elif 1-prob >= 0.55:  signal = "BUY DOWN"
    else:                 signal = "HOLD"

    stake = kelly_stake(prob, virtual_balance)
    log_trade(entry)
    return entry
```

### Full Transformer-PPO Mode — `src/live_trading/live_inference.py`

The `LiveTradingSentinel` class runs a precision-clocked inference loop:

1. **Syncs** to the 5-minute Binance candle boundary (sleeps to next close + 2 s)
2. **Fetches** the last ~288 5-minute candles from Binance REST
3. **Computes** all 38 SOTA features inline
4. **Applies** isotonic calibration to all rows in the 128-candle window
5. **Feeds** the `128×39` observation into the Transformer-PPO brain via `policy.get_distribution()`
6. **Sizes** the stake using Kelly criterion on `ensemble_prob` (not the PPO softmax)
7. **Resolves** the previous candle's trade at the next open
8. **Logs** every decision to `results/live_trading_logs.csv`

**Key implementation fixes in the inference pipeline:**

| Fix | Problem | Solution |
|---|---|---|
| FIX 1 | Raw ML probs not calibrated in live mode | Isotonic calibrators applied to all rows in the window |
| FIX 2 | Stake sized by PPO softmax (incorrect) | Kelly criterion applied to `ensemble_prob` directly |
| FIX 3 | Virtual balance never updated | `pending_trade` dict resolves at next candle open |
| FIX 4 | Sleep loop could fire multiple times per candle | Precise sleep-until-boundary calculation |
| FIX 5 | `ml_prob_up` in RL obs used raw probs | Now uses calibrated probs, consistent with training |

---

## 8. Live Trading Dashboard

### Backend — FastAPI (`dashboard/backend/app.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves `frontend/index.html` |
| `/api/price` | GET | Latest BTC price from Binance ticker |
| `/api/signal` | GET | Run full ML inference; return signal + per-model probs |
| `/api/candles?limit=N` | GET | Last N OHLCV candles for the chart |
| `/api/history?n=N` | GET | Last N trade log entries from CSV |
| `/api/balance` | GET | Current virtual balance |
| `/ws/live` | WebSocket | Pushes `{type:"price", price:..., ts:...}` every 5 seconds |

**Startup sequence:**
1. Load `.env` (searched in `btc_rl_project/`, `dashboard/`, and CWD)
2. Read canonical feature column order from `data/processed/btc_features_ml.csv`
3. Connect Binance client
4. Load all 4 ML models + scaler + calibrators (~30 s for Random Forest)

### Frontend — `dashboard/frontend/`

| File | Purpose |
|---|---|
| `index.html` | Semantic HTML structure; unique IDs on all interactive elements |
| `static/style.css` | Dark glassmorphism design system: CSS variables, animated gauge, responsive grid |
| `static/app.js` | WebSocket price stream, Lightweight-Charts candlestick chart, inference trigger, model breakdown bars, trade history table |

**Key features:**
- **Live price stream** via WebSocket with automatic REST fallback every 5 s
- **Auto-inference scheduling** — calculates milliseconds to the next 5-minute candle boundary and fires inference at candle close + 2 s, then every 5 minutes thereafter
- **Candlestick chart** using [Lightweight Charts v4](https://github.com/tradingview/lightweight-charts) — real Binance OHLCV data
- **Model breakdown bars** — per-model probability bars colored by model identity
- **Calibrated probability gauge** — sliding indicator from Bearish to Bullish
- **Trade history table** — all logged trades with PnL and running balance

### Kelly Criterion Stake Sizing

```
Fractional Kelly stake = 0.25 × (2p − 1) × balance
Hard cap = min(stake, $100)

Example — p = 0.60, balance = $1,000:
  stake = 0.25 × (2 × 0.60 − 1) × 1,000 = $50.00

p < 0.55 → stake = $0 (HOLD, no trade placed)
```

The **0.25 fractional multiplier** derates full Kelly to provide ~¼ the asymptotic growth rate but substantially better drawdown protection.

---

## 9. Data Flow

```
Binance Historical Data
        │
        ▼
btc_5m.csv  (raw OHLCV, 4 years)
        │
        ├──► build_ml_features.py
        │         │
        │         ▼
        │    btc_features_ml.csv  ──► Train XGB / LGB / CB / RF
        │                                    │
        │                             models/ml/*.{json,txt,cbm,pkl}
        │
        └──► build_rl_features.py
                  │
                  ├──► isotonic_calibrators.pkl   (fit on validation set)
                  ├──► robust_scaler.pkl          (fit on RL features)
                  └──► RL observation sequences   (128×39, calibrated)
                                │
                                ▼
                        Train PPO + Transformer
                                │
                                ▼
                        models/rl/ppo_attention.zip


Dashboard (ML-only):
  Binance REST → 300 candles → extract_features() → ML ensemble → P(UP)
                                                          │
                                                  Kelly stake sizing
                                                          │
                                                  Dashboard UI + CSV log

Full sentinel (live_inference.py):
  Binance REST → 300 candles → extract_features()
                                     │
                          128-candle window × calibrated ensemble prob
                                     │
                          Transformer extractor → PPO policy → Action
                                     │
                          Kelly stake on ensemble_prob
                                     │
                          Log + resolve previous trade
```

---

## 10. Setup & Installation

### Requirements

- **Python 3.10**
- A Binance account with a **read-only** API key (the system only calls `GET /klines` and `GET /ticker/price` — no order placement is implemented)

### Install Dependencies

```bash
pip install -r dashboard/backend/requirements.txt
```

For the PPO / Transformer components (`live_inference.py` only), additionally install:

```bash
pip install torch>=2.0.0 stable-baselines3>=2.0.0 gymnasium
```

### Full dependency list

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-binance>=1.0.19
python-dotenv>=1.0.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
joblib>=1.3.0
# PPO / Transformer only:
torch>=2.0.0
stable-baselines3>=2.0.0
gymnasium
```

### API Keys

Copy the example file and fill in your credentials:

```bash
cp dashboard/.env.example .env
```

`.env` contents:
```env
Binance_API=your_api_key_here
Binance_Secret=your_secret_key_here
```

> ⚠️ Use a **read-only** API key. The system never places orders.

---

## 11. How to Run

### Option A — Double-click launcher (Windows)

```
dashboard/start.bat
```

Then open **http://localhost:8000** in your browser.

### Option B — Python launcher

```powershell
# From btc_rl_project/
python dashboard/run.py
```

### Option C — Direct uvicorn (any platform)

```powershell
# From btc_rl_project/
$env:PYTHONPATH = (Get-Item ..).FullName   # PowerShell
python -m uvicorn dashboard.backend.app:app --host 0.0.0.0 --port 8000
```

```bash
# bash / Linux / macOS
export PYTHONPATH=$(dirname $(pwd))
python -m uvicorn dashboard.backend.app:app --host 0.0.0.0 --port 8000
```

### Full Transformer-PPO Sentinel (CLI, no dashboard)

```powershell
python src/live_trading/live_inference.py
```

Wakes at every 5-minute candle close and prints:
```
[01:30:02] Waking up for Inference...
  BTC Price    : $71,339.62
  Ensemble Prob: 0.6123
  Signal       : BUY UP
  Conviction   : 61.23%
  Placed Stake : $28.00
  Balance      : $1,028.00
```

### Data Pipeline (re-run when raw data is updated)

```bash
# 1. Fetch raw OHLCV (4 years, ~420k candles)
python src/data/fetch_btc_data.py

# 2. Build ML feature set
python src/features/build_ml_features.py

# 3. Build RL observations + calibrators
python src/features/build_rl_features.py
```

### Model Training

```bash
python src/models/ml/xgboost_model.py
python src/models/ml/lightgbm_model.py
python src/models/ml/catboost_model.py
python src/models/ml/random_forest.py
python src/models/ml/ensemble_model.py    # evaluation only, no saved artifact
```

---

## 12. Configuration Reference

### Dashboard — `dashboard/backend/app.py`

| Variable | Default | Description |
|---|---|---|
| `SYMBOL` | `BTCUSDT` | Trading pair |
| `INTERVAL` | `5m` | Candle interval |
| `MIN_EDGE` | `0.55` | Minimum ensemble prob to open a trade |
| `MAX_STAKE` | `$100` | Hard cap per trade |
| `KELLY_FRAC` | `0.25` | Fractional Kelly multiplier |

### Full Sentinel — `src/live_trading/live_inference.py`

| Variable | Default | Description |
|---|---|---|
| `SEQ_LEN` | `128` | Candle window for PPO observation |
| `CANDLE_SECONDS` | `300` | 5 minutes in seconds |
| `CLOCK_BUFFER_SECS` | `2` | Post-close buffer before inference fires |
| `MIN_EDGE_THRESHOLD` | `0.55` | Kelly edge threshold |
| `MAX_STAKE_USD` | `$100` | Hard cap per trade |
| `KELLY_FRACTION` | `0.25` | Fractional Kelly multiplier |

---

## 13. Model Performance

All metrics are on the **out-of-sample chronological test set** (last 20% of data, approximately 6 months of BTC 5-minute data):

| Model | Accuracy | ROC-AUC |
|---|---|---|
| XGBoost | ~57–59% | ~0.60–0.63 |
| LightGBM | ~56–58% | ~0.59–0.62 |
| CatBoost | ~56–58% | ~0.59–0.62 |
| Random Forest | ~55–57% | ~0.57–0.60 |
| **Ensemble** | **~58–60%** | **~0.62–0.65** |

> 55%+ accuracy on 5-minute BTC prediction is considered strong in the academic literature. The market is approximately efficient at this granularity — consistent alpha is hard-won and degrades as market conditions shift over time.

The 0.55 trading threshold is intentionally conservative: the system only trades when the ensemble has a meaningful statistical edge, not on every candle.

---

## 14. Known Limitations

- **Paper trading only.** No real orders are placed. Extending to live execution requires implementing Binance Futures or Spot `POST /order` endpoints with appropriate risk controls.
- **Static models.** The ML models do not retrain automatically. Significant market regime changes (macro events, structural breaks) will degrade accuracy until models are retrained on fresh data.
- **5-minute granularity only.** The feature pipeline is fixed to 5m candles. Multi-timeframe features (e.g., 1h trend + 5m signal) are not implemented.
- **Windows-specific launcher paths.** `start.bat` hardcodes a Windows Store Python 3.10 path. Update `start.bat` if Python is installed elsewhere or upgraded.
- **Sequential model loading is slow.** The Random Forest model (~15 MB) takes approximately 20–30 seconds to deserialize via `joblib` on cold start. This is a one-time cost per server start.

---

*Python 3.10 · Binance BTCUSDT 5m · Stable-Baselines3 PPO · FastAPI*
