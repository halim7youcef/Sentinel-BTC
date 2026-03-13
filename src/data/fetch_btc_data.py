"""
fetch_btc_data.py
=================
Fetches 4 years of BTCUSDT 5-minute candles from Binance and saves to:
  data/raw/btc_5m.csv

Notes:
- Binance caps each API call at 1,000 candles. The client paginates automatically.
- 4 years of 5m data = ~420,480 candles (~420 API calls), expect ~3-5 minutes.
- Requires python-binance and python-dotenv installed.
- Place your API keys in .env as:
    Binance_API=...
    Binance_Secret=...
"""

import os
import time
import datetime
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client

# -----------------------
# Config
# -----------------------
YEARS    = 4
LOOKBACK = f"{YEARS * 365} days ago UTC"   # 1,460 days = 4 years
SYMBOL   = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_5MINUTE

OUT_DIR  = "data/raw"
OUT_FILE = os.path.join(OUT_DIR, "btc_5m.csv")

# -----------------------
# Load env and client
# -----------------------
load_dotenv()
API_KEY    = os.getenv("Binance_API")
API_SECRET = os.getenv("Binance_Secret")

if API_KEY is None or API_SECRET is None:
    raise RuntimeError("Binance_API and Binance_Secret must be set in .env")

client = Client(API_KEY, API_SECRET)

# -----------------------
# Estimate size upfront
# -----------------------
CANDLES_PER_DAY  = 288          # 24 * 60 / 5
TOTAL_DAYS       = YEARS * 365
EXPECTED_CANDLES = TOTAL_DAYS * CANDLES_PER_DAY

print("=" * 60)
print(f"Binance BTC/USDT 5-Minute Kline Fetcher")
print("=" * 60)
print(f"Lookback    : {YEARS} years ({TOTAL_DAYS} days)")
print(f"Expected    : ~{EXPECTED_CANDLES:,} candles")
print(f"Output      : {OUT_FILE}")
print("=" * 60)
print("Starting download (pagination handled automatically)...")

t0 = time.time()

# -----------------------
# Fetch data
# python-binance handles pagination internally — it will loop
# until all historical klines have been retrieved.
# -----------------------
klines = client.get_historical_klines(SYMBOL, INTERVAL, LOOKBACK)

elapsed = time.time() - t0

if not klines:
    raise RuntimeError("No klines returned from Binance. Check API keys / rate limits.")

print(f"Downloaded {len(klines):,} candles in {elapsed:.1f}s")

# -----------------------
# Build DataFrame
# -----------------------
columns = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

df = pd.DataFrame(klines, columns=columns)

# Keep useful columns only
df = df[["open_time", "open", "high", "low", "close", "volume"]]
df = df.rename(columns={"open_time": "timestamp"})

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
for c in ["open", "high", "low", "close", "volume"]:
    df[c] = df[c].astype(float)

# Sort ascending and drop any duplicates (rare but can happen at pagination seams)
df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

# -----------------------
# Save
# -----------------------
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

start_date = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
end_date   = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")

print("=" * 60)
print(f"Saved    : {OUT_FILE}")
print(f"Rows     : {len(df):,}")
print(f"Range    : {start_date}  →  {end_date}")
print(f"Columns  : {list(df.columns)}")
print("=" * 60)