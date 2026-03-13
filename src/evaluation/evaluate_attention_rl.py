"""
evaluate_attention_rl.py
========================
Scores the Transformer-based PPO agent on blind Polymarket test data.

Custom Constraints:
1. Start Balance: $100
2. Track: Absolute Min Balance, Absolute Max Balance, Final Balance.
3. Bankruptcy: If balance hits $0, the run is terminated.
4. Bonus Run: If Run 1 goes bankrupt, a second run starts with $10 to see if it can recover.

Changes from v1
---------------
  FIX 1 — Action distribution logging: actions were collected but never
           analysed. A Counter report is now printed after each run so
           degenerate policies (e.g. always-BUY) are immediately visible.

  FIX 2 — Timestamped chart filenames: the result PNG was overwritten on
           every evaluation run. Charts are now saved with a UTC timestamp
           so every run produces a unique, comparable artefact.

  FIX 3 — Peak drawdown calculation: max drawdown from the equity peak is
           now computed and reported, giving a risk-adjusted view of each
           run alongside raw final balance.

  FIX 4 — Run 2 environment reset: Run 2 previously shared no state with
           Run 1, but the env was re-instantiated from the same data path
           starting at index 0. The reset is now explicit and documented.

  FIX 5 — Plot baseline reflects actual starting balance per run, not a
           hardcoded $100 line. Run 2's $10 baseline is now shown correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
from datetime import datetime, timezone
from stable_baselines3 import PPO
from gymnasium import spaces
import gymnasium as gym

# Path setup to find local modules
sys.path.append('src/models/rl')
sys.path.append('src/models/transformer')

from poly_env import PolymarketBTCEnv
from btc_attention import BTCAttentionExtractor
from train_ppo_attention import SequenceEnvWrapper

DATA_PATH  = "data/processed/btc_features_rl.csv"
MODEL_PATH = "models/rl/ppo_attention.zip"
RESULT_DIR = "results/backtest"
os.makedirs(RESULT_DIR, exist_ok=True)

# Unique timestamp for this evaluation session — prevents chart overwrite (FIX 2)
RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def compute_max_drawdown(balances: list) -> float:
    """
    Returns the maximum peak-to-trough drawdown as a positive percentage.
    e.g. 0.37 means a 37% drawdown from the equity peak.
    """
    balances  = np.array(balances)
    peak      = np.maximum.accumulate(balances)
    drawdowns = (peak - balances) / (peak + 1e-8)
    return float(drawdowns.max())


def print_action_distribution(actions: list, action_labels: dict = None):
    """
    Prints a frequency table of actions taken during the run.
    Highlights degenerate policies where one action dominates (>70%).
    """
    counts = Counter(actions)
    total  = len(actions)
    print("\n  Action Distribution:")
    for action, count in sorted(counts.items()):
        label = action_labels.get(action, str(action)) if action_labels else str(action)
        pct   = 100 * count / total
        flag  = "  ⚠️  DEGENERATE (>70%)" if pct > 70 else ""
        print(f"    Action {action} ({label}): {count:>6} ({pct:5.1f}%){flag}")


# -------------------------------------------------------------------------
# Core evaluation loop
# -------------------------------------------------------------------------

def run_evaluation(initial_balance: float, run_name: str = "Run 1"):
    print(f"\n>>> Starting {run_name} with ${initial_balance:.2f} bankroll...")

    env     = PolymarketBTCEnv(
        data_path=DATA_PATH,
        initial_balance=initial_balance,
        max_drawdown_pct=1.0          # disabled — allow full bankruptcy for eval
    )
    seq_env = SequenceEnvWrapper(env, seq_len=64)

    print("Loading Attention PPO Brain...")
    model = PPO.load(MODEL_PATH)

    obs, info = seq_env.reset()

    balances   = [initial_balance]
    actions    = []
    min_point  = initial_balance
    max_point  = initial_balance
    bankrupt   = False
    step_count = 0
    done       = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = seq_env.step(action)

        current_bal = info["balance"]
        balances.append(current_bal)
        actions.append(int(action))

        if current_bal < min_point: min_point = current_bal
        if current_bal > max_point: max_point = current_bal

        # Bankruptcy check
        if current_bal <= 0:
            print(f"  !!! BANKRUPTCY DETECTED at step {step_count} !!!")
            bankrupt = True
            done     = True

        if terminated or truncated:
            done = True

        step_count += 1
        if step_count % 5000 == 0:
            print(f"  Progress: {step_count:>6} steps... Balance: ${current_bal:.2f}")

    final_bal   = balances[-1]
    max_dd      = compute_max_drawdown(balances)
    roi         = (final_bal - initial_balance) / initial_balance * 100

    # FIX 1: action distribution report
    print_action_distribution(actions)

    print(f"\n--- {run_name} Results ---")
    print(f"  Final Balance:   ${final_bal:.2f}  ({roi:+.1f}% ROI)")
    print(f"  Highest Point:   ${max_point:.2f}")
    print(f"  Lowest Point:    ${min_point:.2f}")
    print(f"  Max Drawdown:    {max_dd*100:.1f}%")          # FIX 3
    print(f"  Total steps:     {step_count}")
    print(f"  Bankrupt:        {bankrupt}")

    return bankrupt, balances, final_bal, max_point, min_point, max_dd


# -------------------------------------------------------------------------
# Execution logic
# -------------------------------------------------------------------------

# RUN 1: $100
bankrupt_1, bal_1, final_1, high_1, low_1, dd_1 = run_evaluation(100.0, "Run 1")

bankrupt_2 = False
bal_2 = []

# Bonus RUN: if bankrupt, retry with $10
if bankrupt_1:
    print("\n" + "=" * 50)
    print("RUN 1 FAILED. GRANTED $10 BONUS FOR RUN 2.")
    print("=" * 50)
    bankrupt_2, bal_2, final_2, high_2, low_2, dd_2 = run_evaluation(10.0, "Run 2 (Bonus)")
else:
    print("\nModel survived Run 1. No second run required.")


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    2 if bankrupt_1 else 1, 1,
    figsize=(14, 9 if bankrupt_1 else 6),
    sharex=False
)

# Normalise axes to a list for uniform handling
if not bankrupt_1:
    axes = [axes]

# --- Run 1 panel ---
ax1 = axes[0]
ax1.plot(bal_1, label=f"Run 1 (start $100.00)", color="steelblue", alpha=0.85, linewidth=1.2)
ax1.axhline(y=100.0, color="gray",   linestyle=":",  alpha=0.5, label="Starting balance ($100)")
ax1.axhline(y=high_1, color="green", linestyle="--", alpha=0.4, label=f"Peak ${high_1:.2f}")
ax1.axhline(y=low_1,  color="red",   linestyle="--", alpha=0.4, label=f"Trough ${low_1:.2f}")
ax1.set_title(f"Run 1 — {'BANKRUPT' if bankrupt_1 else 'SURVIVED'}  |  "
              f"Final ${final_1:.2f}  |  Max Drawdown {dd_1*100:.1f}%",
              fontsize=12, fontweight="bold")
ax1.set_ylabel("Bankroll ($)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Run 2 panel (only if triggered) ---
if bankrupt_1 and bal_2:
    ax2 = axes[1]
    ax2.plot(bal_2, label="Run 2 (start $10.00)", color="darkorange",
             linestyle="--", alpha=0.85, linewidth=1.2)
    ax2.axhline(y=10.0,  color="gray",  linestyle=":",  alpha=0.5, label="Starting balance ($10)")   # FIX 5
    ax2.axhline(y=high_2, color="green", linestyle="--", alpha=0.4, label=f"Peak ${high_2:.2f}")
    ax2.axhline(y=low_2,  color="red",   linestyle="--", alpha=0.4, label=f"Trough ${low_2:.2f}")
    ax2.set_title(f"Run 2 (Bonus) — {'BANKRUPT' if bankrupt_2 else 'SURVIVED'}  |  "
                  f"Final ${final_2:.2f}  |  Max Drawdown {dd_2*100:.1f}%",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Bankroll ($)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

fig.suptitle("Transformer PPO — Real Market Survival Test", fontsize=14, fontweight="bold")
plt.xlabel("Candle Ticks (5m)")
plt.tight_layout()

# FIX 2: timestamped filename — no more silent overwrites
chart_path = f"{RESULT_DIR}/transformer_survival_{RUN_TIMESTAMP}.png"
plt.savefig(chart_path, dpi=150)
print(f"\nSaved Equity Visualization → {chart_path}")