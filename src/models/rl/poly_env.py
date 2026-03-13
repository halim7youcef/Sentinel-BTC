"""
poly_env.py
===========
Custom Gymnasium environment simulating Polymarket 5-minute BTC binary contracts.

Actions:
  0 — Hold (no bet placed)
  1 — Buy UP (predicting next 5m candle closes higher)
  2 — Buy DOWN (predicting next 5m candle closes lower)

Reward (unit-normalised for PPO stability):
  Win:  +( 1.0 - contract_price ) — net payout after cost
  Lose: -( contract_price )       — stake lost
  Hold: HOLD_PENALTY (-0.001)     — small cost to discourage degenerate always-hold

  + shaped_reward on all non-Hold actions (see FIX 6).

Balance tracking uses a configurable stake_usd per step (default $1) so that
evaluate_attention_rl.py can observe dollar-denominated bankroll changes via
info['balance'], while the PPO reward signal stays in [-0.72, +0.70] regardless
of stake size.

Changes from v1
---------------
  FIX 1 — Scaler leakage: _scaler was fitted on the full dataset (all rows),
           leaking test-set statistics into the normalised observation space.
           The scaler is now fitted on the first TRAIN_RATIO fraction only,
           matching the split used in build_rl_features.py.

  FIX 2 — Win/loss threshold: future_return > 0 accepted any micro-move
           (e.g. +0.00001%) as a win. Wins now require |future_return| >
           RETURN_THRESHOLD (0.15%) to match build_rl_features.py labels.
           Returns inside the dead-band resolve as PUSH (no loss, no win).

  FIX 3 — Spread noise direction: up_price and down_price now receive
           opposite-sign noise, simulating a realistic bid/ask spread.

  FIX 4 — Stake-aware balance: stake_usd parameter decouples dollar PnL
           (info['balance']) from the unit-normalised PPO reward signal.

  FIX 5 — Balance floor at $0: hard-floored to prevent negative bankroll.

  FIX 6 — Shaped reward (NEW — addresses entropy collapse observed at 118k
           steps where entropy_loss = -1.04):

           Problem: with RETURN_THRESHOLD = 0.0015 a large fraction of
           candles fall in the dead-band where every action returns exactly
           0. HOLD trivially dominates by variance, so the policy converges
           to always-HOLD with high confidence and stops exploring.

           Solution A — Continuous shaped component on non-Hold actions:
             shaped = SHAPE_COEF * clip(correct_direction * future_return,
                                        -SHAPE_CLIP, +SHAPE_CLIP)
           Gives the policy gradient a non-zero learning signal inside
           the dead-band. SHAPE_COEF=0.3 and SHAPE_CLIP=0.01 keep the
           shaped term well below the binary payout (~0.50) so it guides
           without distorting the true contract economics.

           Solution B — HOLD_PENALTY = -0.001 per step:
           Applies a tiny cost to every HOLD action so that doing nothing
           is no longer free. Forces the agent to engage with the market
           and discover whether directional bets have positive EV.

           NOTE: only binary_reward × stake_usd updates self.balance.
           The shaped component and hold penalty are PPO learning signals
           only and do not affect the dollar PnL reported in info['balance'].
"""

import os
import warnings
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# Matches LABEL_THRESHOLD in build_rl_features.py
RETURN_THRESHOLD = 0.0015   # 0.15%
TRAIN_RATIO      = 0.70     # Matches build_rl_features.py train split

# FIX 6 — Shaped reward config
SHAPE_COEF       = 0.3      # scales the continuous directional signal
SHAPE_CLIP       = 0.01     # clips future_return to prevent outlier domination
HOLD_PENALTY     = -0.001   # per-step cost for HOLD to prevent degenerate policy


class PolymarketBTCEnv(gym.Env):
    """
    Polymarket 5-minute BTC Up/Down binary contract simulator.

    Parameters
    ----------
    data_path : str
        Path to btc_features_rl.csv produced by build_rl_features.py.
    initial_balance : float
        Starting bankroll in USD (default $1,000).
    max_drawdown_pct : float
        Episode ends early if bankroll drops below this fraction of the
        initial balance (default 0.20). Pass 1.0 to allow full bankruptcy.
    spread_noise : float
        Std-dev of Gaussian noise on synthesized contract prices (default 0.01).
    stake_usd : float
        Dollar stake per non-Hold action for balance tracking (default $1.0).
        PPO reward is always unit-normalised regardless of this value.
    """

    metadata = {"render_modes": ["console"]}

    _NON_FEATURE_COLS = {
        "timestamp", "close", "target", "future_return",
        "gt_future_return", "target_raw", "open", "high", "low", "volume"
    }

    def __init__(
        self,
        data_path:        str,
        initial_balance:  float = 1000.0,
        max_drawdown_pct: float = 0.20,
        spread_noise:     float = 0.01,
        stake_usd:        float = 1.0,
    ):
        super().__init__()

        # ----------------------------------------------------------------
        # Load dataset
        # ----------------------------------------------------------------
        self.df = pd.read_csv(data_path)
        self.df = (
            self.df
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True)
        )

        required = {"close", "gt_future_return", "ml_prob_up"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"btc_features_rl.csv is missing required columns: {missing}. "
                "Re-run build_rl_features.py to regenerate the dataset."
            )

        self.feature_cols = [
            c for c in self.df.columns if c not in self._NON_FEATURE_COLS
        ]

        # FIX 1: train-only scaler fit
        train_end        = int(len(self.df) * TRAIN_RATIO)
        self._scaler     = RobustScaler()
        self._scaler.fit(self.df[self.feature_cols].iloc[:train_end].values)
        self._scaled_features = self._scaler.transform(
            self.df[self.feature_cols].values
        ).astype(np.float32)

        self.initial_balance  = float(initial_balance)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.spread_noise     = float(spread_noise)
        self.stake_usd        = float(stake_usd)
        self.min_balance      = self.initial_balance * (1.0 - self.max_drawdown_pct)

        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32,
        )

        self.balance      = self.initial_balance
        self.current_step = 0
        self.max_steps    = len(self.df) - 1
        self._rng         = np.random.default_rng()

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return self._scaled_features[self.current_step]

    def _get_dynamic_risk_premium(self) -> tuple[float, float]:
        """FIX 3: opposite-sign noise → realistic bid/ask spread."""
        ml_prob = float(self.df.loc[self.current_step, "ml_prob_up"])
        noise   = float(self._rng.normal(0.0, self.spread_noise))
        up_price   = float(np.clip(ml_prob       + noise + 0.01, 0.30, 0.72))
        down_price = float(np.clip(1.0 - ml_prob - noise + 0.01, 0.30, 0.72))
        return up_price, down_price

    def _shaped_reward(self, action: int, future_return: float) -> float:
        """
        FIX 6A: Continuous directional signal for dead-band candles.
        Returns 0.0 for HOLD — hold penalty is applied separately.
        """
        if action == 0:
            return 0.0
        correct_dir = 1.0 if action == 1 else -1.0
        return SHAPE_COEF * float(np.clip(correct_dir * future_return,
                                          -SHAPE_CLIP, SHAPE_CLIP))

    # ----------------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.balance      = self.initial_balance
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action: int):
        future_return        = float(self.df.loc[self.current_step, "gt_future_return"])
        up_price, down_price = self._get_dynamic_risk_premium()

        # ------------------------------------------------------------------
        # FIX 2: Binary payout with dead-band push
        # ------------------------------------------------------------------
        binary_reward = 0.0
        outcome       = "push"

        if action == 1:
            if future_return > RETURN_THRESHOLD:
                binary_reward = 1.0 - up_price;  outcome = "win"
            elif future_return < -RETURN_THRESHOLD:
                binary_reward = -up_price;        outcome = "loss"

        elif action == 2:
            if future_return < -RETURN_THRESHOLD:
                binary_reward = 1.0 - down_price; outcome = "win"
            elif future_return > RETURN_THRESHOLD:
                binary_reward = -down_price;       outcome = "loss"

        # ------------------------------------------------------------------
        # FIX 6: Shaped reward (PPO signal only — does not affect balance)
        # ------------------------------------------------------------------
        shaped    = self._shaped_reward(action, future_return)
        hold_pen  = HOLD_PENALTY if action == 0 else 0.0
        reward    = binary_reward + shaped + hold_pen

        # ------------------------------------------------------------------
        # FIX 4 + 5: Dollar balance update, floored at $0
        # ------------------------------------------------------------------
        if action != 0:
            self.balance += binary_reward * self.stake_usd
        self.balance = max(0.0, self.balance)

        self.current_step += 1

        terminated = (
            self.current_step >= self.max_steps
            or self.balance   <= self.min_balance
            or self.balance   == 0.0
        )

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(len(self.feature_cols), dtype=np.float32)
        )

        info = {
            "balance"          : self.balance,
            "reward"           : reward,
            "binary_reward"    : binary_reward,
            "shaped_component" : shaped,
            "action"           : int(action),
            "outcome"          : outcome,
            "up_price"         : up_price,
            "down_price"       : down_price,
            "future_ret"       : future_return,
        }

        return obs, reward, terminated, False, info

    def render(self):
        print(f"Step {self.current_step:>6} | Balance: ${self.balance:>9.2f} | "
              f"Stake: ${self.stake_usd:.2f}")

    def close(self):
        pass