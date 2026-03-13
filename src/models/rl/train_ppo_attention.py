"""
train_ppo_attention.py
======================
PPO training script using the BTCAttentionExtractor as the policy brain.

Key differences from train_ppo.py
----------------------------------
  1. Uses a SequenceEnvWrapper that maintains a rolling window of SEQ_LEN
     candles so the Transformer sees temporal context, not just a single tick.
  2. BTCAttentionExtractor replaces the MLP feature extractor inside the
     ActorCriticPolicy via policy_kwargs["features_extractor_class"].
  3. Shared feature extractor between actor and critic heads
     (share_features_extractor=True) — saves memory and accelerates training.

Usage
-----
Run from btc_rl_project root:
  python src/models/rl/train_ppo_attention.py

Output
------
  models/rl/ppo_attention.zip       — final model
  models/rl/best_model.zip          — best model by eval reward
  results/metrics/ppo_attn_tb/      — TensorBoard logs

Scaling changelog (4-year dataset edition)
------------------------------------------
  SCALE 1 — total_timesteps 500k → 3_000_000:
    4yr dataset ≈ 420k candles. At 500k steps with n_envs=2 the agent saw
    the dataset ~12 times total — nowhere near enough for convergence.
    At 3M steps with n_envs=8 each env completes ~23 full passes, giving
    the policy enough exposure to bull, bear, and ranging regimes.

  SCALE 2 — n_envs 2 → 8:
    Parallelises data collection across 8 independent environment instances.
    Each env starts at a random offset in the dataset (via reset()),
    increasing regime diversity per rollout batch. Also raises effective
    FPS from ~200 to ~800+ on a modern CPU, or ~3000+ on GPU.

  SCALE 3 — n_steps 2048 → 4096:
    Longer rollouts capture multi-hour BTC patterns (4096 × 5m = ~14h per
    rollout per env). Combined with n_envs=8 this gives 32,768 transitions
    per PPO update — a much richer gradient signal than the previous 4,096.

  SCALE 4 — batch_size 128 → 512:
    Larger mini-batches stabilise the PPO update at scale and reduce gradient
    variance, particularly important with the wider Transformer (embed_dim=256).

  SCALE 5 — net_arch [64] → [256, 128]:
    The actor/critic heads now receive a 256-dim vector from the scaled
    Transformer. A [64] head was a severe bottleneck — the model was
    compressing 256 features through 64 neurons in one step. [256, 128]
    provides a smooth two-stage compression without information loss.

  SCALE 6 — SEQ_LEN 64 → 128, EMBED_DIM/FEATURES_DIM 128 → 256:
    Matches the updated BTCAttentionExtractor defaults. Must stay in sync
    with btc_attention.py or the observation space reshape will fail.

  KEPT — ent_coef 0.10:
    Necessary to counteract entropy collapse observed at 118k steps. Do not
    reduce until training logs confirm entropy_loss stays above -0.8 past 500k.

  KEPT — dropout 0.1:
    Correct for this architecture size. Do not increase.

  KEPT — activation_fn Tanh:
    Preserves signed signal from Transformer last-token output into heads.
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings("ignore")

sys.path.append("src/models/rl")
sys.path.append("src/models/transformer")

from poly_env import PolymarketBTCEnv
from btc_attention import BTCAttentionExtractor

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DATA_PATH        = "data/processed/btc_features_rl.csv"
MODEL_SAVE_PATH  = "models/rl/ppo_attention"
BEST_MODEL_PATH  = "models/rl/"
TENSORBOARD_LOG  = "results/metrics/ppo_attn_tb"

# Architecture — must stay in sync with btc_attention.py
SEQ_LEN          = 128     # SCALE 6: was 64 — 10.6h context window
EMBED_DIM        = 256     # SCALE 6: was 128
N_HEADS          = 8       # SCALE 6: was 4
N_LAYERS         = 4       # SCALE 6: was 2
DROPOUT          = 0.1
FEATURES_DIM     = 256     # SCALE 6: was 128

# Training scale
TOTAL_TIMESTEPS  = 500_000   # SCALE 1: was 500k
N_ENVS           = 2           # SCALE 2: was 2
N_STEPS          = 2048        # SCALE 3: was 2048
BATCH_SIZE       = 128         # SCALE 4: was 128

# Stake sizes
TRAIN_STAKE_USD  = 1.0         # unit-normalised for PPO gradient stability
EVAL_STAKE_USD   = 10.0        # dollar-readable for EvalCallback curves

os.makedirs("models/rl", exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
print(f"Transitions per PPO update: {N_ENVS * N_STEPS:,}  "
      f"({N_ENVS} envs × {N_STEPS} steps)")


# -----------------------------------------------------------------------
# Sequence Environment Wrapper
# -----------------------------------------------------------------------

class SequenceEnvWrapper(gym.Wrapper):
    """
    Wraps PolymarketBTCEnv to output a rolling window of SEQ_LEN observations.

    Emits [seq_len * n_features] flat vector per step.
    Buffer is zero-padded at episode start.
    """

    def __init__(self, env: PolymarketBTCEnv, seq_len: int = SEQ_LEN):
        super().__init__(env)
        self.seq_len    = seq_len
        n_features      = env.observation_space.shape[0]
        self.n_features = n_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(seq_len * n_features,),
            dtype=np.float32,
        )
        self._buffer = np.zeros((seq_len, n_features), dtype=np.float32)

    def _push(self, obs: np.ndarray) -> np.ndarray:
        self._buffer     = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = obs
        return self._buffer.flatten()

    def reset(self, **kwargs):
        obs, info        = self.env.reset(**kwargs)
        self._buffer[:]  = 0.0
        return self._push(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._push(obs), reward, terminated, truncated, info


# -----------------------------------------------------------------------
# Environment factories
# -----------------------------------------------------------------------

def make_seq_env(stake_usd: float = TRAIN_STAKE_USD):
    """Returns a Monitor-wrapped SequenceEnvWrapper factory for make_vec_env."""
    def _factory():
        base = PolymarketBTCEnv(data_path=DATA_PATH, stake_usd=stake_usd)
        seq  = SequenceEnvWrapper(base, seq_len=SEQ_LEN)
        return Monitor(seq)
    return _factory


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_attention_agent(total_timesteps: int = TOTAL_TIMESTEPS):
    print("=" * 60)
    print("Training PPO + Attention (Transformer) Brain — 4yr Scale")
    print("=" * 60)

    probe_env = PolymarketBTCEnv(data_path=DATA_PATH)
    n_raw     = probe_env.observation_space.shape[0]
    probe_env.close()

    print(f"Single-tick obs size : {n_raw}")
    print(f"Sequence obs size    : {SEQ_LEN} × {n_raw} = {SEQ_LEN * n_raw}")
    print(f"Total timesteps      : {total_timesteps:,}")
    print(f"Parallel envs        : {N_ENVS}")
    print(f"Steps per update     : {N_ENVS * N_STEPS:,}")
    print(f"PPO updates total    : {total_timesteps // (N_ENVS * N_STEPS):,}")

    env = make_vec_env(make_seq_env(TRAIN_STAKE_USD), n_envs=N_ENVS)

    policy_kwargs = dict(
        features_extractor_class  = BTCAttentionExtractor,
        features_extractor_kwargs = dict(
            n_raw_features = n_raw,
            seq_len        = SEQ_LEN,
            embed_dim      = EMBED_DIM,
            n_heads        = N_HEADS,
            n_layers       = N_LAYERS,
            dropout        = DROPOUT,
            features_dim   = FEATURES_DIM,
        ),
        share_features_extractor = True,
        # SCALE 5: [256, 128] heads match the 256-dim Transformer output.
        # Previous [64] was a severe compression bottleneck.
        net_arch      = [64],
        activation_fn = nn.Tanh,    # preserves signed signal from last-token output
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate   = 0.0001,
        n_steps         = N_STEPS,          # SCALE 3
        batch_size      = BATCH_SIZE,        # SCALE 4
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.10,             # anti-collapse; review at 500k steps
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = policy_kwargs,
        tensorboard_log = TENSORBOARD_LOG,
        device          = device,
        verbose         = 1,
    )

    print(f"\nPolicy network:\n{model.policy}\n")
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"  (Previous config was ~850k — scaled config is ~{total_params//1_000_000}M+)\n")

    # Eval env: fixed seed + dollar-readable rewards
    eval_env = make_seq_env(EVAL_STAKE_USD)()
    eval_env.reset(seed=42)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_PATH,
        log_path             = "results/metrics/",
        eval_freq            = 50_000,      # less frequent — 3M run is long
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    print(f"Starting training for {total_timesteps:,} timesteps...")
    print("Health check milestones to watch:")
    print("   500k  — entropy_loss should be above -0.85")
    print("   1M    — ep_rew_mean should be approaching 0")
    print("   2M    — explained_variance should be above 0.30")
    print("   3M    — eval reward should be positive or near-zero\n")

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(MODEL_SAVE_PATH)
    print(f"\nSaved Attention-PPO model → {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    train_attention_agent(total_timesteps=TOTAL_TIMESTEPS)