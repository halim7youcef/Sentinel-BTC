import os
import sys
import argparse
import warnings

# Suppress warnings temporarily for cleaner training logs
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn

# Explicitly use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PPO Training on device: {device}")

# We import the custom env we built
sys.path.append('src/models/rl')
from poly_env import PolymarketBTCEnv

DATA_PATH = "data/processed/btc_features_ml.csv"
PRED_PATH = "results/predictions/ensemble_predictions.csv"
MODEL_SAVE_PATH = "models/rl/ppo_polymarket"
TENSORBOARD_LOG = "results/metrics/ppo_tensorboard"

os.makedirs("models/rl", exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

def make_env():
    # Wrap with Monitor so stable-baselines3 stops complaining about the EvalEnv missing rewards schema
    return Monitor(PolymarketBTCEnv(data_path=DATA_PATH, pred_path=PRED_PATH))

def train_agent(timesteps=500000):
    print("Initializing Polymarket BTC Environment...")
    # Wrap in Vectorized Env for faster Parallel execution
    env = make_vec_env(make_env, n_envs=4)

    # Policy Kwargs dict: defining a deep Multi-Layer Perceptron (MLP) architecture
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # Initialize PPO Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99, # Discount factor for predicting long-term sequences
        gae_lambda=0.95,
        clip_range=0.2, # Limits policy explosions
        ent_coef=0.02, # Forces agent to explore bad actions early on 
        policy_kwargs=policy_kwargs,
        tensorboard_log=TENSORBOARD_LOG,
        device=device,
        verbose=1
    )

    print(f"Beginning PPO Training for {timesteps} Timesteps...")
    
    # EvalCallback allows capturing the best model snapshot during training
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/rl/',
        log_path='results/metrics/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Train (Default log interval prints frequently to terminal)
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    
    # Save Final
    model.save(MODEL_SAVE_PATH)
    print(f"\nSaved PPO model to {MODEL_SAVE_PATH}.zip")

if __name__ == "__main__":
    train_agent(timesteps=500000)
